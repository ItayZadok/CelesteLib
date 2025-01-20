import pygame
import os
import numpy as np
import wave
import scipy.fftpack as fft


def print_grouped_data(data, label="db"):
    line = []
    line_length = 0
    i = 0
    while i < len(data):
        value = data[i]
        count = 1
        while i + count < len(data) and data[i + count] == value:
            count += 1
        if count > 3:
            formatted = f"{count} dup({value}),"
        else:
            formatted = ",".join(map(str, [value] * count)) + ","
        if line_length + len(formatted) > 100:
            print(f"{label} {''.join(line)[:-1]}")
            line = []
            line_length = 0
        line.append(formatted)
        line_length += len(formatted)
        i += count
    if line:
        print(f"{label} {''.join(line)[:-1]}")


def build_palette(images_dir):
    colors = set()
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            image = pygame.image.load(os.path.join(images_dir, file)).convert_alpha()
            width, height = image.get_size()
            for y in range(height):
                for x in range(width):
                    color = image.get_at((x, y))[:3]
                    colors.add(color)
    palette = list(colors)
    print("palette", end=" ")
    palette_data = []
    for rgb in palette:
        palette_data.extend(rgb_to_63(rgb))
    print_grouped_data(palette_data)
    if len(palette) < 256:
        filler = [0] * ((256 - len(palette)) * 4)
        print_grouped_data(filler)
    return palette


def print_image(name, image, palette):
    print('\n')
    width, height = image.get_size()
    indices = []
    for y in range(height):
        for x in range(width):
            color = image.get_at((x, y))
            if color[3] == 0:
                indices.append(-1)
            else:
                rgb = color[:3]
                if rgb in palette:
                    indices.append(palette.index(rgb))
    print(f"{name} dw {width}, {height}")
    print_grouped_data(indices)


def process_images(images_dir, palette):
    images = []
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            image = pygame.image.load(os.path.join(images_dir, file)).convert_alpha()
            name = os.path.splitext(file)[0]
            print_image(name, image, palette)
            images.append((name, image))


def rgb_to_63(rgb):
    r, g, b = rgb
    r_63 = r // 4
    g_63 = g // 4
    b_63 = b // 4
    return r_63, g_63, b_63


def extract_frequencies_from_wav(filename, n_fft=2048):
    """
    Extracts a list of dominant frequencies from a wav file and prints them in the required format.

    Args:
        filename (str): Path to the input .wav file.
        n_fft (int): Number of FFT points (default: 2048).
    """
    hop_length = n_fft // 4
    # Open the WAV file
    with wave.open(filename, 'r') as wav_file:
        # Extract audio properties
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Read the audio frames
        frames = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)

        # If stereo, average channels into mono
        if wav_file.getnchannels() > 1:
            audio_data = np.mean(audio_data.reshape(-1, wav_file.getnchannels()), axis=1)

    # Split the audio into frames for FFT
    notes = []
    for i in range(0, len(audio_data) - n_fft, hop_length):
        frame = audio_data[i:i + n_fft]
        frame = frame * np.hanning(len(frame))  # Apply Hanning window
        fft_result = np.abs(fft.fft(frame))[:n_fft // 2]
        dominant_frequency = np.argmax(fft_result) * framerate / n_fft

        if dominant_frequency > 10:  # Ignore edge noise frequencies
            note_value = int(1193180 / dominant_frequency)
        else:
            note_value = 0
        notes.append(note_value)

    # Return extracted frequencies
    return notes


def main():
    pygame.init()

    screen = pygame.display.set_mode((320, 200))
    images_dir = 'images'
    palette = build_palette(images_dir)
    process_images(images_dir, palette)

    filename = 'super-mario-bros-theme-song.wav'  # Replace with the path to your .wav file
    notes = extract_frequencies_from_wav(filename)

    pygame.quit()


if __name__ == "__main__":
    main()
