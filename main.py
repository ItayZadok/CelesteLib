import os
import wave

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

CHUNK_MS = 50  # The size of each chunk for FFT in milliseconds
PIT_FREQUENCY = 440  # Reference pitch for frequency calculation (A4)
FREQ_THRESHOLD = 0.1  # Frequency change threshold
MIN_FREQUENCY = 60  # Minimum frequency to consider (for musical notes)


def get_frequencies_and_duration(filename):
    with wave.open(filename, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()

        # Read audio samples
        raw_audio = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)

        # Convert to mono if stereo
        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

        # Define chunk size for FFT analysis
        chunk_size = int(frame_rate * (CHUNK_MS / 1000))  # Convert ms to samples
        frequencies = []
        total_time = 0
        note_changes = 0

        last_freq = None
        last_time = 0  # Start time of current note

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]

            # Apply FFT to find dominant frequency
            fft_spectrum = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(len(chunk), d=1 / frame_rate)
            magnitude = np.abs(fft_spectrum)

            # Find peak frequency, ignoring very low values
            peak_idx = np.argmax(magnitude[1:]) + 1  # Ignore DC component (index 0)
            peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0

            if peak_freq > MIN_FREQUENCY:  # Ignore inaudible low frequencies
                pit_value = int(PIT_FREQUENCY / peak_freq)

                # Check if frequency has changed significantly
                if last_freq is None or abs(pit_value - last_freq) > (last_freq * (FREQ_THRESHOLD - 1)):
                    if last_freq is not None:
                        total_time += (i / frame_rate * 1000) - last_time  # Time in ms
                        note_changes += 1

                    # Update current note
                    last_freq = pit_value
                    last_time = i / frame_rate * 1000  # Convert to ms

                    frequencies.append(pit_value)

        # Calculate the average duration per note
        duration = total_time / note_changes if note_changes > 0 else CHUNK_MS

        return frequencies, round(duration)


def write_grouped_data(file, data, label="db"):
    line = []
    line_length = 0
    i = 0
    while i < len(data):
        value = data[i]
        count = 1

        while i + count < len(data) and data[i + count] == value:
            count += 1

        # Use dup for sequences of 4 or more, otherwise list them normally
        if count >= 4:
            formatted = f"{count} dup({value}),"
        else:
            formatted = ",".join(map(str, [value] * count)) + ","

        # Ensure line length doesn't exceed 200 characters
        if line_length + len(formatted) > 200:
            file.write(f"{label} {''.join(line)[:-1]}\n")
            line = []
            line_length = 0

        line.append(formatted)
        line_length += len(formatted)
        i += count

    if line: file.write(f"{label} {''.join(line)[:-1]}")
    file.write('\n')


def quantize_colors(image, n_colors=256):
    pixels = list(image.getdata())

    # Filter out transparent pixels (if any)
    pixels = [(r, g, b) for r, g, b, a in pixels if a > 0]

    # Adjust cluster count if not enough colors
    n_colors = min(n_colors, len(pixels))

    if len(pixels) > 1:
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0, batch_size=128).fit(pixels)
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    else:
        return pixels


def write_image_data(file, name, image, palette):
    width, height = image.size
    indices = []

    # Convert the image to a NumPy array for faster processing
    image_array = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = image_array[y, x]
            if a == 0:  # Fully transparent pixel
                indices.append(-1)
            else:
                # Find the closest color in the palette
                distances = np.linalg.norm(np.array(palette) - np.array((r, g, b)), axis=1)
                indices.append(np.argmin(distances))

    file.write(f"\n{name} db ImageId")
    file.write(f"\ndw {width}, {height}\n")
    write_grouped_data(file, indices)


def build_palette(images_dir, file):
    colors = set()
    for file_name in os.listdir(images_dir):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(os.path.join(images_dir, file_name)).convert("RGBA")
            reduced_palette = quantize_colors(image)
            colors.update(reduced_palette)

    color_list = list(colors)

    if len(color_list) > 1:
        kmeans = MiniBatchKMeans(n_clusters=min(256, len(color_list)), random_state=0, batch_size=128).fit(color_list)
        color_list = [tuple(map(int, color)) for color in kmeans.cluster_centers_]

    palette = color_list[:256]

    file.write(f"palette dw {len(palette)}\n")
    palette_data = []
    for rgb in palette:
        palette_data.extend(rgb_to_63(rgb))
    write_grouped_data(file, palette_data)

    return palette


def process_images(images_dir, palette, file):
    for file_name in os.listdir(images_dir):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(os.path.join(images_dir, file_name)).convert("RGBA")
            name = os.path.splitext(file_name)[0]
            write_image_data(file, name, image, palette)


def rgb_to_63(rgb):
    r, g, b = rgb
    return r // 4, g // 4, b // 4


def write_sound_data(name, file, notes, durations):
    file.write(f"{name} dw {len(notes)}, {durations}\n")
    write_grouped_data(file, notes, label='dw')


class Component:
    def __init__(self, name):
        self.name = name

    def ask_questions(self):
        return 0, []


class RenderComponent(Component):
    def __init__(self):
        super().__init__("Render")

    def ask_questions(self):
        width = int(input("Enter width: "))
        height = int(input("Enter height: "))
        return [(
                f"dw 0, 0, offset IMAGE, seg IMAGE, 0, 0, 0, {width}, {height} ; x, y, image pointer, prev x, prev y, rotation, width, height\n"
                + f"db {width}*{height} dup(0) ; background save place"
        )]


class PhysicsComponent(Component):
    def __init__(self):
        super().__init__("Physics")

    def ask_questions(self):
        return [(
                "dw 0, 0, 0, MAX_X  ; vx, ax, fx, mx\n" +
                "dw 0, 0, 0, MAX_Y  ; vy, ay, fy, my"
        )]


class MouseComponent(Component):
    def __init__(self):
        super().__init__("Mouse")

    def ask_questions(self):
        return [(
            f"db 0, 0, 0, 0; left pressed, right pressed, left clicked, right clicked"
        )]


class AnimatorComponent(Component):
    def __init__(self):
        super().__init__("Animator")

    def ask_questions(self):
        animation_amount = int(input("Enter amount of Animations: "))
        max_animation_amount = max(10, animation_amount)
        return [(
                f"db 0, 0, {animation_amount} ; cur anim, cur frame, anim amount\n"
                + "dw " + "offset ANIMATION, seg ANIMATION " * animation_amount
                + f"\ndw {max_animation_amount - animation_amount} dup(0)"
        )]


COMPONENT_TYPES = {
    "Render": RenderComponent,
    "Physics": PhysicsComponent,
    "Animator": AnimatorComponent,
    "Mouse": MouseComponent
}


def create_controller():
    name = input("Enter controller name: ")
    max_amount = input("Enter max amount of components (default 10): ")
    max_amount = int(max_amount) if max_amount.isdigit() else 10

    components = []

    while True:
        comp_name = input(f"Enter component name {COMPONENT_TYPES.keys()} or press Enter to finish: ")
        if not comp_name:
            break

        if comp_name in COMPONENT_TYPES:

            component = COMPONENT_TYPES[comp_name]()
            output = component.ask_questions()
            components.append((component, output))
            print("Component added!")
        else:
            print("Invalid component type. Try again.")

    output_lines = [f"{name} db {max_amount}"]
    for component, output in components:
        output_lines.append(f"db {component.name}ComponentId"
                            f"\ndw offset {name + component.name}, seg {name + component.name}")

    padding = max_amount - len(components)
    if padding > 0:
        output_lines.append(f"db {padding * 5} dup(0)")

    for component, component_output in components:
        component_output[0] = name + component.name + " " + component_output[0]
        output_lines.extend(component_output)

    print("\n".join(output_lines))


def handle_images():
    images_dir = 'images'
    image_output_file = 'image_output.text'
    with open(image_output_file, "w") as file:
        palette = build_palette(images_dir, file)
        process_images(images_dir, palette, file)
    print('Success! check image result in ' + image_output_file)


def handle_stacked_sprites(spriteName="stackedSprite", num_rotations=16, height=1):
    images_dir = 'stackedSprites'
    image_output_dir = 'stacked_sprite_output'
    image_output_file = 'stacked_sprite_output.text'

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Clear output directory
    for filename in os.listdir(image_output_dir):
        file_path = os.path.join(image_output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Determine max size across all rotations
    max_width, max_height = 0, 0
    for image_file in image_files:
        img = Image.open(os.path.join(images_dir, image_file))
        for i in range(num_rotations):
            angle = (360 / num_rotations) * i
            rotated_img = img.rotate(angle, expand=True)
            max_width = max(max_width, rotated_img.width)
            max_height = max(max_height, rotated_img.height)

    # Increase canvas height to fit all stacked layers
    stacked_height = max_height + len(image_files) - height

    array_data = []

    # Process and center each rotation
    for i in range(num_rotations):
        angle = (360 / num_rotations) * i
        canvas = Image.new("RGBA", (max_width, stacked_height), (0, 0, 0, 0))

        for layer_index, image_file in enumerate(image_files):
            img = Image.open(os.path.join(images_dir, image_file))
            rotated_img = img.rotate(angle, expand=True)

            # Calculate position: centered horizontally, stacked vertically
            paste_x = (max_width - rotated_img.width) // 2
            paste_y = (stacked_height - rotated_img.height) // 2 - layer_index  # 1-pixel shift per layer

            # Paste image onto canvas
            canvas.paste(rotated_img, (paste_x, paste_y), rotated_img)

        # Save stacked image
        name = f"{spriteName}Rotation{i}"
        array_data.append(name)
        output_path = os.path.join(image_output_dir, name + '.png')
        canvas.save(output_path)

    with open(image_output_file, "w") as file:
        palette = build_palette(images_dir, file)
        process_images(image_output_dir, palette, file)
        file.write(f"\n{spriteName}Rotations db ArrayId\ndw {num_rotations}, 2\n")
        write_grouped_data(file, array_data, "dw")

    print("Success! Check images in", image_output_dir)


def handle_sounds():
    sound_output_file = 'sound_output.text'
    frequencies, durations = get_frequencies_and_duration("sounds/super-mario-bros-theme-song.wav")
    with open(sound_output_file, "w") as file:
        write_sound_data('notes', file, frequencies, durations)
    print('Success! check sound result in ' + sound_output_file)


def main():
    # handle_images()
    # handle_sounds()
    # handle_stacked_sprites('car')

    create_controller()


if __name__ == "__main__":
    main()
