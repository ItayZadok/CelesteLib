import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image


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

    file.write(f"\n{name} dw {width}, {height}\n")
    write_grouped_data(file, indices, label="db")


def build_palette(images_dir, file):
    colors = set()
    for file_name in os.listdir(images_dir):
        if file_name.endswith('.png'):
            image = Image.open(os.path.join(images_dir, file_name)).convert("RGBA")
            reduced_palette = quantize_colors(image)
            colors.update(reduced_palette)

    # Ensure black is first in the palette
    colors.add((0, 0, 0))
    palette = list(colors)[:256]
    palette.sort()

    file.write("palette ")
    palette_data = []
    for rgb in palette:
        palette_data.extend(rgb_to_63(rgb))
    write_grouped_data(file, palette_data, label="db")

    return palette


def process_images(images_dir, palette, file):
    for file_name in os.listdir(images_dir):
        if file_name.endswith('.png'):
            image = Image.open(os.path.join(images_dir, file_name)).convert("RGBA")
            name = os.path.splitext(file_name)[0]
            write_image_data(file, name, image, palette)


def rgb_to_63(rgb):
    r, g, b = rgb
    return r // 4, g // 4, b // 4


def main():
    images_dir = 'images'
    with open("image_output.text", "w") as file:
        palette = build_palette(images_dir, file)
        process_images(images_dir, palette, file)


if __name__ == "__main__":
    main()
