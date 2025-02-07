import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image


def print_grouped_data(data, label="db"):
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
            print(f"{label} {''.join(line)[:-1]}")
            line = []
            line_length = 0

        line.append(formatted)
        line_length += len(formatted)
        i += count

    if line: print(f"{label} {''.join(line)[:-1]}")


def quantize_colors(image, n_colors=256):
    pixels = list(image.getdata())

    # Filter out transparent pixels (if any)
    pixels = [(r, g, b) for r, g, b, a in pixels if a > 0]  # Skip fully transparent

    # If there are not enough pixels, adjust the number of clusters
    if len(pixels) < n_colors:
        n_colors = len(pixels)

    # Perform MiniBatchKMeans on the reduced set of pixels
    if len(pixels) > 1:  # Ensure there are enough samples for clustering
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0, batch_size=128).fit(pixels)
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    else:
        # Return the pixels directly if there are not enough for clustering
        return pixels


def print_image(name, image, palette):
    print('')
    width, height = image.size
    indices = []

    # Convert the image to a NumPy array for faster processing
    image_array = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = image_array[y, x]  # Extract alpha as well

            # If pixel is fully transparent, use -1
            if a == 0:
                indices.append(-1)
            else:
                # Find the closest color in the palette
                distances = np.linalg.norm(np.array(palette) - np.array((r, g, b)), axis=1)
                closest_index = np.argmin(distances)
                indices.append(closest_index)

    print(f"{name} dw {width}, {height}")
    print_grouped_data(indices)


def build_palette(images_dir):
    colors = set()
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            image = Image.open(os.path.join(images_dir, file)).convert("RGBA")
            reduced_palette = quantize_colors(image)
            colors.update(reduced_palette)

    # Add pure black as the first color (if it's not already in the palette)
    if (0, 0, 0) not in colors:
        colors.add((0, 0, 0))

    # Convert the set of colors to a list and ensure it contains no more than 256 colors
    palette = list(colors)[:256]

    # Ensure pure black is the first color in the palette
    if (0, 0, 0) in palette:
        palette.remove((0, 0, 0))  # Remove if already in the list
    palette.insert(0, (0, 0, 0))  # Insert pure black at the start

    print("palette", end=" ")
    palette_data = []
    for rgb in palette: palette_data.extend(rgb_to_63(rgb))
    print_grouped_data(palette_data)
    return palette


def process_images(images_dir, palette):
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            image = Image.open(os.path.join(images_dir, file)).convert("RGBA")
            name = os.path.splitext(file)[0]
            print_image(name, image, palette)


def rgb_to_63(rgb):
    r, g, b = rgb
    return r // 4, g // 4, b // 4


def main():
    images_dir = 'images'
    palette = build_palette(images_dir)
    process_images(images_dir, palette)


if __name__ == "__main__":
    main()
