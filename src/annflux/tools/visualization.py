import colorsys
import numpy as np


def brighten_hex_color(hex_color, factor=0.2):
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0–255 range)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Normalize RGB to 0–1 range
    r, g, b = [x / 255.0 for x in (r, g, b)]

    # Convert to HLS (Hue, Lightness, Saturation)
    h, l_, s = colorsys.rgb_to_hls(r, g, b)

    # Increase lightness by given factor (capped at 1.0)
    l_ = min(1.0, l_ + (l_ * factor))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l_, s)
    r, g, b = [int(x * 255) for x in (r, g, b)]

    average_brightness = int(255 - np.mean([r, g, b]))
    # Convert to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def most_contrasting_gray(hex_color: str) -> str:
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]

    # Calculate perceived brightness (simple average method)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    # The most contrasting gray is the one furthest from the brightness of the input color
    # If the color is bright, choose a dark gray; if it's dark, choose a light gray
    # We'll select from 0 (black) to 255 (white)
    contrast_gray = 0 if brightness > 128 else 255

    # Optional: Allow mid-gray (like #888888) if it's closer to optimal contrast
    # Here we could return a range of grayscale colors, but let's return the one that maximizes contrast
    return "#{0:02x}{0:02x}{0:02x}".format(contrast_gray)
