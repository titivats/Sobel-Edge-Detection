from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "assets"
PNG_PATH = ASSET_DIR / "edge_detection_monitor.png"
ICO_PATH = ASSET_DIR / "edge_detection_monitor.ico"


def draw_icon(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for y in range(size):
        t = y / max(size - 1, 1)
        r = int(13 + 11 * t)
        g = int(26 + 20 * t)
        b = int(46 + 38 * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    margin = int(size * 0.1)
    screen = [margin, int(size * 0.17), size - margin, int(size * 0.75)]
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle(screen, radius=int(size * 0.07), outline=(56, 189, 248, 200), width=int(size * 0.035))
    image = Image.alpha_composite(image, glow.filter(ImageFilter.GaussianBlur(int(size * 0.018))))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(screen, radius=int(size * 0.07), fill=(15, 23, 42, 255), outline=(125, 211, 252, 255), width=int(size * 0.018))
    inner = [screen[0] + int(size * 0.045), screen[1] + int(size * 0.055), screen[2] - int(size * 0.045), screen[3] - int(size * 0.055)]
    draw.rectangle(inner, fill=(4, 8, 18, 255))

    # Edge trace and bounding boxes.
    edge_points = [
        (inner[0] + int(size * 0.06), inner[3] - int(size * 0.09)),
        (inner[0] + int(size * 0.18), inner[1] + int(size * 0.12)),
        (inner[0] + int(size * 0.31), inner[3] - int(size * 0.14)),
        (inner[0] + int(size * 0.45), inner[1] + int(size * 0.17)),
        (inner[0] + int(size * 0.62), inner[3] - int(size * 0.12)),
        (inner[0] + int(size * 0.78), inner[1] + int(size * 0.11)),
    ]
    draw.line(edge_points, fill=(14, 165, 233, 255), width=int(size * 0.022), joint="curve")
    draw.line([(x, y + int(size * 0.045)) for x, y in edge_points], fill=(34, 197, 94, 220), width=int(size * 0.011), joint="curve")

    box_w = int(size * 0.2)
    box_h = int(size * 0.16)
    draw.rectangle(
        [inner[0] + int(size * 0.11), inner[1] + int(size * 0.15), inner[0] + int(size * 0.11) + box_w, inner[1] + int(size * 0.15) + box_h],
        outline=(34, 197, 94, 255),
        width=int(size * 0.012),
    )
    draw.rectangle(
        [inner[2] - int(size * 0.14) - box_w, inner[2] * 0 + inner[1] + int(size * 0.22), inner[2] - int(size * 0.14), inner[1] + int(size * 0.22) + box_h],
        outline=(239, 68, 68, 255),
        width=int(size * 0.012),
    )

    stand_y = int(size * 0.79)
    draw.rounded_rectangle([int(size * 0.45), screen[3], int(size * 0.55), stand_y], radius=int(size * 0.025), fill=(71, 85, 105, 255))
    draw.rounded_rectangle([int(size * 0.32), stand_y, int(size * 0.68), int(size * 0.86)], radius=int(size * 0.035), fill=(51, 65, 85, 255))

    lens_center = (int(size * 0.74), int(size * 0.73))
    lens_radius = int(size * 0.12)
    draw.ellipse(
        [
            lens_center[0] - lens_radius,
            lens_center[1] - lens_radius,
            lens_center[0] + lens_radius,
            lens_center[1] + lens_radius,
        ],
        fill=(8, 47, 73, 245),
        outline=(125, 211, 252, 255),
        width=int(size * 0.014),
    )
    draw.line(
        [
            (lens_center[0] + int(size * 0.08), lens_center[1] + int(size * 0.08)),
            (lens_center[0] + int(size * 0.2), lens_center[1] + int(size * 0.2)),
        ],
        fill=(125, 211, 252, 255),
        width=int(size * 0.035),
    )
    draw.ellipse(
        [
            lens_center[0] - int(size * 0.045),
            lens_center[1] - int(size * 0.045),
            lens_center[0] + int(size * 0.045),
            lens_center[1] + int(size * 0.045),
        ],
        fill=(56, 189, 248, 210),
    )

    return image


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    image = draw_icon()
    image.save(PNG_PATH)
    image.save(ICO_PATH, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Created {PNG_PATH}")
    print(f"Created {ICO_PATH}")


if __name__ == "__main__":
    main()
