#!/usr/bin/env python3

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "assets" / "icons"
ICO_PATH = ICON_DIR / "linuxdo.ico"
PNG_SIZES = [32, 48, 64, 72, 80, 96, 112, 120, 128, 144, 160, 192, 224, 256, 512]
ICO_SIZES = [
    (16, 16),
    (20, 20),
    (24, 24),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
    (72, 72),
    (80, 80),
    (96, 96),
    (112, 112),
    (120, 120),
    (128, 128),
    (144, 144),
    (160, 160),
    (192, 192),
    (224, 224),
    (256, 256),
]
MASTER_SIZE = 4096


def build_master() -> Image.Image:
    image = Image.new("RGBA", (MASTER_SIZE, MASTER_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    cx = cy = MASTER_SIZE / 2
    outer_r = MASTER_SIZE * 0.476
    inner_r = MASTER_SIZE * 0.438

    outer_box = (
        int(cx - outer_r),
        int(cy - outer_r),
        int(cx + outer_r),
        int(cy + outer_r),
    )
    inner_box = (
        int(cx - inner_r),
        int(cy - inner_r),
        int(cx + inner_r),
        int(cy + inner_r),
    )

    draw.ellipse(outer_box, fill=(255, 255, 255, 255))

    inner = Image.new("RGBA", (MASTER_SIZE, MASTER_SIZE), (0, 0, 0, 0))
    inner_draw = ImageDraw.Draw(inner)
    inner_draw.rectangle((0, 0, MASTER_SIZE, int(MASTER_SIZE * 0.305)), fill=(30, 30, 34, 255))
    inner_draw.rectangle(
        (0, int(MASTER_SIZE * 0.305), MASTER_SIZE, int(MASTER_SIZE * 0.695)),
        fill=(245, 245, 245, 255),
    )
    inner_draw.rectangle((0, int(MASTER_SIZE * 0.695), MASTER_SIZE, MASTER_SIZE), fill=(247, 173, 26, 255))

    mask = Image.new("L", (MASTER_SIZE, MASTER_SIZE), 0)
    ImageDraw.Draw(mask).ellipse(inner_box, fill=255)
    image.alpha_composite(Image.composite(inner, Image.new("RGBA", inner.size, (0, 0, 0, 0)), mask))
    return image


def render_icon(base: Image.Image, size: int) -> Image.Image:
    icon = base.resize((size, size), Image.Resampling.LANCZOS)
    radius = 0.8 if size <= 24 else 1.0 if size <= 48 else 1.2
    percent = 165 if size <= 24 else 150 if size <= 48 else 120
    icon = icon.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    return icon


def main() -> None:
    master = build_master()

    for size in PNG_SIZES:
        render_icon(master, size).save(ICON_DIR / f"{size}x{size}.png")

    master.save(ICO_PATH, format="ICO", sizes=ICO_SIZES)


if __name__ == "__main__":
    main()
