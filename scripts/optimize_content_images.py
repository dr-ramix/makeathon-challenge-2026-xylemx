"""Downscale and compress content/images assets for README usage."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path("content/images")
MAX_SIDE = 1600
JPEG_QUALITY = 80


def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE:
        return img
    scale = MAX_SIDE / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _optimize_jpeg(path: Path) -> None:
    with Image.open(path) as img:
        img = _resize_if_needed(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)


def _optimize_png(path: Path) -> None:
    with Image.open(path) as img:
        img = _resize_if_needed(img)
        if img.mode in ("RGBA", "LA"):
            img.save(path, format="PNG", optimize=True, compress_level=9)
            return
        # Quantize opaque PNGs to reduce file size while keeping visuals sharp enough for README.
        quantized = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        quantized.save(path, format="PNG", optimize=True, compress_level=9)


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Missing directory: {ROOT}")

    for path in sorted(ROOT.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            _optimize_jpeg(path)
        elif suffix == ".png":
            _optimize_png(path)


if __name__ == "__main__":
    main()
