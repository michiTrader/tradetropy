"""
Regenerate the Tradetropy badge logo (icon + wordmark on a rounded pill).

The badge is rebuilt from two faithful sources so no font is needed:

- The icon comes from ``docs/assets/logo.png`` (the clean transparent icon),
  scaled down with LANCZOS so it stays crisp.
- The "Tradetropy" wordmark is extracted as an alpha matte from the previous
  ``docs/assets/logo-badge.png``. Because the old badge background is a flat
  color, the glyph alpha is recovered exactly by un-blending each pixel between
  the background and the text color - this preserves the original typeface
  pixel-for-pixel while letting us re-composite it onto a new pill.

The pill is a stadium (fully rounded ends) in the original brand color. The new
layout makes the icon a bit larger and, most importantly, adds noticeably more
padding on the RIGHT so the wordmark no longer crowds the rounded edge.

Usage:
    python tools/gen_logo.py                 # write the final badge
    python tools/gen_logo.py --out out.png   # write to a custom path (preview)
    python tools/gen_logo.py --pad-right 180 # override right padding (preview)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Repo-relative asset paths
ASSETS = Path(__file__).resolve().parent.parent / 'docs' / 'assets'
ICON_SRC = ASSETS / 'logo.png'
WORDMARK_SRC = ASSETS / 'logo-wordmark-src.png'
BADGE_OUT = ASSETS / 'logo-badge.png'

# Brand colors (sampled from the original badge)
PILL_COLOR = (26, 32, 28, 255)      # dark green-black pill
TEXT_COLOR = (238, 241, 237)        # near-white wordmark
_BG = np.array([26, 32, 28], dtype=np.float64)
_TXT_L = 237.0                      # wordmark luminance in the source
_BG_L = float(_BG.mean())           # background luminance

# Geometry of the new badge (final render is high-resolution for crisp display)
HEIGHT = 640                        # pill height in px (was 300)
ICON_H = 470                        # icon height (0.73 * H, larger than before)
PAD_LEFT = 175                      # left edge -> icon (more air on the icon side)
GAP = 145                           # icon right -> wordmark left
PAD_RIGHT = 200                     # wordmark right -> right edge (was ~41 @300)
TEXT_SCALE = 2.0                    # upscale factor for the extracted wordmark


def _crop_to_alpha(rgba: np.ndarray, thresh: int = 15) -> np.ndarray:
    """Crop an RGBA array to its non-transparent bounding box."""
    a = rgba[:, :, 3]
    cols = np.where(a.max(0) > thresh)[0]
    rows = np.where(a.max(1) > thresh)[0]
    return rgba[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def load_icon() -> Image.Image:
    """Load the transparent icon, cropped to its content bounding box."""
    icon = np.array(Image.open(ICON_SRC).convert('RGBA'))
    return Image.fromarray(_crop_to_alpha(icon))


def load_wordmark() -> Image.Image:
    """
    Load the "Tradetropy" wordmark as a clean RGBA sprite.

    The sprite (``logo-wordmark-src.png``) is a pre-extracted alpha matte of the
    original typeface on transparency, so generation never depends on (nor is
    corrupted by) the badge it also writes. It was matted once from the original
    flat-color badge by un-blending each text pixel between background and text
    color; kept as a stable source asset here.
    """
    return Image.open(WORDMARK_SRC).convert('RGBA')


def build_badge(pad_left: int = PAD_LEFT, pad_right: int = PAD_RIGHT) -> Image.Image:
    """Compose the icon + wordmark onto a new rounded-pill background."""
    # Scale the icon to the target height, preserving aspect ratio.
    icon = load_icon()
    icon_w = round(ICON_H * icon.width / icon.height)
    icon = icon.resize((icon_w, ICON_H), Image.LANCZOS)

    # Scale the wordmark.
    word = load_wordmark()
    word_w = round(word.width * TEXT_SCALE)
    word_h = round(word.height * TEXT_SCALE)
    word = word.resize((word_w, word_h), Image.LANCZOS)

    width = pad_left + icon_w + GAP + word_w + pad_right
    canvas = Image.new('RGBA', (width, HEIGHT), (0, 0, 0, 0))

    # Draw the stadium (fully rounded ends).
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([0, 0, width - 1, HEIGHT - 1],
                           radius=HEIGHT // 2, fill=PILL_COLOR)

    # Paste the icon, vertically centered.
    icon_x = pad_left
    icon_y = (HEIGHT - ICON_H) // 2
    canvas.alpha_composite(icon, (icon_x, icon_y))

    # Paste the wordmark, vertically centered on its bounding box.
    word_x = pad_left + icon_w + GAP
    word_y = (HEIGHT - word_h) // 2
    canvas.alpha_composite(word, (word_x, word_y))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate the Tradetropy badge logo.')
    parser.add_argument('--out', type=Path, default=BADGE_OUT,
                        help='output PNG path (default: overwrite the badge)')
    parser.add_argument('--pad-left', type=int, default=PAD_LEFT,
                        help='left padding in px (left edge -> icon)')
    parser.add_argument('--pad-right', type=int, default=PAD_RIGHT,
                        help='right padding in px (wordmark -> right edge)')
    args = parser.parse_args()

    badge = build_badge(pad_left=args.pad_left, pad_right=args.pad_right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    badge.save(args.out)
    print(f'Wrote {args.out} ({badge.width} x {badge.height})')


if __name__ == '__main__':
    main()
