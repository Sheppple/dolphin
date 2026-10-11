"""Animated page-header titles for an AYN Thor launcher, on a transparent background.

Titles: HOME, LIBRARY, PLATFORMS, COLLECTIONS, APPS
Colours: red (light sweep + colour-split glitch + neon flicker), blue (light sweep + colour-split glitch)
Sizes:   banner 900 x 300 (3x1 tile), strip 1800 x 300 (full-width 6x1 tile); the title is centred
         tile-2x1 600 x 288: GIF for a 2-square Cocoon tile, full soft glow on a black rounded card

Two formats per header:
  .gif   GIF transparency is on/off only, so the glow is blended onto black and cut out around it.
         Seamless on a black (OLED) wallpaper; shows a dark halo on light wallpapers.
  .webp  Animated WebP with real alpha, so the glow fades smoothly over any wallpaper.

Fonts (SIL Open Font License, from github.com/google/fonts): ofl/unbounded/Unbounded[wght].ttf

Usage:
  python3 make_headers.py --fonts /path/to/fonts --out .
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FRAMES = 36
FRAME_MS = 55
GIF_ALPHA_CUTOFF = 0.05

STYLES = {
    "red": dict(glow=(255, 30, 62), top=(255, 238, 241), bot=(255, 96, 116)),
    "blue": dict(glow=(38, 118, 255), top=(236, 248, 255), bot=(100, 178, 255)),
}
TITLES = ["HOME", "LIBRARY", "PLATFORMS", "COLLECTIONS", "APPS"]
SIZES = {"banner": (900, 300), "strip": (1800, 300)}
# 2x1 Cocoon tile (two 279 px squares + a 24 px gap is about 582 x 279): soft glow on a black rounded card
TILE_SIZE = (600, 288)
TILE_RADIUS = 34
CLEAN_ALPHA_CUTOFF = 0.3


def title_font(font_dir, size):
    f = ImageFont.truetype(str(Path(font_dir) / "Unbounded[wght].ttf"), size)
    try:
        f.set_variation_by_axes([800])
    except Exception:
        pass
    return f


def render(style_key, title, w, h, font_dir, clean=False):
    """clean=True: a tight glow only, so a hard-edged GIF has no dark halo on light backgrounds."""
    S = {k: np.array(v, dtype=np.float32) / 255.0 for k, v in STYLES[style_key].items()}
    pad = int(h * (0.1 if clean else 0.2))
    size = int(h * 0.46)
    while True:
        ft = title_font(font_dir, size)
        if ft.getlength(title) <= w - 2 * pad or size < 20:
            break
        size -= 2
    mask_im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_im).text((w / 2, h / 2), title, font=ft, fill=255, anchor="mm")
    x0, y0, x1, y1 = mask_im.getbbox()

    tm = np.asarray(mask_im, np.float32) / 255.0
    g1 = np.asarray(mask_im.filter(ImageFilter.GaussianBlur(h * 0.025)), np.float32) / 255.0 * 1.5
    g2 = np.asarray(mask_im.filter(ImageFilter.GaussianBlur(h * 0.09)), np.float32) / 255.0 * 2.4
    glow_base = np.clip(0.9 * g1 + 0.55 * g2, 0, 1)
    if clean:
        tight = np.asarray(mask_im.filter(ImageFilter.GaussianBlur(h * 0.022)), np.float32) / 255.0
        glow_base = np.clip(tight * 1.7, 0, 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = np.clip((yy - y0) / max(1, y1 - y0), 0, 1)[..., None]
    fill = S["top"] * (1 - v) + S["bot"] * v
    diag = (xx - x0 + 0.45 * (yy - y0)) / max(1, x1 - x0)

    frames = []
    for f in range(FRAMES):
        t = f / FRAMES
        if style_key == "red":
            inten = 1.0 + 0.04 * math.sin(2 * math.pi * 3 * t)
            if 0.60 <= t < 0.64:
                inten = 0.35
            elif 0.66 <= t < 0.70:
                inten = 0.7
        else:
            inten = 1.0 + 0.05 * math.sin(2 * math.pi * 2 * t)

        shine = np.exp(-((diag - (-0.3 + 1.6 * t)) / 0.06) ** 2)[..., None]
        # clean tiles: the flicker only cuts the neon edge; letters stay bright so they never turn muddy on light backgrounds
        title_rgb = np.clip(fill * (max(min(inten, 1.0), 0.85) if clean else min(inten, 1.0)) + shine * 0.85, 0, 1)
        glow_a = np.clip(glow_base * inten, 0, 1)

        # premultiplied colour and alpha: title over its own glow
        pre = title_rgb * tm[..., None] + S["glow"] * (glow_a * (1 - tm))[..., None]
        alpha = tm + glow_a * (1 - tm)

        if 0.40 <= t < 0.43 or 0.47 <= t < 0.49:
            dx = int(h * 0.03)
            mag, cyn = np.zeros_like(tm), np.zeros_like(tm)
            mag[:, dx:] = tm[:, :-dx] * 0.7
            cyn[:, :-dx] = tm[:, dx:] * 0.7
            pre += mag[..., None] * np.array([1.0, 0.2, 0.8]) * (1 - alpha)[..., None]
            alpha = alpha + mag * (1 - alpha)
            pre += cyn[..., None] * np.array([0.2, 0.9, 1.0]) * (1 - alpha)[..., None]
            alpha = alpha + cyn * (1 - alpha)

        frames.append((np.clip(pre, 0, 1), np.clip(alpha, 0, 1)))
    return frames


def save_webp(frames, path):
    ims = []
    for pre, a in frames:
        rgb = np.where(a[..., None] > 1e-4, pre / np.maximum(a[..., None], 1e-4), 0)
        rgba = np.dstack([np.clip(rgb, 0, 1), a])
        ims.append(Image.fromarray((rgba * 255).astype(np.uint8), "RGBA"))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=FRAME_MS, loop=0, quality=88, alpha_quality=90, method=6)


def save_gif(frames, path):
    rgbs = [Image.fromarray((pre * 255).astype(np.uint8), "RGB") for pre, _ in frames]
    w, h = rgbs[0].size
    sample = Image.new("RGB", (w, h * 6))
    for i in range(6):
        sample.paste(rgbs[i * len(rgbs) // 6], (0, h * i))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = pal.getpalette()[:255 * 3] + [0, 0, 0]
    out = []
    for im, (_, a) in zip(rgbs, frames):
        q = np.array(im.quantize(palette=pal, dither=Image.Dither.NONE))
        q[a < GIF_ALPHA_CUTOFF] = 255
        p = Image.fromarray(q.astype(np.uint8), "P")
        p.putpalette(palette)
        out.append(p)
    out[0].save(path, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, disposal=2, transparency=255, optimize=False)


def save_gif_clean(frames, path):
    """Hard-edged GIF with no dark fringe: every kept pixel uses its true (unpremultiplied) colour."""
    rgbs, alphas = [], []
    for pre, a in frames:
        rgb = np.where(a[..., None] > 1e-4, pre / np.maximum(a[..., None], 1e-4), 0)
        rgbs.append(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB"))
        alphas.append(a)
    w, h = rgbs[0].size
    # build the palette from every frame's visible pixels, so dim flicker frames get their own colours
    vis = np.concatenate([np.asarray(im)[al >= CLEAN_ALPHA_CUTOFF] for im, al in zip(rgbs, alphas)])
    sample = Image.fromarray(vis.reshape(1, -1, 3).astype(np.uint8), "RGB")
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = pal.getpalette()[:255 * 3] + [0, 0, 0]
    out = []
    for im, a in zip(rgbs, alphas):
        q = np.array(im.quantize(palette=pal, dither=Image.Dither.NONE))
        q[a < CLEAN_ALPHA_CUTOFF] = 255
        p = Image.fromarray(q.astype(np.uint8), "P")
        p.putpalette(palette)
        out.append(p)
    out[0].save(path, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, disposal=2, transparency=255, optimize=False)


def save_gif_card(frames, path, radius=TILE_RADIUS):
    """Soft glow composited onto an OLED-black rounded rectangle; only the corners are transparent."""
    w, h = frames[0][0].shape[1], frames[0][0].shape[0]
    card = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(card).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), radius=radius * 4, fill=255)
    inside = np.asarray(card.resize((w, h), Image.LANCZOS)) >= 128
    rgbs = [Image.fromarray((np.clip(pre, 0, 1) * 255).astype(np.uint8), "RGB") for pre, _ in frames]
    vis = np.concatenate([np.asarray(im)[inside] for im in rgbs])
    pal = Image.fromarray(vis.reshape(1, -1, 3), "RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = pal.getpalette()[:255 * 3] + [0, 0, 0]
    out = []
    for im in rgbs:
        q = np.array(im.quantize(palette=pal, dither=Image.Dither.NONE))
        q[~inside] = 255
        p = Image.fromarray(q.astype(np.uint8), "P")
        p.putpalette(palette)
        out.append(p)
    out[0].save(path, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, disposal=2, transparency=255, optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--styles", nargs="*", default=list(STYLES), help="subset of styles to render, e.g. --styles red")
    ap.add_argument("--tiles-only", action="store_true", help="only make the clean 2x1 tile GIFs")
    a = ap.parse_args()
    for style in a.styles:
        for title in TITLES:
            path = Path(a.out) / style / "tile-2x1" / f"{title.lower()}.gif"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_gif_card(render(style, title, *TILE_SIZE, a.fonts), path)
            print(path, path.stat().st_size // 1024, "KB")
    if a.tiles_only:
        return
    for style in a.styles:
        for title in TITLES:
            for size_key, (w, h) in SIZES.items():
                frames = render(style, title, w, h, a.fonts)
                base = Path(a.out) / style / f"{title.lower()}-{size_key}"
                base.parent.mkdir(parents=True, exist_ok=True)
                save_gif(frames, base.with_suffix(".gif"))
                save_webp(frames, base.with_suffix(".webp"))
                print(base, base.with_suffix(".gif").stat().st_size // 1024, "KB gif,", base.with_suffix(".webp").stat().st_size // 1024, "KB webp")


if __name__ == "__main__":
    main()
