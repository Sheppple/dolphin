"""Animated page-header GIFs for an AYN Thor launcher.

Makes a red and a blue set of looping title GIFs on pure black (OLED friendly):
  HOME, LIBRARY, PLATFORMS, COLLECTIONS, APPS

Sizes:
  banner  900 x 300   (3:1, fits a 3x1 tile: half the screen width)
  strip  1800 x 300   (6:1, fits a full-width 6x1 tile; HOME, LIBRARY and APPS only)

Fonts (SIL Open Font License, from github.com/google/fonts):
  ofl/unbounded/Unbounded[wght].ttf, ofl/manrope/Manrope[wght].ttf, ofl/monoton/Monoton-Regular.ttf

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

STYLES = {
    "red": dict(glow=(255, 30, 62), top=(255, 238, 241), bot=(255, 96, 116), sub=(255, 82, 106), haze=(90, 0, 16), ember=(255, 110, 70)),
    "blue": dict(glow=(38, 118, 255), top=(236, 248, 255), bot=(100, 178, 255), sub=(104, 186, 255), haze=(0, 22, 80), ember=(120, 190, 255)),
}
TITLES = [
    ("home", "HOME", "Pick up where you left off", "01"),
    ("library", "LIBRARY", "Platforms and collections", "02"),
    ("platforms", "PLATFORMS", "Every console on one shelf", "02"),
    ("collections", "COLLECTIONS", "Smart folders, sorted by rule", "02"),
    ("apps", "APPS", "Music, video and everything else", "03"),
]
SIZES = {"banner": (900, 300), "strip": (1800, 300)}
STRIP_TITLES = {"home", "library", "apps"}


def load_fonts(font_dir):
    d = Path(font_dir)

    def var(name, weight, size):
        f = ImageFont.truetype(str(d / name), size)
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
        return f

    return {
        "title": lambda s: var("Unbounded[wght].ttf", 800, s),
        "sub": lambda s: var("Manrope[wght].ttf", 700, s),
        "num": lambda s: ImageFont.truetype(str(d / "Monoton-Regular.ttf"), s),
    }


def mask_from(draw_fn, w, h):
    im = Image.new("L", (w, h), 0)
    draw_fn(ImageDraw.Draw(im))
    return im


def blur(im, r):
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0


def arr(im):
    return np.asarray(im, dtype=np.float32) / 255.0


def layout(fonts, title, sub, num, w, h, with_num):
    pad = int(h * 0.16)
    reserve = int(h * 1.05) if with_num else 0
    size = int(h * 0.36)
    while True:
        ft = fonts["title"](size)
        tw = ft.getlength(title)
        if tw <= w - pad * 2 - reserve or size < 20:
            break
        size -= 2
    base = int(h * 0.56)
    fs = fonts["sub"](max(12, int(h * 0.072)))
    track = fs.size * 0.2

    def draw_title(d):
        d.text((pad, base), title, font=ft, fill=255, anchor="ls")

    def draw_sub(d):
        x = pad + 2
        for ch in sub.upper():
            d.text((x, base + int(h * 0.15)), ch, font=fs, fill=255, anchor="ls")
            x += fs.getlength(ch) + track

    def draw_num(d):
        if with_num:
            d.text((w - pad, int(h * 0.5)), num, font=fonts["num"](int(h * 0.5)), fill=255, anchor="rm")

    bbox = ft.getbbox(title, anchor="ls")
    return dict(
        pad=pad, base=base, title_w=tw,
        t_top=base + bbox[1], t_bot=base + bbox[3],
        title=mask_from(draw_title, w, h), sub=mask_from(draw_sub, w, h), num=mask_from(draw_num, w, h),
        bar_y=base + int(h * 0.215),
    )


def render(style_key, item, size_key, fonts, rng_seed):
    name, title, sub, num = item
    w, h = SIZES[size_key]
    S = {k: np.array(v, dtype=np.float32) / 255.0 for k, v in STYLES[style_key].items()}
    L = layout(fonts, title, sub, num, w, h, size_key == "strip")
    rng = np.random.default_rng(rng_seed)

    tm = arr(L["title"])
    g1, g2 = blur(L["title"], h * 0.025) * 1.5, blur(L["title"], h * 0.09) * 2.4
    sm, sg = arr(L["sub"]), blur(L["sub"], h * 0.02) * 1.2
    nm, ng = arr(L["num"]), blur(L["num"], h * 0.04) * 1.8

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = np.clip((yy - L["t_top"]) / max(1, L["t_bot"] - L["t_top"]), 0, 1)[..., None]
    fill = S["top"] * (1 - v) + S["bot"] * v
    haze = np.exp(-((xx - w * 0.35) / (w * 0.55)) ** 2 - ((yy - h * 1.1) / (h * 0.6)) ** 2)[..., None] * S["haze"]
    diag = (xx + 0.45 * yy) / w

    # particles: embers rising (red) or rain falling (blue); integer cycles keep the loop seamless
    n = int(w / 12)
    px, py = rng.random(n), rng.random(n)
    pk = rng.integers(1, 3, n)
    pph = rng.random(n)
    psz = rng.uniform(1.0, 2.6, n)
    plen = rng.uniform(16, 40, n)

    frames = []
    for f in range(FRAMES):
        t = f / FRAMES
        out = np.zeros((h, w, 3), np.float32) + haze

        if style_key == "red":
            inten = 1.0 + 0.04 * math.sin(2 * math.pi * 3 * t)
            if 0.60 <= t < 0.64:
                inten = 0.35
            elif 0.66 <= t < 0.70:
                inten = 0.7
            layer = np.zeros((h, w), np.float32)
            for i in range(n):
                y = h * (1.08 - ((py[i] + pk[i] * t) % 1.0) * 1.2)
                x = w * px[i] + 10 * math.sin(2 * math.pi * (t * pk[i] + pph[i]))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                b = (0.55 + 0.45 * math.sin(2 * math.pi * (2 * t + pph[i]))) * (y / h) ** 0.7
                r = psz[i]
                x0, x1, y0, y1 = int(max(0, x - 3 * r)), int(min(w, x + 3 * r + 1)), int(max(0, y - 3 * r)), int(min(h, y + 3 * r + 1))
                gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
                layer[y0:y1, x0:x1] += b * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * (r * 0.8) ** 2))
            out += layer[..., None] * S["ember"]
        else:
            inten = 1.0 + 0.05 * math.sin(2 * math.pi * 2 * t)
            rain = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(rain)
            for i in range(n):
                ln = plen[i]
                y = ((py[i] + pk[i] * t) % 1.0) * (h + ln) - ln
                x = w * px[i] - 0.25 * y
                x = x % w
                d.line([(x, y), (x - 0.25 * ln, y + ln)], fill=int(60 + 70 * pph[i]), width=1)
            out += arr(rain)[..., None] * S["glow"] * 0.45
            sy = (t % 1.0) * h * 1.4 - h * 0.2
            scan = np.exp(-((yy - sy) / (h * 0.025)) ** 2)[..., None]
            out += scan * S["glow"] * 0.12

        # title: two glows, gradient fill, shine sweep
        c = -0.25 + 1.6 * t
        shine = np.exp(-((diag - c) / 0.035) ** 2)[..., None]
        out += (g2[..., None] * 0.55 + g1[..., None] * 0.9) * S["glow"] * inten
        title_rgb = fill * inten + shine * 0.85
        if style_key == "blue" and (0.40 <= t < 0.43 or 0.47 <= t < 0.49):
            dx = int(h * 0.03)
            out[:, dx:, 0] += tm[:, :-dx] * 0.6
            out[:, :-dx, 1:] += tm[:, dx:, None] * np.array([0.7, 0.9])
        out = out * (1 - tm[..., None]) + title_rgb * tm[..., None]

        # subtitle and underline bar
        out += sg[..., None] * S["glow"] * 0.6 * inten
        out = out * (1 - sm[..., None]) + S["sub"] * inten * sm[..., None]
        bar_full = L["title_w"] * 0.32
        by, bx = L["bar_y"], L["pad"] + 2
        if style_key == "red":
            bw = bar_full * (0.55 + 0.45 * (0.5 + 0.5 * math.sin(2 * math.pi * t)))
            out[by:by + 3, bx:bx + int(bw)] = S["glow"] * 1.2
        else:
            out[by:by + 3, bx:bx + int(bar_full)] = S["glow"] * 0.35
            seg = int(bar_full * 0.25)
            sx = bx + int((bar_full - seg) * (0.5 - 0.5 * math.cos(2 * math.pi * t)))
            out[by:by + 3, sx:sx + seg] = S["top"]

        # outlined neon page number on wide strips
        if size_key == "strip":
            pulse = 0.55 + 0.25 * math.sin(2 * math.pi * t)
            out += ng[..., None] * S["glow"] * pulse
            out = out * (1 - nm[..., None]) + (S["glow"] * 0.6 + 0.4) * pulse * nm[..., None]

        frames.append(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB"))
    return frames


def save_gif(frames, path):
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 6))
    for i, k in enumerate(range(0, len(frames), max(1, len(frames) // 6))):
        if i < 6:
            sample.paste(frames[k], (0, frames[0].height * i))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=FRAME_MS, loop=0, disposal=1, optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    fonts = load_fonts(a.fonts)
    for style in STYLES:
        for idx, item in enumerate(TITLES):
            for size in SIZES:
                if size == "strip" and item[0] not in STRIP_TITLES:
                    continue
                path = Path(a.out) / style / f"{item[0]}-{size}.gif"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_gif(render(style, item, size, fonts, rng_seed=idx * 7 + len(style)), path)
                print(f"{path}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
