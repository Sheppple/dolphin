"""Builds the three OLED launcher prototypes from template.html.

Run: python3 build.py  (writes ../crimson-web.html, ../night-pinboard.html, ../poster-wall.html)
"""
from pathlib import Path

HERE = Path(__file__).parent
OUT = {
    "crimson": ("Crimson Web", "crimson-web.html"),
    "pinboard": ("Night Pinboard", "night-pinboard.html"),
    "poster": ("Poster Wall", "poster-wall.html"),
}

template = (HERE / "template.html").read_text()
for key, (title, name) in OUT.items():
    html = template.replace("__KEY__", key).replace("__TITLE__", title)
    (HERE.parent / name).write_text(html)
    print("wrote", name)
