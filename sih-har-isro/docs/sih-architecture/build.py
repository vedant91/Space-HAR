"""Build the SIH architecture slides: template.html -> index.html (+ PNGs).

    python build.py            # write index.html
    python build.py --png      # also render slide1.png / slide2.png (Edge headless, 2x)

Placeholders in template.html:
    {{i:name}}   Lucide line icon     (lucide.json)
    {{b:slug}}   brand logo, own hex  (brands.json)
"""
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

lucide = json.loads((HERE / "lucide.json").read_text(encoding="utf-8"))
brands = json.loads((HERE / "brands.json").read_text(encoding="utf-8"))
tpl = (HERE / "template.html").read_text(encoding="utf-8")

used_i = sorted(set(re.findall(r"\{\{i:([\w-]+)\}\}", tpl)))
used_b = sorted(set(re.findall(r"\{\{b:([\w-]+)\}\}", tpl)))
missing = [n for n in used_i if n not in lucide] + [n for n in used_b if n not in brands]
assert not missing, f"unknown icons: {missing}"

sprite = ['<svg width="0" height="0" style="position:absolute" aria-hidden="true">']
for n in used_i:
    sprite.append(f'<symbol id="i-{n}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                  f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{lucide[n]}</symbol>')
for n in used_b:
    sprite.append(f'<symbol id="b-{n}" viewBox="0 0 24 24"><path fill="currentColor" d="{brands[n]["d"]}"/></symbol>')
sprite.append("</svg>")

html = tpl.replace("<!--SPRITE-->", "\n".join(sprite))
html = re.sub(r"\{\{i:([\w-]+)\}\}", r'<svg class="ic"><use href="#i-\1"/></svg>', html)
html = re.sub(r"\{\{b:([\w-]+)\}\}",
              lambda m: f'<svg class="bi" style="color:{brands[m[1]]["hex"]}"><use href="#b-{m[1]}"/></svg>', html)
out = HERE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({len(used_i)} icons, {len(used_b)} logos)")

if "--png" in sys.argv:
    for s in range(1, html.count('<section class="slide"') + 1):
        png = HERE / f"slide{s}.png"
        # own profile dir per slide: a shared/busy profile makes headless Edge hand off and return without writing
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={Path(tempfile.gettempdir()) / f'edge-shot-{s}'}",
                        "--window-size=1920,1080", "--force-device-scale-factor=2",
                        "--virtual-time-budget=3000", f"--screenshot={png}",
                        out.as_uri() + f"?only={s}"], check=True, capture_output=True)
        # msedge.exe can return before its child process writes the file: wait for it
        for _ in range(60):
            if png.exists() and png.stat().st_mtime > out.stat().st_mtime:
                break
            time.sleep(0.5)
        else:
            raise SystemExit(f"{png} was not rewritten")
        print(f"rendered {png}")
