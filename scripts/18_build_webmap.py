"""
18_build_webmap.py -- assemble the web map page from webmap/index.src.html and webmap/data/summary.json.

Writes two versions:
  webmap/index.html   for a claude.ai artifact: no document wrapper (the host adds it), MapLibre CSS inlined
  docs/index.html     standalone page for GitHub Pages: full HTML document, MapLibre CSS linked from jsDelivr,
                      data files copied to docs/data/

    python scripts\\18_build_webmap.py [--pages-dir docs]
"""
import argparse, os, shutil, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "webmap")
CSS_URL = "https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.css"
DATA = ["roads_severe.json", "roads_elevated.json", "roads_minor.json", "roads_context.json", "rain.geojson",
        "adm1.geojson", "facilities.geojson", "hillshade.png", "slip.json", "mask.geojson"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pages-dir", default=os.path.join(ROOT, "docs")); a = ap.parse_args()
    src = open(os.path.join(WEB, "index.src.html"), encoding="utf-8").read()
    summary = open(os.path.join(WEB, "data", "summary.json"), encoding="utf-8").read()
    page = src.replace("/*SUMMARY*/", summary)

    css_cache = os.path.join(WEB, "vendor", "maplibre-gl.css")
    if not os.path.exists(css_cache):
        os.makedirs(os.path.dirname(css_cache), exist_ok=True)
        urllib.request.urlretrieve(CSS_URL, css_cache)
    css = open(css_cache, encoding="utf-8").read()
    open(os.path.join(WEB, "index.html"), "w", encoding="utf-8").write(page.replace("/*MAPLIBRE_CSS*/", css))

    # standalone: split the head-type tags (title, meta, link, style) from the body
    body_start = page.index('<div class="app">')
    head = page[:body_start].replace("<style>\n/*MAPLIBRE_CSS*/\n</style>\n", f'<link rel="stylesheet" href="{CSS_URL}">\n')
    head = head.replace('<meta charset="utf-8">\n', "")
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + head + "<style>* { box-sizing: border-box; } body { margin: 0; } [hidden] { display: none !important; }</style>\n"
           "</head>\n<body>\n" + page[body_start:] + "\n</body>\n</html>\n")
    os.makedirs(os.path.join(a.pages_dir, "data"), exist_ok=True)
    open(os.path.join(a.pages_dir, "index.html"), "w", encoding="utf-8").write(doc)
    for f in DATA:
        shutil.copyfile(os.path.join(WEB, "data", f), os.path.join(a.pages_dir, "data", f))
    open(os.path.join(a.pages_dir, ".nojekyll"), "w").close()
    print(f"wrote webmap/index.html and {a.pages_dir}/index.html + {len(DATA)} data files")
