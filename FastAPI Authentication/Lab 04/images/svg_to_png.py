"""Convert architecture-overview.svg -> architecture-overview.png using headless Chromium."""

from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
svg_path = HERE / "architecture-overview.svg"
png_path = HERE / "architecture-overview.png"

# Wrap the SVG in a tiny HTML page so Chromium can render it as a flat image.
html = f"""<!doctype html>
<html><head><style>
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  body {{ display: flex; }}
  svg {{ display: block; }}
</style></head>
<body>{svg_path.read_text()}</body></html>
"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(viewport={"width": 1600, "height": 1200},
                                  device_scale_factor=2)
    page = context.new_page()
    page.set_content(html, wait_until="load")
    # Grab the SVG bounding box and screenshot it exactly.
    svg = page.locator("svg")
    svg.screenshot(path=str(png_path), omit_background=False)
    browser.close()

print(f"Wrote {png_path}")
