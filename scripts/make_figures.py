"""Capture app screenshots and the 'Try to break it' GIF for the README (needs Playwright + Chromium + Pillow).

    streamlit run app/Home.py --server.port 8599 &
    python scripts/make_figures.py --url http://localhost:8599
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def main() -> None:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        exe = os.environ.get("CHROMIUM_PATH")
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)

        def settle():
            page.wait_for_timeout(1500)
            page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
                                   timeout=60_000)
            page.wait_for_timeout(1200)

        for name, path in [("home", ""), ("simulate", "Simulate_and_Compare"), ("veto", "Physics_Veto"),
                           ("thermal_darcy", "Thermal_and_Darcy_Cases"), ("roi", "ROI_Business_Case"),
                           ("benchmark", "Benchmark_Lab"), ("gtm", "GTM_Studio")]:
            page.goto(f"{a.url}/{path}")
            settle()
            page.screenshot(path=str(FIG / f"{name}.png"))
            print("saved", name)

        page.goto(f"{a.url}/Physics_Veto")
        settle()
        frames = [Image.open(io.BytesIO(page.screenshot()))]
        for label in ["💥 Try to break it", "✅ Normal case", "💥 Try to break it", "💥 Try to break it"]:
            page.get_by_role("button", name=label).click()
            settle()
            frames.append(Image.open(io.BytesIO(page.screenshot())))
        small = [f.convert("RGB").resize((960, 667)) for f in frames]
        small[0].save(FIG / "try_to_break_it.gif", save_all=True, append_images=small[1:], duration=1800, loop=0)
        print("saved try_to_break_it.gif")
        b.close()


if __name__ == "__main__":
    main()
