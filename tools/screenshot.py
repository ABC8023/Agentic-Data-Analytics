"""
Capture the running Streamlit app with headless Chrome, for design review.

Streamlit renders over a websocket after the page loads, so a plain
"chrome --screenshot" captures the loading skeleton. This drives Chrome
over the DevTools protocol instead and waits until the app has finished
running before capturing the full page.

    streamlit run app.py --server.headless true --server.port 8599
    python tools/screenshot.py before-landing
    python tools/screenshot.py after-dashboard --query sample=store_orders
    python tools/screenshot.py after-ai --query sample=store_orders --tab "AI analysis"

Images go to tools/results/screens/, which is git-ignored. No image is
taller or wider than 1,800 px: a long page is saved as name-1.png,
name-2.png and so on.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tools", "results", "screens")
# Largest side of any saved image, and the tallest page captured at all.
MAX_TILE = 1800
MAX_PAGE_HEIGHT = 9000

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
]

# True once Streamlit has rendered something and is no longer running.
READY_JS = """
(() => {
  const app = document.querySelector('[data-testid="stAppViewContainer"]');
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const skeleton = document.querySelector('[data-testid="stSkeleton"], .stSkeleton');
  const blocks = document.querySelectorAll('[data-testid="stVerticalBlock"]').length;
  // The app's own top bar is drawn on every page, so its presence means the
  // script has actually run rather than the loading skeleton being shown.
  const drawn = !!document.querySelector('.st-key-topbar');
  return !!app && !running && !skeleton && blocks > 0 && drawn;
})()
"""


class Session:

    def __init__(self, connection):
        self.connection = connection
        self.next_id = 0

    async def send(self, method, **params):
        self.next_id += 1
        message_id = self.next_id
        await self.connection.send(
            json.dumps({"id": message_id, "method": method, "params": params})
        )

        while True:
            reply = json.loads(await self.connection.recv())

            if reply.get("id") == message_id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])

                return reply.get("result", {})

    async def evaluate(self, expression):
        result = await self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True
        )

        return result.get("result", {}).get("value")


async def wait_ready(session, timeout):
    deadline = time.monotonic() + timeout
    stable = 0

    while time.monotonic() < deadline:
        # Ready for two checks in a row, so a rerun that starts right after
        # the first render is not captured half-way.
        stable = stable + 1 if await session.evaluate(READY_JS) else 0

        if stable >= 2:
            return True

        time.sleep(0.75)

    return False


async def capture(arguments, debug_port):
    with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/list") as response:
        pages = [p for p in json.load(response) if p.get("type") == "page"]

    connection = await websockets.connect(
        pages[0]["webSocketDebuggerUrl"], max_size=256 * 1024 * 1024
    )
    session = Session(connection)

    await session.send("Page.enable")
    await session.send("Runtime.enable")
    await session.send(
        "Emulation.setDeviceMetricsOverride",
        width=arguments.width, height=arguments.height,
        deviceScaleFactor=1, mobile=arguments.width < 700,
    )

    url = f"http://localhost:{arguments.port}/"

    if arguments.query:
        url += f"?{arguments.query}"

    await session.send("Page.navigate", url=url)
    time.sleep(2)

    if not await wait_ready(session, arguments.timeout):
        print("warning: the app did not report ready; capturing anyway", file=sys.stderr)

    if arguments.tab:
        # A sample opened from the link loads on the first run, so the tabs
        # can appear a few seconds after the page first reports ready.
        click_js = (
            "(() => { const tab = [...document.querySelectorAll('[role=\"tab\"]')]"
            f".find(t => t.innerText.trim() === {json.dumps(arguments.tab)});"
            " if (tab) { tab.click(); return true; } return false; })()"
        )
        clicked = False
        deadline = time.monotonic() + arguments.timeout

        while not clicked and time.monotonic() < deadline:
            clicked = await session.evaluate(click_js)

            if not clicked:
                time.sleep(1.0)

        if not clicked:
            print(f"warning: no tab called {arguments.tab!r}", file=sys.stderr)

        time.sleep(1.5)
        await wait_ready(session, arguments.timeout)

    # Press buttons or tick checkboxes by their visible text, in order,
    # waiting for the app to settle after each one.
    for text in arguments.click:
        press_js = (
            "(() => { const wanted = " + json.dumps(text) + ";"
            " const box = [...document.querySelectorAll('input[type=checkbox]')]"
            ".find(e => e.getAttribute('aria-label') === wanted);"
            " if (box) { box.scrollIntoView(); box.click(); return true; }"
            " const label = [...document.querySelectorAll('label')]"
            ".find(e => e.innerText.trim() === wanted && e.querySelector('input'));"
            " if (label) { label.scrollIntoView(); label.querySelector('input').click(); return true; }"
            " const found = [...document.querySelectorAll('button')]"
            ".find(e => e.innerText.trim() === wanted && e.offsetParent);"
            " if (found) { found.scrollIntoView(); found.click(); return true; }"
            " return false; })()"
        )

        if not await session.evaluate(press_js):
            print(f"warning: nothing visible called {text!r}", file=sys.stderr)

        time.sleep(2.0)
        await wait_ready(session, arguments.timeout)

    time.sleep(arguments.settle)

    page_height = arguments.height

    if arguments.full:
        # Streamlit scrolls inside its own container, so the body is only
        # as tall as the window. Measure the content block instead.
        measured = await session.evaluate(
            "Math.max(...['[data-testid=\"stMainBlockContainer\"]', '.block-container',"
            " '[data-testid=\"stMain\"]', '[data-testid=\"stAppViewContainer\"]']"
            ".map(s => document.querySelector(s))"
            ".filter(Boolean).map(e => Math.max(e.scrollHeight, e.offsetHeight)),"
            " document.body.scrollHeight) + 40"
        )
        page_height = int(min(max(measured or 0, arguments.height), MAX_PAGE_HEIGHT))
        await session.send(
            "Emulation.setDeviceMetricsOverride",
            width=arguments.width, height=page_height,
            deviceScaleFactor=1, mobile=arguments.width < 700,
        )
        time.sleep(1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []

    # Every image stays under MAX_TILE px on both sides. Image-reading
    # tools, including model APIs, refuse larger images, so a long page is
    # written as numbered tiles instead of one tall strip.
    tiles = max(1, -(-page_height // MAX_TILE))

    for index in range(tiles):
        top = index * MAX_TILE
        height = min(MAX_TILE, page_height - top)
        shot = await session.send(
            "Page.captureScreenshot",
            format="png",
            clip={
                "x": 0, "y": top,
                "width": arguments.width, "height": height,
                "scale": 1,
            },
        )
        suffix = f"-{index + 1}" if tiles > 1 else ""
        path = os.path.join(OUT_DIR, f"{arguments.name}{suffix}.png")

        with open(path, "wb") as handle:
            handle.write(base64.b64decode(shot["data"]))

        paths.append(path)

    await connection.close()

    return "\n".join(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name")
    parser.add_argument("--query", default="")
    parser.add_argument("--tab", default="")
    parser.add_argument(
        "--click",
        action="append",
        default=[],
        help="Visible text of a button or checkbox to press. Repeatable.",
    )
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--full", action="store_true", help="Capture the whole page height.")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--settle", type=float, default=1.5)
    arguments = parser.parse_args()

    if arguments.width > MAX_TILE:
        parser.error(f"--width can be at most {MAX_TILE}.")

    arguments.height = min(arguments.height, MAX_TILE)

    browser = next((b for b in BROWSERS if os.path.exists(b) or shutil.which(b)), None)

    if browser is None:
        sys.exit("No Chrome, Chromium or Edge found.")

    debug_port = 9333
    profile = tempfile.mkdtemp(prefix="streamlit-shot-")
    process = subprocess.Popen(
        [
            browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={debug_port}", f"--user-data-dir={profile}",
            "--no-first-run", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                break

            except OSError:
                time.sleep(0.25)

        print(asyncio.run(capture(arguments, debug_port)))

    finally:
        process.terminate()

        try:
            process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            process.kill()

        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
