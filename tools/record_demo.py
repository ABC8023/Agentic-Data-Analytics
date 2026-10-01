"""
Record a short demo of the running app as an animated GIF.

It drives headless Chrome through a fixed script: open a sample, ask a
question and show its plan and SQL, ask one the data can't answer, then
train models and download the best one. Each scene gets a caption strip.

    streamlit run app.py --server.headless true --server.port 8599
    python tools/record_demo.py                   # docs/demo.gif

The declined question is only sent to the model planner when a key is
set. Without one, the rules parser declines it by itself, which is
also a true demo of the product.

Frames are kept only when the page changes, and waiting is shown sped
up, so the GIF stays small.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshot import BROWSERS, READY_JS, ROOT, Session  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "docs", "demo.gif")

VIEW_WIDTH = 1280
VIEW_HEIGHT = 760
CAPTION_HEIGHT = 64

TEAL = (15, 110, 99)
WHITE = (255, 255, 255)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuisb.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def caption_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default(size=size)


class Recorder:
    """Collects frames with durations, merging frames that didn't change."""

    def __init__(self, session, width):
        self.session = session
        self.width = width
        self.frames = []
        self.caption = ""
        self.font = caption_font(26)
        self.last_digest = None

    async def snap(self, duration_ms):
        shot = await self.session.send("Page.captureScreenshot", format="png")
        raw = base64.b64decode(shot["data"])
        digest = hashlib.sha256(raw + self.caption.encode()).hexdigest()

        if digest == self.last_digest and self.frames:
            self.frames[-1][1] += duration_ms
            return

        self.last_digest = digest
        self.frames.append([self.compose(raw), duration_ms])

    def compose(self, raw):
        page = Image.open(io.BytesIO(raw)).convert("RGB")
        frame = Image.new("RGB", (page.width, page.height + CAPTION_HEIGHT), TEAL)
        frame.paste(page, (0, 0))

        draw = ImageDraw.Draw(frame)
        draw.text(
            (32, page.height + CAPTION_HEIGHT // 2),
            self.caption,
            fill=WHITE,
            font=self.font,
            anchor="lm",
        )

        if frame.width != self.width:
            height = round(frame.height * self.width / frame.width)
            frame = frame.resize((self.width, height), Image.LANCZOS)

        return frame

    async def hold(self, seconds):
        await self.snap(int(seconds * 1000))

    async def settle(self, timeout=180, frame_ms=220):
        """Record while the app works, shown sped up, until it is idle."""

        return await self.wait_for("true", timeout, frame_ms)

    async def wait_for(self, condition, timeout=180, frame_ms=220):
        """
        Record, sped up, until the app is idle and condition holds.

        Streamlit's running indicator appears a moment after a click, so
        a spinner on the page also counts as busy, and callers can wait
        for the content they expect instead of trusting idleness alone.
        """

        check = (
            f"({READY_JS})"
            " && !document.querySelector('[data-testid=\"stSpinner\"]')"
            f" && (() => {{ return {condition}; }})()"
        )
        deadline = time.monotonic() + timeout
        stable = 0

        await asyncio.sleep(0.8)

        while time.monotonic() < deadline:
            await self.snap(frame_ms)
            stable = stable + 1 if await self.session.evaluate(check) else 0

            if stable >= 3:
                return True

            await asyncio.sleep(0.35)

        print(f"warning: timed out waiting for {condition}", file=sys.stderr)

        return False

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # One shared palette keeps the teal and the text colours steady
        # from frame to frame, instead of each frame picking its own.
        palette_source = self.frames[len(self.frames) // 2][0]
        palette = palette_source.quantize(colors=128, method=Image.Quantize.MEDIANCUT)

        images = [
            frame.quantize(palette=palette, dither=Image.Dither.NONE)
            for frame, _ in self.frames
        ]

        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=[max(60, duration) for _, duration in self.frames],
            loop=0,
            optimize=True,
            disposal=1,
        )


async def js(session, expression):
    return await session.evaluate(f"(() => {{ {expression} }})()")


async def click_text(session, text, selector="button", timeout=20):
    """Click the last visible element with this text, retrying until it appears."""

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        found = await js(
            session,
            # Expander summaries carry an icon name before the label, so
            # an exact match is tried first, then a partial one.
            f"const wanted = {json.dumps(text)};"
            f" const all = [...document.querySelectorAll({json.dumps(selector)})]"
            ".filter(e => e.getClientRects().length);"
            " const e = all.filter(e => e.innerText.trim() === wanted).pop()"
            " || all.filter(e => e.innerText.includes(wanted)).pop();"
            " if (!e) return false;"
            " e.scrollIntoView({block: 'center'}); e.click(); return true;",
        )

        if found:
            return

        await asyncio.sleep(0.5)

    shot = await session.send("Page.captureScreenshot", format="png")
    debug = os.path.join(ROOT, "tools", "results", "screens", "demo-failure.png")

    with open(debug, "wb") as handle:
        handle.write(base64.b64decode(shot["data"]))

    raise RuntimeError(f"Nothing visible called {text!r}. See {debug}")


async def click_tab(session, name):
    await click_text(session, name, '[role="tab"]')


async def scroll_to(session, selector, text="", block="center"):
    await js(
        session,
        f"const wanted = {json.dumps(text)};"
        f" const e = [...document.querySelectorAll({json.dumps(selector)})]"
        ".filter(e => e.innerText.includes(wanted)).pop();"
        " if (!e) return false;"
        f" e.scrollIntoView({{block: {json.dumps(block)}, behavior: 'instant'}});"
        # The sticky top bar covers the first 90 px or so of the page.
        " if (" + json.dumps(block) + " === 'start') {"
        " let p = e.parentElement;"
        " while (p && !(/(auto|scroll)/.test(getComputedStyle(p).overflowY)"
        " && p.scrollHeight > p.clientHeight)) p = p.parentElement;"
        " (p || document.scrollingElement).scrollBy(0, -110); }"
        " return true;",
    )


async def type_question(recorder, text):
    session = recorder.session
    await js(
        session,
        "const box = document.querySelector('[data-testid=\"stChatInputTextArea\"]');"
        " box.scrollIntoView({block: 'center'}); box.focus(); return true;",
    )

    for start in range(0, len(text), 2):
        await session.send("Input.insertText", text=text[start:start + 2])
        await recorder.snap(70)

    await recorder.hold(0.5)

    for kind in ("keyDown", "keyUp"):
        await session.send(
            "Input.dispatchKeyEvent",
            type=kind, key="Enter", code="Enter",
            windowsVirtualKeyCode=13, nativeVirtualKeyCode=13,
        )


async def choose_option(session, label, option):
    """Pick an option in a Streamlit select box, found by its label."""

    # The select box only opens on real pointer events, so this clicks
    # with the mouse at the control's position, then types to filter.
    point = await js(
        session,
        f"const wanted = {json.dumps(label)};"
        " const box = [...document.querySelectorAll('[data-testid=\"stSelectbox\"]')]"
        ".find(b => b.innerText.startsWith(wanted));"
        " if (!box) return null;"
        " box.scrollIntoView({block: 'center', behavior: 'instant'});"
        " const control = box.querySelector('input') || box;"
        " const r = control.getBoundingClientRect();"
        " return [r.left + r.width / 2, r.top + r.height / 2];",
    )

    if not point:
        raise RuntimeError(f"No select box called {label!r}")

    await mouse_click(session, *point)
    await asyncio.sleep(0.6)
    await session.send("Input.insertText", text=option)
    await asyncio.sleep(0.6)

    for kind in ("keyDown", "keyUp"):
        await session.send(
            "Input.dispatchKeyEvent",
            type=kind, key="Enter", code="Enter",
            windowsVirtualKeyCode=13, nativeVirtualKeyCode=13,
        )


async def mouse_click(session, x, y):
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        await session.send(
            "Input.dispatchMouseEvent",
            type=kind, x=x, y=y, button="left", clickCount=1,
        )


async def scene_dashboard(recorder, base_url):
    recorder.caption = "1  Open a table. The headline figures each show how they were worked out."
    await recorder.session.send("Page.navigate", url=base_url)
    await recorder.settle()
    await recorder.hold(1.2)
    await click_text(recorder.session, "Store orders")
    await recorder.settle()
    await recorder.hold(2.5)


async def scene_question(recorder):
    session = recorder.session
    recorder.caption = "2  Ask in plain English. The plan is read locally, run in pandas, and shown with its SQL."
    await click_tab(session, "AI analysis")
    await recorder.settle()
    await recorder.hold(0.8)

    await type_question(recorder, "revenue by region last quarter")
    await recorder.settle()
    await scroll_to(session, '[data-testid="stChatMessage"]', "Read as", block="start")
    await recorder.hold(2.5)

    await click_text(session, "Equivalent SQL", '[data-testid="stExpander"] summary')
    await asyncio.sleep(0.6)
    await scroll_to(session, '[data-testid="stExpander"]', "Equivalent SQL")
    await recorder.hold(3.5)


async def scene_declined(recorder):
    session = recorder.session
    recorder.caption = "3  The table has no age column, so the question is declined, not guessed."

    await type_question(recorder, "what is the average age of our customers")

    # Two questions and two answers.
    await recorder.wait_for(
        "document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 4"
    )
    await scroll_to(session, '[data-testid="stChatMessage"]', "", block="end")
    await recorder.hold(5.0)


async def scene_models(recorder, base_url, download_dir):
    session = recorder.session
    recorder.caption = "4  Train three models. They're ranked by 5-fold cross-validation, then checked on held-out rows."

    await session.send("Page.navigate", url=f"{base_url}?sample=iris")
    await recorder.settle()
    await click_tab(session, "Machine learning")
    await recorder.settle()
    await choose_option(session, "Target column", "target")
    await recorder.settle()
    await recorder.hold(1.2)

    await click_text(session, "Train and compare models")
    await recorder.wait_for(
        "[...document.querySelectorAll('button')]"
        ".some(b => b.innerText.includes('Download best model'))"
    )
    await scroll_to(session, ".sec", "Model leaderboard", block="start")
    await recorder.hold(5.0)

    recorder.caption = "5  Download the best model: a skops file (not a pickle), a model card and a loading guide."
    await scroll_to(session, "button", "Download best model")
    await recorder.hold(1.2)
    await click_text(session, "Download best model")

    deadline = time.monotonic() + 90
    downloaded = None

    while time.monotonic() < deadline and downloaded is None:
        await recorder.snap(200)
        done = [
            name for name in os.listdir(download_dir)
            if name.endswith(".zip")
        ]
        downloaded = done[0] if done else None
        await asyncio.sleep(0.5)

    if downloaded:
        recorder.caption = f"5  Downloaded {downloaded}: the model, its model card and a loading guide."

    await recorder.hold(3.5)

    return downloaded


async def record(arguments, debug_port, download_dir):
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
        width=VIEW_WIDTH, height=VIEW_HEIGHT, deviceScaleFactor=1, mobile=False,
    )
    try:
        await session.send(
            "Browser.setDownloadBehavior",
            behavior="allow", downloadPath=download_dir,
        )

    except RuntimeError:
        # Older Chrome builds only accept the page-level command.
        await session.send(
            "Page.setDownloadBehavior",
            behavior="allow", downloadPath=download_dir,
        )

    recorder = Recorder(session, arguments.width)
    base_url = f"http://localhost:{arguments.port}/"

    await scene_dashboard(recorder, base_url)
    await scene_question(recorder)
    await scene_declined(recorder)
    downloaded = await scene_models(recorder, base_url, download_dir)

    await connection.close()

    recorder.save(arguments.out)

    seconds = sum(duration for _, duration in recorder.frames) / 1000

    return recorder, seconds, downloaded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--width", type=int, default=1000, help="Width of the GIF.")
    arguments = parser.parse_args()

    browser = next((b for b in BROWSERS if os.path.exists(b) or shutil.which(b)), None)

    if browser is None:
        sys.exit("No Chrome, Chromium or Edge found.")

    debug_port = 9334
    profile = tempfile.mkdtemp(prefix="streamlit-demo-")
    download_dir = tempfile.mkdtemp(prefix="streamlit-demo-downloads-")
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

        recorder, seconds, downloaded = asyncio.run(
            record(arguments, debug_port, download_dir)
        )

        size_mb = os.path.getsize(arguments.out) / (1024 ** 2)
        print(f"{arguments.out}: {len(recorder.frames)} frames, {seconds:.0f} s, {size_mb:.1f} MB")

        if downloaded:
            path = os.path.join(download_dir, downloaded)
            print(f"downloaded {downloaded} ({os.path.getsize(path):,} bytes)")

        else:
            print("warning: the model download did not arrive", file=sys.stderr)

    finally:
        process.terminate()

        try:
            process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            process.kill()

        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
