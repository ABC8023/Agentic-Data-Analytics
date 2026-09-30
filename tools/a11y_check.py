"""
Automated accessibility checks on the running app, in headless Chrome.

Two checks per page:

    axe-core, the standard automated WCAG rule engine, injected into the
    page. Reports violations of WCAG 2.1 A and AA rules it can test.

    A keyboard walk: Tab is pressed repeatedly and each focused element is
    recorded, so a control that cannot be reached, or focus that is not
    visible, shows up.

    streamlit run app.py --server.headless true --server.port 8599
    python tools/a11y_check.py
    python tools/a11y_check.py --page ai --query sample=store_orders --tab "AI analysis"

Automated checks find roughly a third of accessibility problems. They do
not replace testing with a screen reader and a keyboard by a person.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import screenshot  # noqa: E402

AXE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
AXE_CACHE = os.path.join(screenshot.ROOT, "tools", "results", "axe.min.js")

PAGES = {
    "landing": {"query": "", "tab": ""},
    "dashboard": {"query": "sample=store_orders", "tab": ""},
    "overview": {"query": "sample=customers_messy", "tab": "Dataset overview"},
    "ai": {"query": "sample=store_orders", "tab": "AI analysis"},
    "ml": {"query": "sample=iris", "tab": "Machine learning"},
}

RUN_AXE = """
axe.run(document, {
  runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']},
  resultTypes: ['violations']
}).then(r => JSON.stringify(r.violations.map(v => ({
  id: v.id, impact: v.impact, help: v.help,
  nodes: v.nodes.slice(0, 4).map(n => ({target: n.target.join(' '), summary: (n.failureSummary || '').slice(0, 220)})),
  count: v.nodes.length
}))))
"""

FOCUSED = """
(() => {
  const e = document.activeElement;
  if (!e || e === document.body) return null;
  const s = getComputedStyle(e);
  const label = e.getAttribute('aria-label') || e.innerText || e.getAttribute('placeholder') || e.value || '';
  return JSON.stringify({
    tag: e.tagName.toLowerCase(),
    role: e.getAttribute('role') || '',
    label: label.trim().replace(/\\s+/g, ' ').slice(0, 60),
    outline: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0,
    shadow: s.boxShadow !== 'none'
  });
})()
"""


def axe_source():
    if not os.path.exists(AXE_CACHE):
        os.makedirs(os.path.dirname(AXE_CACHE), exist_ok=True)

        with urllib.request.urlopen(AXE_URL, timeout=30) as response:
            data = response.read()

        with open(AXE_CACHE, "wb") as handle:
            handle.write(data)

    with open(AXE_CACHE, encoding="utf-8") as handle:
        return handle.read()


async def press_tab(session):
    for kind in ("keyDown", "keyUp"):
        await session.send(
            "Input.dispatchKeyEvent", type=kind, key="Tab", code="Tab",
            windowsVirtualKeyCode=9, nativeVirtualKeyCode=9,
        )


async def check(page, arguments, debug_port, source):
    with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/list") as response:
        target = [p for p in json.load(response) if p.get("type") == "page"][0]

    connection = await websockets.connect(target["webSocketDebuggerUrl"], max_size=2**28)
    session = screenshot.Session(connection)
    await session.send("Page.enable")
    await session.send("Runtime.enable")
    await session.send(
        "Emulation.setDeviceMetricsOverride",
        width=1440, height=900, deviceScaleFactor=1, mobile=False,
    )

    spec = PAGES[page]
    url = f"http://localhost:{arguments.port}/"

    if spec["query"]:
        url += f"?{spec['query']}"

    await session.send("Page.navigate", url=url)
    time.sleep(2)
    await screenshot.wait_ready(session, 90)

    if spec["tab"]:
        clicked = False
        deadline = time.monotonic() + 60

        while not clicked and time.monotonic() < deadline:
            clicked = await session.evaluate(
                "(() => { const t = [...document.querySelectorAll('[role=\"tab\"]')]"
                f".find(x => x.innerText.trim() === {json.dumps(spec['tab'])});"
                " if (t) { t.click(); return true; } return false; })()"
            )
            time.sleep(0.5 if clicked else 1.0)

        await screenshot.wait_ready(session, 90)

    time.sleep(3)

    await session.send("Runtime.evaluate", expression=source)
    result = await session.send(
        "Runtime.evaluate", expression=RUN_AXE, awaitPromise=True, returnByValue=True
    )
    violations = json.loads(result["result"]["value"])

    await session.evaluate("document.activeElement && document.activeElement.blur()")
    stops = []

    for _ in range(arguments.tabs):
        await press_tab(session)
        time.sleep(0.05)
        value = await session.evaluate(FOCUSED)
        stops.append(json.loads(value) if value else None)

    await connection.close()

    return violations, stops


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--page", choices=list(PAGES), action="append")
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--tabs", type=int, default=25, help="Tab presses to record.")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    pages = arguments.page or list(PAGES)
    source = axe_source()
    browser = next(b for b in screenshot.BROWSERS if os.path.exists(b) or shutil.which(b))
    debug_port = 9334
    profile = tempfile.mkdtemp(prefix="a11y-")
    process = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", f"--remote-debugging-port={debug_port}",
         f"--user-data-dir={profile}", "--no-first-run", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    report = {}

    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/version")
                break

            except OSError:
                time.sleep(0.25)

        for page in pages:
            violations, stops = asyncio.run(check(page, arguments, debug_port, source))
            report[page] = {"violations": violations, "tab_stops": stops}

    finally:
        process.terminate()
        shutil.rmtree(profile, ignore_errors=True)

    if arguments.json:
        print(json.dumps(report, indent=2))

        return

    for page, found in report.items():
        print(f"\n== {page}: {len(found['violations'])} axe violation(s)")

        for violation in found["violations"]:
            print(f"  [{violation['impact']}] {violation['id']} x{violation['count']}: {violation['help']}")

            for node in violation["nodes"][:2]:
                print(f"      {node['target'][:110]}")

        unseen = [s for s in found["tab_stops"] if s and not (s["outline"] or s["shadow"])]
        print(f"  keyboard: {sum(1 for s in found['tab_stops'] if s)} stops, {len(unseen)} without a visible focus style")

        for stop in found["tab_stops"][:25]:
            if stop:
                mark = "" if (stop["outline"] or stop["shadow"]) else "  <- no visible focus"
                print(f"      {stop['tag']}{('[' + stop['role'] + ']') if stop['role'] else ''} {stop['label']!r}{mark}")


if __name__ == "__main__":
    main()
