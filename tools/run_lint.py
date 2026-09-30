"""
Run ruff and write the result where it can be read reliably.

    python tools/run_lint.py
"""

import datetime
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lint_summary.txt"
)


def main() -> int:
    ruff = os.path.join(REPO, ".venv", "Scripts", "ruff.exe")

    if not os.path.exists(ruff):
        ruff = "ruff"

    completed = subprocess.run(
        [ruff, "check", ".", "--output-format=concise"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    lines = [
        f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"exit: {completed.returncode}",
        f"status: {'CLEAN' if completed.returncode == 0 else 'ISSUES'}",
        "",
    ]

    output = (completed.stdout + completed.stderr).strip()
    lines += output.splitlines() if output else ["no output"]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    # A uniquely named copy, so a cached reader cannot serve a stale run.
    results = os.path.join(os.path.dirname(OUTPUT_PATH), "results")
    os.makedirs(results, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(
        os.path.join(results, f"lint-{stamp}.txt"),
        "w",
        encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))

    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
