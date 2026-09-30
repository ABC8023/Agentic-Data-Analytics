"""
Run the test suite and write a short summary.

The verbose unittest output is mixed with library warnings, which makes it
awkward to read from a log. This writes just the counts and any failures
to tools/test_summary.txt.

    python tools/run_tests.py
    python tools/run_tests.py tests.test_anomalies
"""

import datetime
import io
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_summary.txt"
)

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
sys.path.insert(0, REPO)


RESULTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)


def write_summary(lines: list[str]) -> None:
    """
    Write the summary file, and echo it for whoever is watching.

    A second copy goes to a uniquely named file under tools/results. A
    reader that caches file contents by path cannot tell a fresh run from
    a stale one when every run overwrites the same name.
    """

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_path = os.path.join(RESULTS_DIRECTORY, f"tests-{stamp}.txt")

    with open(unique_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def main() -> int:
    try:
        return run()

    except BaseException as error:  # noqa: BLE001
        import traceback

        write_summary([
            f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}",
            "status: RUNNER CRASHED",
            f"exception: {type(error).__name__}: {error}",
            "",
            *traceback.format_exc().splitlines()[-25:],
        ])

        return 2


def run() -> int:
    selected = sys.argv[1:]

    loader = unittest.TestLoader()

    if selected:
        suite = loader.loadTestsFromNames(selected)

    else:
        suite = loader.discover(
            start_dir=os.path.join(REPO, "tests"),
            top_level_dir=REPO
        )

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)

    lines = [
        f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"ran: {result.testsRun}",
        f"failures: {len(result.failures)}",
        f"errors: {len(result.errors)}",
        f"skipped: {len(result.skipped)}",
        f"status: {'OK' if result.wasSuccessful() else 'BROKEN'}",
    ]

    for label, entries in (
        ("FAILURE", result.failures),
        ("ERROR", result.errors),
    ):
        for test, traceback_text in entries:
            lines.append("")
            lines.append(f"{label}: {test}")

            for line in traceback_text.strip().splitlines()[-6:]:
                lines.append(f"    {line}")

    write_summary(lines)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
