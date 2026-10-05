"""Fetch the timeline of every assignee-test issue and extract the triage bot's first assignment.

For each issue: `gh api repos/microsoft/vscode/issues/N/timeline --paginate`; raw
timelines are cached in data/timelines/N.json (resumable). The bot's top-1
prediction is the assignee of the FIRST `assigned` event whose actor is
vs-code-engineering[bot].

Output: results/bot_assign.json
Usage:  python scripts/fetch_bot.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from common import DATA, REPO, RESULTS, TRIAGE_BOT, write_json


def fetch_timeline(num: int) -> list[dict]:
    path = DATA / "timelines" / f"{num}.json"
    if path.exists():
        return json.load(open(path))
    delay = 5
    for attempt in range(6):
        p = subprocess.run(["gh", "api", f"repos/{REPO}/issues/{num}/timeline", "--paginate", "--slurp"],
                           capture_output=True, text=True)
        if p.returncode == 0:
            pages = json.loads(p.stdout)
            events = [e for page in pages for e in page]
            write_json(path, events)
            return events
        print(f"[gh] #{num} try {attempt + 1}: {p.stderr.strip()[:200]}", file=sys.stderr)
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"timeline fetch failed for #{num}")


def main() -> None:
    sets = json.load(open(RESULTS / "sets.json"))["sets"]["assignee"]["test"]
    out = {}
    for i, item in enumerate(sets):
        num = item["number"]
        events = fetch_timeline(num)
        first = next((e for e in events if e.get("event") == "assigned"
                      and (e.get("actor") or {}).get("login") == TRIAGE_BOT), None)
        out[str(num)] = {
            "bot_assignee": (first.get("assignee") or {}).get("login") if first else None,
            "bot_assigned_at": first.get("created_at") if first else None,
            "n_events": len(events)}
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(sets)}")
    write_json(RESULTS / "bot_assign.json", out)
    n_bot = sum(v["bot_assignee"] is not None for v in out.values())
    print(f"{n_bot}/{len(out)} test issues have a bot assignment")


if __name__ == "__main__":
    main()
