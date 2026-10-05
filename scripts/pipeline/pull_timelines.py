"""A5 (part 2): REST timelines for the assignee TEST issues; the triage bot's first assignment.

For every issue in the assignee test set (data/sets.json, from build_sets.py):
`gh api repos/microsoft/vscode/issues/N/timeline?per_page=100 --paginate --slurp`.
Raw timelines are cached in data/raw/timelines/N.json.gz (resumable). The bot's
first assignment is the assignee of the FIRST `assigned` event whose actor is
vs-code-engineering[bot].

Pacing: REST allows ~5,000 requests/hour. The script checks the core budget every
50 issues and sleeps until reset when fewer than 200 requests are left.

Output: data/bot_first_assignment.json and manifest/bot_first_assignment.tsv
        data/timeline_meta.json (first/last fetch time, REST pages, coverage)
Usage : .venv/bin/python scripts/pipeline/pull_timelines.py [--extract-only]
"""
from __future__ import annotations

import json
import sys
import time

from common import (DATA, MANIFEST, REPO, SETS, TIMELINES, TRIAGE_BOT, core_rate, gh, log, now,
                    read_gz, write_gz, write_json)


def fetch(num: int) -> tuple[list | None, bool]:
    """Return (events or None if the issue is gone, fetched_now)."""
    path = TIMELINES / f"{num}.json.gz"
    if path.exists():
        d = json.loads(read_gz(path))
        return d["events"], False
    try:
        out = gh(["api", f"repos/{REPO}/issues/{num}/timeline?per_page=100", "--paginate", "--slurp"])
        pages = json.loads(out)
        d = {"number": num, "fetched_at": now(), "pages": len(pages),
             "events": [e for page in pages for e in page]}
    except FileNotFoundError as e:
        log(f"#{num} not available: {e}")
        d = {"number": num, "fetched_at": now(), "pages": 0, "events": None, "error": str(e)[:200]}
    write_gz(path, json.dumps(d, ensure_ascii=False, sort_keys=True))
    return d["events"], True


def pace() -> None:
    left, reset = core_rate()
    if left < 200:
        wait = max(0, reset - int(time.time())) + 10
        log(f"core budget {left} left, sleeping {wait}s until reset")
        time.sleep(wait)


def main() -> None:
    extract_only = "--extract-only" in sys.argv
    TIMELINES.mkdir(parents=True, exist_ok=True)
    test = json.load(open(SETS))["assignee"]["test"]
    meta_f = DATA / "timeline_meta.json"
    meta = {}
    out, fetched, missing = {}, 0, 0
    for i, it in enumerate(test):
        num = it["number"]
        if extract_only:
            p = TIMELINES / f"{num}.json.gz"
            events = json.loads(read_gz(p))["events"] if p.exists() else None
        else:
            if i % 50 == 0:
                pace()
            events, new = fetch(num)
            fetched += new
        if events is None:
            missing += 1
            out[str(num)] = {"bot_assignee": None, "bot_assigned_at": None, "timeline": False,
                             "final_assignees": it["truth"]}
            continue
        first = next((e for e in events if e.get("event") == "assigned"
                      and (e.get("actor") or {}).get("login") == TRIAGE_BOT), None)
        out[str(num)] = {
            "bot_assignee": ((first or {}).get("assignee") or {}).get("login"),
            "bot_assigned_at": (first or {}).get("created_at"),
            "timeline": True,
            "final_assignees": it["truth"],
        }
        if (i + 1) % 250 == 0:
            log(f"{i + 1}/{len(test)} done ({fetched} fetched this run)")
    # Pull window = first .. last `fetched_at` over the cached timelines of this test set
    # (robust to interrupted and resumed runs).
    pages, stamps = 0, []
    for it in test:
        p = TIMELINES / f"{it['number']}.json.gz"
        if p.exists():
            d = json.loads(read_gz(p))
            pages += d.get("pages", 0)
            stamps.append(d["fetched_at"])
    meta["first_fetched_at"] = min(stamps) if stamps else None
    meta["last_fetched_at"] = max(stamps) if stamps else None
    n_bot = sum(v["bot_assignee"] is not None for v in out.values())
    n_bot_in_final = sum(v["bot_assignee"] in v["final_assignees"] for v in out.values()
                         if v["bot_assignee"] is not None)
    meta.update({"issues": len(test), "timelines_available": len(test) - missing,
                 "timelines_missing": missing, "rest_pages": pages,
                 "with_bot_first_assignment": n_bot,
                 "bot_first_assignee_in_final_assignees": n_bot_in_final})
    write_json(meta_f, meta)
    write_json(DATA / "bot_first_assignment.json", out)
    lines = ["number\tbot_first_assignee\tbot_assigned_at\ttimeline_available"]
    for k in sorted(out, key=int):
        v = out[k]
        lines.append(f"{k}\t{v['bot_assignee'] or ''}\t{v['bot_assigned_at'] or ''}\t{int(v['timeline'])}")
    (MANIFEST / "bot_first_assignment.tsv").write_text("\n".join(lines) + "\n")
    log(f"{n_bot}/{len(test)} assignee test issues have a bot first assignment; missing timelines: {missing}")


if __name__ == "__main__":
    main()
