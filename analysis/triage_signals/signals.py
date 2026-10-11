"""Triage-flow facts behind the type and module judgments (2026-10-10, discussion with the user).

Questions: does the reporter already give the type? Who applies type labels, and when? What does
a type label change downstream? How does vscode assign owners? Only train and tune issues are
used for statistics on the sets; the bot-assignment figures reuse stage-1 data (test period).

Parts
  A  reporter's own choice (bug / feature template, or the Type field of VS Code's issue reporter)
     against the final type label, train + tune type issues; share of issues by team members.
     "Team member" is approximated by "account that appears as an assignee anywhere in the
     snapshot" (an approximation, not a membership list).
  B  how many issues created 2025-01..2026-06 (bots and empty bodies removed) ever got a type label,
     by author group, and the most common labels on outside issues without one.
  C  who applied the type label and how long after creation: 60 team + 60 outside tune issues
     (random, seed 0), GitHub GraphQL LabeledEvent (first event for the final type label).
  D  current state and milestone of outside tune issues by type (random, seed 1; up to 80 bug,
     80 feature-request, 20 *question, 20 under-discussion, 20 polish, 15 upstream).
  E  vscode's own bot: first bot assignment on stage-1 test issues (data/bot_first_assignment.json).
  F  owners and feature areas in vscode's current .github/classifier.json.
The vscode wiki pages read on 2026-10-10 (Issues-Triaging, Automated-Issue-Triaging) are summarised
in EXPERIMENTS.md, not here.

GitHub responses (C, D, F) are cached in data/triage_signals/ (gitignored); a rerun without the
cache queries GitHub again and the live state (D) may have changed.
Output: analysis/triage_signals/summary.json
Usage:  .venv/bin/python analysis/triage_signals/signals.py
"""
from __future__ import annotations

import base64
import json
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from common import load_issues, write_json  # noqa: E402

CACHE = ROOT / "data" / "triage_signals"
TYPES = {"bug", "feature-request", "*question", "under-discussion", "debt", "polish", "upstream",
         "engineering", "upstream-issue-linked", "electron"}


def gh_graphql(query: str, key: str) -> dict:
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.load(open(path))
    p = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True)
    data = (json.loads(p.stdout or "{}").get("data") or {}).get("repository") or {}
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w"))
    return data


def user_choice(body: str) -> str | None:
    m = re.search(r"^\s*Type:\s*<b>([^<]+)</b>", body, re.M)
    if m:
        return {"Bug": "bug", "Feature Request": "feature-request", "Performance Issue": "bug"}.get(m.group(1).strip())
    if "bug_report_template" in body or "Copilot Chat Extension Version" in body:
        return "bug"
    if "feature_request_template" in body:
        return "feature-request"
    return None


def main() -> None:
    issues = load_issues()
    sets = json.load(open(ROOT / "data" / "sets.json"))["type"]
    truth = {x["number"]: x["truth"] for sp in ("train", "tune") for x in sets[sp]}
    team = set()
    for r in issues.values():
        team.update(r["assignees"] or [])
    out = {"team_proxy_accounts": len(team)}

    # A
    n_team = sum(issues[n]["author"] in team for n in truth)
    outside = [n for n in truth if issues[n]["author"] not in team]
    chosen = [(n, user_choice(issues[n]["body"] or "")) for n in outside]
    with_choice = [(n, c) for n, c in chosen if c]
    wrong = Counter((c, truth[n]) for n, c in with_choice if c != truth[n])
    out["A_reporter_choice"] = {
        "type_issues_train_tune": len(truth), "by_team_proxy": n_team, "by_outside": len(outside),
        "outside_with_choice": len(with_choice),
        "choice_matches_final_type": sum(c == truth[n] for n, c in with_choice),
        "top_mismatches": [{"chose": c, "final": t, "n": v} for (c, t), v in wrong.most_common(8)]}

    # B
    groups = defaultdict(Counter)
    untyped_labels = Counter()
    for r in issues.values():
        if not ("2025-01" <= r["created_at"][:7] <= "2026-06"):
            continue
        if r["author_type"] == "Bot" or not (r["body"] or "").strip():
            continue
        who = "team" if r["author"] in team else "outside"
        has = any(l in TYPES for l in r["labels"])
        groups[who]["typed" if has else "untyped"] += 1
        if not has and who == "outside":
            untyped_labels.update(r["labels"] or ["(no labels)"])
    out["B_type_label_coverage_2025_01_to_2026_06"] = {
        "by_group": {k: dict(v) for k, v in groups.items()},
        "outside_untyped_top_labels": dict(untyped_labels.most_common(12))}

    # C
    tune = {x["number"]: x["truth"] for x in sets["tune"]}
    rows = [(n, issues[n]["author"] in team, tune[n]) for n in issues if n in tune]
    random.seed(0)
    random.shuffle(rows)
    sample = [r for r in rows if r[1]][:60] + [r for r in rows if not r[1]][:60]
    res = []
    for i in range(0, len(sample), 20):
        chunk = sample[i:i + 20]
        q = "query{repository(owner:\"microsoft\",name:\"vscode\"){" + "".join(
            f"i{n}:issue(number:{n}){{createdAt author{{login}} timelineItems(itemTypes:[LABELED_EVENT],first:30)"
            "{nodes{... on LabeledEvent{createdAt label{name} actor{login __typename}}}}}" for n, _, _ in chunk) + "}}"
        d = gh_graphql(q, f"labeled_{i}")
        for n, is_team, t in chunk:
            it = d.get(f"i{n}")
            ev = [e for e in (it or {}).get("timelineItems", {}).get("nodes", []) if e and e["label"]["name"] == t]
            if not ev:
                res.append({"team": is_team, "by": "unavailable" if not it else "no-event"})
                continue
            e, a = ev[0], ev[0]["actor"] or {}
            login = a.get("login") or ""
            by = "bot" if a.get("__typename") == "Bot" or login.lower().endswith("bot") else \
                ("author" if login == (it["author"] or {}).get("login") else "other-human")
            hours = (datetime.fromisoformat(e["createdAt"].replace("Z", "+00:00"))
                     - datetime.fromisoformat(it["createdAt"].replace("Z", "+00:00"))).total_seconds() / 3600
            res.append({"team": is_team, "by": by, "hours": hours})
    c_out = {}
    for is_team in (True, False):
        r = [x for x in res if x["team"] == is_team]
        h = [x["hours"] for x in r if "hours" in x and x["by"] != "author"]
        ha = [x["hours"] for x in r if x["by"] == "author"]
        c_out["team" if is_team else "outside"] = {
            "n": len(r), "labelled_by": dict(Counter(x["by"] for x in r)),
            "median_hours_not_author": round(statistics.median(h), 2) if h else None,
            "within_24h_not_author": sum(v <= 24 for v in h),
            "median_hours_author": round(statistics.median(ha), 2) if ha else None}
    out["C_who_applies_type_label_tune_sample"] = c_out

    # D
    rows = [n for n in issues if n in tune and issues[n]["author"] not in team]
    random.seed(1)
    random.shuffle(rows)
    by = defaultdict(list)
    for n in rows:
        by[tune[n]].append(n)
    quota = {"bug": 80, "feature-request": 80, "*question": 20, "under-discussion": 20, "polish": 20, "upstream": 15}
    pick = [n for t, k in quota.items() for n in by[t][:k]]
    res_d = defaultdict(Counter)
    for i in range(0, len(pick), 25):
        chunk = pick[i:i + 25]
        q = "query{repository(owner:\"microsoft\",name:\"vscode\"){" + "".join(
            f"i{n}:issue(number:{n}){{state stateReason milestone{{title}}}}" for n in chunk) + "}}"
        d = gh_graphql(q, f"state_{i}")
        for n in chunk:
            it = d.get(f"i{n}")
            if not it:
                continue
            m = (it["milestone"] or {}).get("title")
            mk = "none" if not m else m if m in ("Backlog Candidates",) else \
                "Backlog/On Deck" if m in ("Backlog", "On Deck") else "iteration/release"
            res_d[tune[n]][f"{it['state']}/{it['stateReason'] or '-'} | milestone {mk}"] += 1
    out["D_state_and_milestone_outside_tune_sample"] = {t: dict(c.most_common()) for t, c in res_d.items()}

    # E
    bot = json.load(open(ROOT / "data" / "bot_first_assignment.json"))
    assigned = [v for v in bot.values() if v["bot_assignee"]]
    out["E_vscode_bot_first_assignment_stage1_test"] = {
        "issues": len(bot), "bot_assigned": len(assigned),
        "bot_assignee_among_final_assignees": sum(v["bot_assignee"] in (v["final_assignees"] or []) for v in assigned)}

    # F
    path = CACHE / "classifier.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        b64 = subprocess.run(["gh", "api", "repos/microsoft/vscode/contents/.github/classifier.json", "--jq", ".content"],
                             capture_output=True, text=True).stdout
        path.write_bytes(base64.b64decode(b64))
    labels = json.load(open(path))["labels"]
    per_owner = Counter(a for v in labels.values() for a in (v.get("assign") or []))
    vals = sorted(per_owner.values())
    out["F_classifier_json_current"] = {
        "labels": len(labels), "owners": len(per_owner), "areas_per_owner_median": vals[len(vals) // 2],
        "areas_per_owner_max": vals[-1], "owners_with_one_area": sum(v == 1 for v in vals)}

    write_json(HERE / "summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
