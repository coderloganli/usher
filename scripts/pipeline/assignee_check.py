"""A9 (revised 2026-10-07): did any FINAL assignee actually act on the issue?

The assignee ground truth is the maintainers' final decision. Whether that decision was
right cannot be judged from public content, so instead we check whether it was carried
out: for every issue in the assignee TEST set we look at the cached REST timeline
(data/raw/timelines/N.json.gz, from pull_timelines.py) and ask whether any final assignee
is the actor of an event, other than being assigned / unassigned by someone else.

Each issue gets ONE category: the strongest action by any final assignee, in this order
(see CATEGORIES below for the exact event rules):

  closed              a final assignee is the actor of a `closed` event.
  code_reference      a final assignee referenced the issue from code: a `referenced` event
                      (a commit that mentions the issue), a `cross-referenced` event whose
                      source is a pull request, or a `connected` event (linked a PR).
  commented           a final assignee wrote a comment (`commented` event).
  issue_reference     a final assignee mentioned the issue from another issue (a
                      `cross-referenced` event whose source is not a pull request).
  other_edit          any other event with a final assignee as actor: labels, milestone,
                      title, reopen, lock, issue type, project fields, marking as duplicate,
                      assigning or unassigning someone ELSE, etc.
  self_assigned_only  the only events by a final assignee are assigning / unassigning
                      themselves.
  none                no event at all by any final assignee.

Passive events are ignored even though GitHub records the person as `actor`:
`mentioned` (someone else @-mentioned them), `subscribed` / `unsubscribed` (GitHub
subscribes assignees automatically), and `assigned` / `unassigned` where the actor is
not the assignee (someone else assigned them).

Opening the issue is not a timeline event, so it is not counted as an action. We report
separately how many issues were authored by one of their final assignees, and, for the
`none` group, whether the issue was still open when its timeline was fetched, closed by the
triage bot, or closed by another person.

Inputs : data/sets.json (assignee test set), data/raw/timelines/, data/bot_first_assignment.json,
         data/issues.jsonl.gz (issue author)
Outputs: manifest/assignee_check.json  category definitions and counts (overall, by bot
                                       subset, by creation month)
         manifest/assignee_check.tsv   per issue: number, category, acting assignees, bot subset
Offline: reads cached data only, never calls GitHub.
Usage  : .venv/bin/python scripts/pipeline/assignee_check.py
"""
from __future__ import annotations

import json

from common import DATA, MANIFEST, SETS, TIMELINES, TRIAGE_BOT, load_issues, log, read_gz, write_json

# (key, Chinese name, Chinese definition) - strongest first.
CATEGORIES = [
    ("closed", "关闭了 issue", "某位最终负责人是这条 issue 的“关闭”（closed）操作的执行者。"),
    ("code_reference", "通过提交或 PR 关联",
     "某位最终负责人在代码层面关联了这条 issue：提交的说明里提到了它（referenced 事件），"
     "或在一个 PR 中引用了它（来源是 PR 的 cross-referenced 事件），或把一个 PR 关联到它（connected 事件）。"),
    ("commented", "评论", "某位最终负责人在这条 issue 下发表过评论（commented 事件）。"),
    ("issue_reference", "在其他 issue 中引用",
     "某位最终负责人在另一条 issue（不是 PR）中提到了这条 issue（来源不是 PR 的 cross-referenced 事件）。"),
    ("other_edit", "其他编辑操作",
     "某位最终负责人做过上面以外的操作，例如加减标签、设置里程碑、修改标题、重新打开、锁定、"
     "设置 issue 类型或项目字段、标为重复，或者把这条 issue 分派给别人、取消别人的分派。"),
    ("self_assigned_only", "只做了自我分派", "最终负责人做过的唯一操作是把自己分派到这条 issue（或取消自己的分派）。"),
    ("none", "没有任何操作", "在时间线里找不到任何由最终负责人执行的操作。"),
]
ORDER = [k for k, _, _ in CATEGORIES]
PASSIVE = {"mentioned", "subscribed", "unsubscribed"}
BOT_SUBSETS = [
    ("bot_in_final", "bot 首次分派的人在最终负责人中"),
    ("bot_not_in_final", "bot 首次分派的人不在最终负责人中"),
    ("no_bot_assignment", "bot 没有分派过"),
]


def login(u) -> str | None:
    return (u or {}).get("login")


def event_category(e: dict, finals: set[str]) -> str | None:
    """Category of one event if a final assignee performed it, else None."""
    t = e.get("event")
    actor = login(e.get("actor")) or (login(e.get("user")) if t == "commented" else None)
    if actor not in finals or t in PASSIVE:
        return None
    if t in ("assigned", "unassigned"):
        return "self_assigned_only" if login(e.get("assignee")) == actor else "other_edit"
    if t == "closed":
        return "closed"
    if t in ("referenced", "connected"):
        return "code_reference"
    if t == "cross-referenced":
        src = (e.get("source") or {}).get("issue") or {}
        return "code_reference" if src.get("pull_request") else "issue_reference"
    if t == "commented":
        return "commented"
    return "other_edit"


def classify(events: list[dict], finals: set[str]) -> tuple[str, list[str]]:
    best, acting = len(ORDER) - 1, set()
    for e in events:
        c = event_category(e, finals)
        if c is None:
            continue
        acting.add(login(e.get("actor")) or login(e.get("user")))
        best = min(best, ORDER.index(c))
    return ORDER[best], sorted(acting)


def tally(rows: list[dict]) -> dict:
    n = len(rows)
    counts = {k: sum(r["category"] == k for r in rows) for k in ORDER}
    return {
        "issues": n,
        "counts": counts,
        "any_action": n - counts["none"],
        "substantive_action": n - counts["none"] - counts["self_assigned_only"],
        "authored_by_final_assignee": sum(r["author_is_final"] for r in rows),
        "authored_by_final_assignee_among_none": sum(r["author_is_final"] for r in rows if r["category"] == "none"),
        "none_still_open": sum(r["category"] == "none" and not r["closed"] for r in rows),
        "none_closed_by_bot": sum(r["category"] == "none" and r["closed_by"] == TRIAGE_BOT for r in rows),
        "none_closed_by_other_person": sum(r["category"] == "none" and r["closed"] and r["closed_by"] != TRIAGE_BOT
                                           for r in rows),
    }


def state(events: list[dict]) -> tuple[bool, str | None]:
    """(closed at timeline fetch time, actor of the last close) from the last closed/reopened event."""
    last = None
    for e in events:
        if e.get("event") in ("closed", "reopened"):
            last = e
    if last is None or last["event"] == "reopened":
        return False, None
    return True, login(last.get("actor"))


def main() -> None:
    test = json.load(open(SETS))["assignee"]["test"]
    bot = json.load(open(DATA / "bot_first_assignment.json"))
    issues = load_issues()
    rows, missing = [], 0
    for it in test:
        num, finals = it["number"], set(it["truth"])
        p = TIMELINES / f"{num}.json.gz"
        events = json.loads(read_gz(p))["events"] if p.exists() else None
        if events is None:
            missing += 1
            continue
        cat, acting = classify(events, finals)
        b = bot.get(str(num), {}).get("bot_assignee")
        subset = "no_bot_assignment" if b is None else ("bot_in_final" if b in finals else "bot_not_in_final")
        rows.append({"number": num, "created_at": it["created_at"], "category": cat, "acting": acting,
                     "bot_subset": subset, "author_is_final": issues[num]["author"] in finals,
                     **dict(zip(("closed", "closed_by"), state(events)))})

    months = sorted({r["created_at"][:7] for r in rows})
    out = {
        "definition": "For each assignee test issue: the strongest action (in category order) taken by any "
                      "final assignee, read from the cached REST timeline. Passive events (mentioned, "
                      "subscribed, unsubscribed) and being assigned/unassigned by someone else are not actions.",
        "categories": [{"key": k, "name_zh": n, "definition_zh": d} for k, n, d in CATEGORIES],
        "passive_events_ignored": sorted(PASSIVE),
        "timelines_missing": missing,
        "overall": tally(rows),
        "by_bot_subset": {k: {"name_zh": n, **tally([r for r in rows if r["bot_subset"] == k])}
                          for k, n in BOT_SUBSETS},
        "by_month": {m: tally([r for r in rows if r["created_at"][:7] == m]) for m in months},
    }
    write_json(MANIFEST / "assignee_check.json", out)
    lines = ["number\tcategory\tacting_final_assignees\tbot_subset\tauthor_is_final_assignee"]
    lines += [f"{r['number']}\t{r['category']}\t{';'.join(r['acting'])}\t{r['bot_subset']}\t{int(r['author_is_final'])}"
              for r in sorted(rows, key=lambda r: r["number"])]
    (MANIFEST / "assignee_check.tsv").write_text("\n".join(lines) + "\n")
    o = out["overall"]
    log(f"assignee check: {o['issues']} issues, none={o['counts']['none']}, "
        f"self_assigned_only={o['counts']['self_assigned_only']}, missing timelines={missing}")


if __name__ == "__main__":
    main()
