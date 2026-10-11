"""Build a "code path -> module" table from how train-period module issues were fixed.

Support tool for the agent pilot (agreed with the user on 2026-10-10). For every train-period
module issue (created 2025-01..2026-03), GitHub GraphQL gives the pull requests and commits that
closed it (Issue.closedByPullRequestsReferences, merged only; ClosedEvent.closer). The files each
fix changed are computed locally from the blobless vscode clone (diff against the first parent).

No look-ahead: only fixes whose commit is dated before 2026-04-01T00:00:00Z are used, the same
cut-off as the code snapshot, so the table is usable for every tune-period issue. Fixes that
change more than 50 files are skipped (mass refactors say little about ownership).

Every changed file adds the issue's module labels to the file and to each of its parent
directories; one issue counts at most once per path.

Outputs
  data/agent_pilot/fix_links.json                   issue -> fix commits (cached GraphQL results)
  data/agent_pilot/path_modules.json                path -> {module: issue count}
  analysis/agent_module_pilot/path_map_stats.json   coverage numbers
Usage: .venv/bin/python analysis/agent_module_pilot/build_path_map.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from common import load_sets, write_json  # noqa: E402

CACHE = ROOT / "data" / "agent_pilot"
REPO = ROOT / "data" / "vscode_repo"
CUTOFF = "2026-04-01T00:00:00Z"
MAX_FILES = 50
BATCH = 40

FIELDS = ("closedByPullRequestsReferences(first:10,includeClosedPrs:false){nodes{number merged mergeCommit{oid}}} "
          "timelineItems(itemTypes:[CLOSED_EVENT],first:10){nodes{... on ClosedEvent{closer{__typename "
          "... on Commit{oid} ... on PullRequest{number merged mergeCommit{oid}}}}}}")


def fetch_links(numbers: list[int]) -> dict[str, list[str]]:
    path = CACHE / "fix_links.json"
    links = json.load(open(path)) if path.exists() else {}
    todo = [n for n in numbers if str(n) not in links]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        q = "query{repository(owner:\"microsoft\",name:\"vscode\"){" + "".join(
            f"i{n}:issue(number:{n}){{{FIELDS}}}" for n in chunk) + "}}"
        p = subprocess.run(["gh", "api", "graphql", "-f", f"query={q}"], capture_output=True, text=True)
        repo = (json.loads(p.stdout or "{}").get("data") or {}).get("repository") or {}
        for n in chunk:
            it = repo.get(f"i{n}")
            if it is None:
                links[str(n)] = []  # deleted, transferred or not readable
                continue
            oids = set()
            for pr in it["closedByPullRequestsReferences"]["nodes"]:
                if pr["merged"] and pr["mergeCommit"]:
                    oids.add(pr["mergeCommit"]["oid"])
            for ev in it["timelineItems"]["nodes"]:
                c = (ev or {}).get("closer") or {}
                if c.get("__typename") == "Commit":
                    oids.add(c["oid"])
                elif c.get("__typename") == "PullRequest" and c.get("merged") and c.get("mergeCommit"):
                    oids.add(c["mergeCommit"]["oid"])
            links[str(n)] = sorted(oids)
        write_json(path, links)
        print(f"links {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    return links


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def main() -> None:
    train = load_sets("module")["train"]
    truth = {x["number"]: x["truth"] for x in train}
    links = fetch_links(list(truth))

    stats = Counter()
    path_issues: dict[str, set[int]] = defaultdict(set)
    for n, labels in truth.items():
        oids = links.get(str(n), [])
        stats["issues"] += 1
        stats["issues_with_fix_link"] += bool(oids)
        files: set[str] = set()
        for oid in oids:
            ts = git("show", "-s", "--format=%ct", oid).strip()
            if not ts:
                stats["commits_missing_locally"] += 1
                continue
            if int(ts) >= CUTOFF_TS:
                stats["commits_after_cutoff"] += 1
                continue
            changed = [f for f in git("diff-tree", "-r", "--name-only", "--no-commit-id", f"{oid}^1", oid).splitlines() if f]
            if len(changed) > MAX_FILES:
                stats["commits_too_large"] += 1
                continue
            stats["commits_used"] += 1
            files.update(changed)
        if not files:
            continue
        stats["issues_used"] += 1
        for f in files:
            parts = f.split("/")
            for k in range(1, len(parts) + 1):
                path_issues["/".join(parts[:k])].add(n)

    table = {p: dict(Counter(m for n in ns for m in truth[n]).most_common())
             for p, ns in sorted(path_issues.items())}
    write_json(CACHE / "path_modules.json", table)
    stats["paths"] = len(table)
    stats["module_labels_covered"] = len({m for n in {n for ns in path_issues.values() for n in ns} for m in truth[n]})
    write_json(HERE / "path_map_stats.json", {"cutoff": CUTOFF, "max_files_per_commit": MAX_FILES, **stats})
    print(dict(stats))


CUTOFF_TS = 1775001600  # 2026-04-01T00:00:00Z

if __name__ == "__main__":
    main()
