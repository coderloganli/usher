"""A6: freeze the snapshot - pull times, issue-number list and SHA-256 of every data file.

Outputs (tracked):
  manifest/snapshot.json      pull start/finish times, counts, the GraphQL query, file count
  manifest/issue_numbers.txt  every pulled issue number (created 2023-01-01..2026-09-30)
  manifest/issue_hashes.tsv   per issue: first 16 hex chars of SHA-256 over its canonical JSON
                              record (number, created_at, title, body, author, labels,
                              assignees, duplicate_of), so a re-pull can be compared issue by issue
  manifest/checksums.sha256   SHA-256 of every file under data/ (sha256sum format, paths
                              relative to the repo root). gzip files are written with mtime=0.

Usage: .venv/bin/python scripts/pipeline/archive.py
"""
from __future__ import annotations

import hashlib
import json

from common import DATA, MANIFEST, PAGES, ROOT, load_issues, now, sha256_file, write_json

HASH_FIELDS = ["number", "created_at", "title", "body", "author", "labels", "assignees", "duplicate_of"]


def record_hash(r: dict) -> str:
    canon = json.dumps({k: r[k] for k in HASH_FIELDS}, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    issues = load_issues()
    nums = sorted(issues)
    (MANIFEST / "issue_numbers.txt").write_text("\n".join(map(str, nums)) + "\n")
    (MANIFEST / "issue_hashes.tsv").write_text(
        "number\tsha256_16\n" + "".join(f"{n}\t{record_hash(issues[n])}\n" for n in nums))

    files = sorted(p for p in DATA.rglob("*") if p.is_file() and "logs" not in p.parts
                   and not p.name.endswith(".tmp"))
    lines = [f"{sha256_file(p)}  {p.relative_to(ROOT)}" for p in files]
    (MANIFEST / "checksums.sha256").write_text("\n".join(lines) + "\n")

    pull = json.load(open(DATA / "pull_meta.json"))
    tl = json.load(open(DATA / "timeline_meta.json"))
    st = json.load(open(PAGES / "state.json"))
    snap = {
        "archived_at": now(),
        "repo": "microsoft/vscode",
        "issue_pull": {"started_at": pull["pull_started_at"], "finished_at": pull["pull_finished_at"],
                       "range_utc": pull["range"], "issues": pull["issues"], "graphql_pages": pull["pages"],
                       "truncated_connections": pull["truncated_connections"], "limits": pull["limits"],
                       "query": st["query"]},
        "timeline_pull": {k: tl.get(k) for k in ("first_fetched_at", "last_fetched_at", "issues", "timelines_available",
                                                 "timelines_missing", "rest_pages")},
        "files_checksummed": len(files),
        "key_files": {str(p.relative_to(ROOT)): sha256_file(p)
                      for p in [DATA / "issues.jsonl.gz", DATA / "sets.json", DATA / "bot_first_assignment.json"]},
        "issue_numbers_sha256": sha256_file(MANIFEST / "issue_numbers.txt"),
    }
    write_json(MANIFEST / "snapshot.json", snap)
    print(f"snapshot: {len(nums)} issues, {len(files)} files checksummed")


if __name__ == "__main__":
    main()
