"""Module label list for the "module" judgment (spec A7, revised 2026-10-07).

Module labels are the keys of `labels` in vscode's `.github/classifier.json`. The file is part
of the snapshot at data/raw/classifier.json (checksummed by archive.py like every raw file).

Provenance of data/raw/classifier.json:
  * copied from the PoC copy (poc/feasibility/data/classifier.json), which was fetched from the
    vscode default branch on 2026-10-03 (file time 2026-10-03T20:12:32Z);
  * its git blob id equals the blob of `.github/classifier.json` at vscode commit
    9de77c29 (2026-05-22), which a single GraphQL query on 2026-10-07T01:58Z
    (`history(path: ".github/classifier.json")` on the default branch) returned as the latest
    commit touching the file. The script verifies the blob id on every run.

Rule: a classifier.json key is a module label only if it occurs as a label on at least one
pulled issue (created 2023-01-01..2026-09-30). The snapshot holds no repository label list,
so "is an actual repo label" is checked against the labels present in the snapshot. Keys that
never occur cannot be ground truth; dropping them only shrinks the label space of baselines.

Output (tracked): manifest/module_labels.json
Usage : .venv/bin/python scripts/pipeline/module_labels.py           # verify + derive (offline)
        .venv/bin/python scripts/pipeline/module_labels.py --fetch   # download the pinned commit's
                                                                     # file first if it is missing
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

from common import DATA, MANIFEST, REPO, gh, load_issues, log, sha256_file, write_json

CLASSIFIER = DATA / "raw" / "classifier.json"
SOURCE_PATH = ".github/classifier.json"
SOURCE_URL = f"https://github.com/{REPO}/blob/main/{SOURCE_PATH}"
FETCHED_ON = "2026-10-03"
FETCHED_FILE_TIME = "2026-10-03T20:12:32Z"
COMMIT = "9de77c295b1bd405e4c21e3e089a8af538a0c83f"
COMMIT_DATE = "2026-05-22T18:26:32Z"
COMMIT_HEADLINE = "Add browser-integration assignment to classifier.json (#317925)"
BLOB = "f38027d88bbde65a62e2e38b19c634b80b78ce50"
COMMIT_LOOKUP_AT = "2026-10-07T01:58Z"


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch() -> None:
    """Download the file exactly as it was at the pinned commit (one REST request)."""
    raw = gh(["api", "-H", "Accept: application/vnd.github.raw",
              f"repos/{REPO}/contents/{SOURCE_PATH}?ref={COMMIT}"])
    CLASSIFIER.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFIER.write_bytes(raw.encode("utf-8"))
    log(f"fetched {SOURCE_PATH} at {COMMIT[:8]}")


def main() -> None:
    if "--fetch" in sys.argv and not CLASSIFIER.exists():
        fetch()
    data = CLASSIFIER.read_bytes()
    blob = git_blob_id(data)
    if blob != BLOB:
        raise SystemExit(f"{CLASSIFIER}: git blob {blob} != expected {BLOB} (commit {COMMIT[:8]})")
    keys = sorted(json.loads(data)["labels"])

    seen = Counter(l for r in load_issues().values() for l in r["labels"])
    kept = [k for k in keys if seen[k] > 0]
    dropped = [k for k in keys if seen[k] == 0]
    out = {
        "source": {
            "repo": REPO, "path": SOURCE_PATH, "url": SOURCE_URL,
            "fetched_on": FETCHED_ON, "fetched_file_time_utc": FETCHED_FILE_TIME,
            "commit": COMMIT, "commit_date": COMMIT_DATE, "commit_headline": COMMIT_HEADLINE,
            "git_blob": BLOB,
            "commit_determination": (
                "git blob id of the fetched file equals the blob of the file at this commit; "
                f"one GraphQL history query on {COMMIT_LOOKUP_AT} returned it as the latest "
                "commit touching the file on the default branch"),
            "snapshot_file": str(CLASSIFIER.relative_to(DATA.parent)),
            "sha256": sha256_file(CLASSIFIER),
        },
        "rule": ("module labels = keys of `labels` in classifier.json that occur as a label on at "
                 "least one pulled issue (created 2023-01-01..2026-09-30)"),
        "classifier_keys": len(keys),
        "module_labels_count": len(kept),
        "dropped_never_used_count": len(dropped),
        "module_labels": kept,
        "dropped_never_used": dropped,
    }
    write_json(MANIFEST / "module_labels.json", out)
    print(f"module labels: {len(kept)} of {len(keys)} classifier keys (dropped {len(dropped)} never used)")


if __name__ == "__main__":
    main()
