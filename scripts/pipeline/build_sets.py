"""A7 filter + A8 split: build the per-judgment train / tune / test sets.

Input : data/issues.jsonl.gz (from pull_issues.py --flatten)
Output: data/sets.json                    full sets with ground truth (gitignored)
        manifest/sets/<judgment>_<split>.tsv   issue numbers + ground truth (tracked)
        manifest/counts.json              counts per judgment x split, removals per rule,
                                          exact-duplicate flags (tracked)

Rules (stage1_spec.html A7/A8):
  * every judgment: drop issues whose author is a bot (`is_bot_login`, or GraphQL
    author __typename == "Bot"), then issues whose body is empty (whitespace only);
  * type: exactly one of the 10 TYPE_LABELS (ground truth = that label);
  * module (added 2026-10-07): at least one module label, i.e. a classifier.json key listed in
    manifest/module_labels.json (ground truth = the sorted set of its module labels; issues
    with several module labels are kept and every one of them counts as correct);
  * assignee (discontinued 2026-10-07, still built as a record): at least one final assignee
    (ground truth = all final assignees);
  * duplicate: `duplicateOf` points to microsoft/vscode, the canonical was created
    before the issue and on/after 2023-01-01 (ground truth = canonical number);
  * split by createdAt: train 2025-01-01..2026-03-31, tune 2026-04..06, test 2026-07..09
    (tune widened from 2026-06 on 2026-10-07).
    Duplicate queries retrieve from all pulled issues (2023+) created before them.
  * exact-duplicate texts are kept; a test issue is FLAGGED when its normalized
    `issue_text` equals the text of some issue in the same judgment's train set.

Usage: .venv/bin/python scripts/pipeline/build_sets.py
"""
from __future__ import annotations

from collections import Counter

import json

from common import (ACTIVE_JUDGMENTS, DISCONTINUED, JUDGMENTS, MANIFEST, MODULE_LABELS_FILE,
                    PULL_START, SETS, SPLITS, TYPE_LABELS, is_bot_login, load_issues, log,
                    normalized_text, split_of, write_json)

TYPE_SET = set(TYPE_LABELS)
MODULE_SET = set(json.load(open(MODULE_LABELS_FILE))["module_labels"])


def judge(r: dict, judgment: str) -> tuple[str | None, object]:
    """Return (removal_rule, ground_truth). removal_rule is None when the issue is kept."""
    if judgment == "type":
        types = [l for l in r["labels"] if l in TYPE_SET]
        if not types:
            return "no_type_label", None
        if len(types) > 1:
            return "multiple_type_labels", None
        return None, types[0]
    if judgment == "module":
        mods = sorted(l for l in r["labels"] if l in MODULE_SET)
        if not mods:
            return "no_module_label", None
        return None, mods
    if judgment == "assignee":
        if not r["assignees"]:
            return "no_assignee", None
        return None, r["assignees"]
    if judgment == "duplicate":
        d = r["duplicate_of"]
        if not d:
            return "no_duplicate_of", None
        if d["repo"].lower() != "microsoft/vscode":
            return "canonical_other_repo", None
        if not d["created_at"] < r["created_at"]:
            return "canonical_not_earlier", None
        if d["created_at"] < PULL_START:
            return "canonical_before_2023", None
        return None, d["number"]
    raise ValueError(judgment)


RULE_ORDER = {
    "type": ["bot_author", "empty_body", "no_type_label", "multiple_type_labels"],
    "module": ["bot_author", "empty_body", "no_module_label"],
    "assignee": ["bot_author", "empty_body", "no_assignee"],
    "duplicate": ["bot_author", "empty_body", "no_duplicate_of", "canonical_other_repo",
                  "canonical_not_earlier", "canonical_before_2023"],
}


def is_bot_author(r: dict) -> bool:
    return is_bot_login(r["author"]) or r.get("author_type") == "Bot"


def gt_str(judgment: str, gt) -> str:
    return ",".join(gt) if judgment in ("assignee", "module") else str(gt)


def main() -> None:
    issues = load_issues()
    log(f"{len(issues)} issues loaded")
    sets = {j: {s: [] for s in SPLITS} for j in JUDGMENTS}
    removed = {j: {s: Counter() for s in SPLITS} for j in JUDGMENTS}
    in_split = Counter()
    for num in sorted(issues):
        r = issues[num]
        s = split_of(r["created_at"])
        if s is None:
            continue  # 2023-2024: only part of the duplicate retrieval corpus
        in_split[s] += 1
        bot = is_bot_author(r)
        empty = not (r["body"] or "").strip()
        for j in JUDGMENTS:
            if bot:
                removed[j][s]["bot_author"] += 1
                continue
            if empty:
                removed[j][s]["empty_body"] += 1
                continue
            rule, gt = judge(r, j)
            if rule:
                removed[j][s][rule] += 1
            else:
                sets[j][s].append({"number": num, "created_at": r["created_at"], "truth": gt})

    # Exact-duplicate flags: test issue text == some train issue text, per judgment.
    flags = {}
    for j in JUDGMENTS:
        train_texts: dict[str, int] = {}
        for it in sets[j]["train"]:
            r = issues[it["number"]]
            train_texts.setdefault(normalized_text(r["title"], r["body"]), it["number"])
        flagged = []
        for it in sets[j]["test"]:
            r = issues[it["number"]]
            m = train_texts.get(normalized_text(r["title"], r["body"]))
            it["exact_dup_of_train"] = m
            if m is not None:
                flagged.append({"number": it["number"], "train_match": m})
        flags[j] = flagged

    # Extra diagnostics (reported, not used as filters).
    diag = {}
    diag["bot_by_typename_only"] = sum(1 for r in issues.values() if split_of(r["created_at"])
                                       and r["author_type"] == "Bot" and not is_bot_login(r["author"]))
    dq = [it for s in SPLITS for it in sets["duplicate"][s]]
    diag["duplicate_canonical_missing_from_pull"] = sum(1 for it in dq if it["truth"] not in issues)
    diag["duplicate_canonical_bot_authored"] = sum(
        1 for it in dq if it["truth"] in issues and is_bot_author(issues[it["truth"]]))
    diag["duplicate_corpus_size"] = len(issues)
    diag["type_label_distribution_test"] = dict(Counter(it["truth"] for it in sets["type"]["test"]).most_common())
    diag["type_label_distribution_train"] = dict(Counter(it["truth"] for it in sets["type"]["train"]).most_common())

    # Module diagnostics: multi-module share, label distribution, and the labels most often
    # carried by issues dropped for having no module label (classifier.json is outdated).
    multi = {}
    for s in SPLITS:
        n = len(sets["module"][s])
        k = sum(1 for it in sets["module"][s] if len(it["truth"]) > 1)
        multi[s] = {"issues": n, "multi_module": k, "share": k / n if n else None,
                    "modules_per_issue": dict(sorted(Counter(len(it["truth"]) for it in sets["module"][s]).items()))}
    diag["module_multi_label"] = multi
    for s in ("train", "test"):
        diag[f"module_label_distribution_{s}"] = dict(
            Counter(m for it in sets["module"][s] for m in it["truth"]).most_common())
    train_mods = set(diag["module_label_distribution_train"])
    test_unseen = sorted(set(diag["module_label_distribution_test"]) - train_mods)
    diag["module_test_labels_not_in_train"] = test_unseen
    diag["module_test_issues_with_label_not_in_train"] = sum(
        1 for it in sets["module"]["test"] if set(it["truth"]) - train_mods)
    kept_mod = {it["number"] for s in SPLITS for it in sets["module"][s]}
    no_mod_test = [r for r in issues.values() if split_of(r["created_at"]) == "test"
                   and not is_bot_author(r) and (r["body"] or "").strip() and r["number"] not in kept_mod]
    diag["module_dropped_test_top_labels"] = dict(
        Counter(l for r in no_mod_test for l in r["labels"]).most_common(30))
    diag["module_dropped_test_issues"] = len(no_mod_test)
    diag["module_dropped_test_without_any_label"] = sum(1 for r in no_mod_test if not r["labels"])

    counts = {
        "issues_pulled": len(issues),
        "issues_per_split_before_filter": dict(in_split),
        "kept": {j: {s: len(sets[j][s]) for s in SPLITS} for j in JUDGMENTS},
        "removed": {j: {s: {rule: removed[j][s][rule] for rule in RULE_ORDER[j]} for s in SPLITS}
                    for j in JUDGMENTS},
        "rule_order": RULE_ORDER,
        "active_judgments": ACTIVE_JUDGMENTS,
        "discontinued": DISCONTINUED,
        "module_labels": {"file": "manifest/module_labels.json", "count": len(MODULE_SET)},
        "exact_dup_flagged_test": {j: len(flags[j]) for j in JUDGMENTS},
        "exact_dup_flags": flags,
        "diagnostics": diag,
    }
    for j in JUDGMENTS:
        for s in SPLITS:
            assert in_split[s] == len(sets[j][s]) + sum(removed[j][s].values())

    write_json(SETS, sets)
    write_json(MANIFEST / "counts.json", counts)
    for j in JUDGMENTS:
        for s in SPLITS:
            lines = ["number\tcreated_at\ttruth" + ("\texact_dup_of_train" if s == "test" else "")]
            for it in sets[j][s]:
                row = [str(it["number"]), it["created_at"], gt_str(j, it["truth"])]
                if s == "test":
                    row.append(str(it["exact_dup_of_train"] or ""))
                lines.append("\t".join(row))
            p = MANIFEST / "sets" / f"{j}_{s}.tsv"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + "\n")
    log(f"kept: {counts['kept']}")
    log(f"flagged: {counts['exact_dup_flagged_test']}")


if __name__ == "__main__":
    main()
