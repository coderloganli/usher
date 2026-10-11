"""How much signal does the vscode code base carry for an issue's module? (no LLM)

Agreed with the user on 2026-10-10 (option A after the agent pilot in analysis/agent_module_pilot/):
a fixed program does the "research" that the gpt-4o-mini agent did badly, on all 986 tune-period
module issues, and the evidence is compared with the stage-1 LR baseline. Analysis only; nothing
here is adopted as a method.

Steps, all fixed before the run:
1. Text: the issue's cleaned title + body (scripts/stage2/dedup_clean.clean_text: templates,
   system-info blocks and images removed).
2. Identifiers, first 15 distinct per issue:
     dotted     dotted names such as setting or command ids (terminal.integrated.fontSize);
                names ending in a common file extension or web domain are skipped
     backtick   text inside `backticks`, 3 to 80 characters
     quoted     text inside double quotes, 6 to 80 characters
3. Search: ripgrep fixed-string over src/, extensions/ and cli/ of the vscode working tree at the
   last main commit before 2026-04-01 (case-sensitive for dotted and backtick, case-insensitive
   for quoted); test folders, test files, .md and localisation files are excluded. An identifier
   found in more than 20 files is treated as too generic and dropped.
4. Path -> module: the table from analysis/agent_module_pilot/build_path_map.py (modules of
   train-period issues whose fixes, committed before 2026-04-01, changed the path). A file
   without data falls back to its parent or grandparent, never to a directory with fewer than
   3 path parts (src/vs, extensions/git are too broad).
5. Score: each file's module counts are normalised to sum 1; an identifier's distribution is the
   mean over its files that have data; the issue's evidence is the sum over its identifiers.

LR baseline: stage-1 module LR (TF-IDF defaults, one-vs-rest, C = 10, balanced, fitted on train),
refitted here. One fixed combination rule, with no tuning: use the evidence top-1 when the issue
has evidence, otherwise the LR top-1. Paired bootstrap (10,000 resamples, fixed seed).

Outputs
  analysis/module_repo_evidence/summary.json        coverage, hit rates, complementarity
  analysis/module_repo_evidence/per_issue.json      number, truth, LR top 3, evidence top 3 (no text)
  data/module_repo_evidence/identifiers.json        identifiers per issue (gitignored: issue text)
Usage: .venv/bin/python analysis/module_repo_evidence/evidence.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "stage2"))
from common import N_BOOT, SEED, issue_text, load_issues, load_sets, write_json  # noqa: E402
from dedup_clean import clean_text  # noqa: E402
from stats import mean_boot_values, paired_difference, resample_indices, summary  # noqa: E402

REPO = ROOT / "data" / "vscode_repo"
REPO_COMMIT = "9ed1e079300da92c7051094a05b4dbe136f9c23e"
PATH_MODULES = ROOT / "data" / "agent_pilot" / "path_modules.json"
CACHE = ROOT / "data" / "module_repo_evidence"
SEARCH_DIRS = ("src", "extensions", "cli")
EXCLUDE_GLOBS = ("!**/test/**", "!**/tests/**", "!**/*.test.ts", "!**/*.md", "!**/i18n/**",
                 "!**/*.nls.*.json", "!**/package-lock.json", "!**/cgmanifest.json")
MAX_IDS, MAX_FILES, MIN_DIR_PARTS, MAX_UP = 15, 20, 3, 2

SKIP_SUFFIX = {"ts", "js", "json", "md", "txt", "log", "exe", "py", "html", "css", "png", "jpg", "gif", "svg",
               "zip", "dll", "so", "sh", "ps1", "yml", "yaml", "xml", "tsx", "jsx", "java", "cs", "go", "rs",
               "cpp", "c", "h", "ipynb", "com", "org", "net", "io", "dev", "ai", "app", "vsix", "dmg", "deb", "rpm"}
DOTTED = re.compile(r"(?<![\w./@-])([A-Za-z][\w-]*(?:\.[A-Za-z][\w-]*)+)(?![\w/(-])")
BACKTICK = re.compile(r"`([^`\n]{3,80})`")
QUOTED = re.compile(r"[\"“]([^\"”\n]{6,80})[\"”]")


def identifiers(text: str) -> list[tuple[str, str]]:
    out, seen = [], set()
    for kind, rx in (("dotted", DOTTED), ("backtick", BACKTICK), ("quoted", QUOTED)):
        for m in rx.finditer(text):
            s = m.group(1).strip()
            if kind == "dotted" and (s.rsplit(".", 1)[-1].lower() in SKIP_SUFFIX or s.lower() in {"e.g", "i.e"}):
                continue
            key = s.lower() if kind == "quoted" else s
            if len(s) < 3 or key in seen:
                continue
            seen.add(key)
            out.append((kind, s))
    # keep text order across kinds, then cap
    pos = {s: text.find(s) for _, s in out}
    return sorted(out, key=lambda x: pos[x[1]])[:MAX_IDS]


def rg(patterns: list[str], ignore_case: bool) -> dict[str, set[str]]:
    """pattern -> files (relative to the repo) containing it."""
    if not patterns:
        return {}
    cmd = ["rg", "-F", "-o", "-H", "--no-line-number", "--no-heading", "--max-columns", "1000"]
    if ignore_case:
        cmd.append("-i")
    for g in EXCLUDE_GLOBS:
        cmd += ["--glob", g]
    for p in patterns:
        cmd += ["-e", p]
    cmd += ["--", *SEARCH_DIRS]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300).stdout
    want = {(p.lower() if ignore_case else p): p for p in patterns}
    found: dict[str, set[str]] = defaultdict(set)
    for line in out.splitlines():
        f, _, match = line.partition(":")
        p = want.get(match.lower() if ignore_case else match)
        if p is not None:
            found[p].add(f)
    return found


def file_modules(f: str, table: dict) -> dict[str, float] | None:
    parts = f.split("/")
    for up in range(0, MAX_UP + 1):
        k = len(parts) - up
        if k < 1 or (up > 0 and k < MIN_DIR_PARTS):
            break
        d = table.get("/".join(parts[:k]))
        if d:
            tot = sum(d.values())
            return {m: c / tot for m, c in d.items()}
    return None


def evidence_for(text: str, table: dict) -> tuple[list[dict], dict[str, float]]:
    ids = identifiers(text)
    cs = [s for k, s in ids if k != "quoted"]
    ci = [s for k, s in ids if k == "quoted"]
    hits = {**rg(cs, False), **rg(ci, True)}
    score: Counter = Counter()
    rows = []
    for kind, s in ids:
        files = hits.get(s, set())
        row = {"kind": kind, "text": s, "files": len(files), "used": False}
        if 0 < len(files) <= MAX_FILES:
            dists = [d for d in (file_modules(f, table) for f in sorted(files)) if d]
            if dists:
                row["used"] = True
                for d in dists:
                    for m, v in d.items():
                        score[m] += v / len(dists)
        rows.append(row)
    return rows, dict(score)


def lr_top10(issues: dict) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    sets = load_sets("module")
    tr, tu = sets["train"], sets["tune"]
    mlb = MultiLabelBinarizer()
    Ytr = mlb.fit_transform([x["truth"] for x in tr])
    classes = np.array(mlb.classes_)
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform([issue_text(issues[x["number"]]) for x in tr])
    Xtu = vec.transform([issue_text(issues[x["number"]]) for x in tu])
    lr = OneVsRestClassifier(LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000,
                                                random_state=SEED), n_jobs=-1).fit(Xtr, Ytr)
    P = lr.predict_proba(Xtu)
    return {x["number"]: [str(c) for c in classes[np.argsort(-p, kind="stable")[:10]]] for x, p in zip(tu, P)}


def main() -> None:
    issues = load_issues()
    tune = load_sets("module")["tune"]
    table = json.load(open(PATH_MODULES, encoding="utf-8"))
    lr = lr_top10(issues)

    def one(x):
        rows, score = evidence_for(clean_text(issues[x["number"]]), table)
        ranked = [m for m, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
        return x["number"], x["truth"], rows, ranked

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, tune))

    hit = lambda truth, ranked, k: bool(set(ranked[:k]) & set(truth))
    n = len(results)
    has_id = np.array([bool(r[2]) for r in results])
    has_match = np.array([any(0 < i["files"] for i in r[2]) for r in results])
    has_ev = np.array([bool(r[3]) for r in results])
    lr1 = np.array([hit(t, lr[num], 1) for num, t, _, _ in results], dtype=float)
    lr3 = np.array([hit(t, lr[num], 3) for num, t, _, _ in results], dtype=float)
    lr10 = np.array([hit(t, lr[num], 10) for num, t, _, _ in results], dtype=float)
    ev1 = np.array([hit(t, ev, 1) for _, t, _, ev in results], dtype=float)
    ev3 = np.array([hit(t, ev, 3) for _, t, _, ev in results], dtype=float)
    ev10 = np.array([hit(t, ev, 10) for _, t, _, ev in results], dtype=float)
    combo = np.where(has_ev, ev1, lr1)
    union = np.maximum(lr1, ev1)

    idx = resample_indices(n, N_BOOT, SEED)
    b_lr, b_combo = mean_boot_values(lr1, idx), mean_boot_values(combo, idx)
    sub = np.where(has_ev)[0]
    idx_s = resample_indices(len(sub), N_BOOT, SEED)
    b_lr_s, b_ev_s = mean_boot_values(lr1[sub], idx_s), mean_boot_values(ev1[sub], idx_s)

    kinds = Counter()
    kind_used = Counter()
    for _, _, rows, _ in results:
        for k in {i["kind"] for i in rows}:
            kinds[k] += 1
        for k in {i["kind"] for i in rows if i["used"]}:
            kind_used[k] += 1

    e = has_ev
    out = {
        "repo_commit": REPO_COMMIT, "tune_issues": n,
        "path_table": "data/agent_pilot/path_modules.json (train-period fixes before 2026-04-01)",
        "coverage": {"with_identifier": int(has_id.sum()), "with_repo_match": int(has_match.sum()),
                     "with_evidence": int(e.sum())},
        "issues_with_kind": dict(kinds), "issues_with_kind_used_as_evidence": dict(kind_used),
        "all_issues": {
            "lr_top1": summary(lr1.mean(), b_lr), "lr_top3": float(lr3.mean()), "lr_top10": float(lr10.mean()),
            "rule_evidence_else_lr_top1": summary(combo.mean(), b_combo),
            "rule_minus_lr_top1": paired_difference(combo.mean(), b_combo, lr1.mean(), b_lr),
            "either_lr_or_evidence_top1_right": float(union.mean()),
        },
        "issues_with_evidence": {
            "n": int(e.sum()),
            "lr_top1": summary(lr1[e].mean(), b_lr_s), "evidence_top1": summary(ev1[e].mean(), b_ev_s),
            "evidence_minus_lr_top1": paired_difference(ev1[e].mean(), b_ev_s, lr1[e].mean(), b_lr_s),
            "lr_top3": float(lr3[e].mean()), "evidence_top3": float(ev3[e].mean()),
            "lr_top10": float(lr10[e].mean()), "evidence_top10": float(ev10[e].mean()),
            "both_right": int(((lr1 == 1) & (ev1 == 1) & e).sum()),
            "only_lr_right": int(((lr1 == 1) & (ev1 == 0) & e).sum()),
            "only_evidence_right": int(((lr1 == 0) & (ev1 == 1) & e).sum()),
            "both_wrong": int(((lr1 == 0) & (ev1 == 0) & e).sum()),
        },
        "issues_without_evidence": {"n": int((~e).sum()), "lr_top1": float(lr1[~e].mean())},
    }
    write_json(HERE / "summary.json", out)
    write_json(HERE / "per_issue.json", [{"number": num, "truth": t, "lr_top3": lr[num][:3], "evidence_top3": ev[:3]}
                                         for num, t, _, ev in results])
    write_json(CACHE / "identifiers.json", {str(num): rows for num, _, rows, _ in results})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
