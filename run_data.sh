#!/usr/bin/env bash
# Rerun the whole stage-1 data pipeline (spec A5-A9).
#   ./run_data.sh            pull (resumable) + filter/split + timelines + assignee check + archive
#                            + review sheet + AI review summary + report
#   ./run_data.sh --offline  skip all GitHub requests; rebuild everything from cached raw data
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
S=scripts/pipeline
OFFLINE=0; [[ "${1:-}" == "--offline" ]] && OFFLINE=1

[[ $OFFLINE == 0 ]] && $PY $S/pull_issues.py           # A5: GraphQL bulk pull
$PY $S/pull_issues.py --flatten                        # A5: raw pages -> data/issues.jsonl.gz
$PY $S/build_sets.py                                   # A7 + A8: filter, split, exact-dup flags
if [[ $OFFLINE == 0 ]]; then
  $PY $S/pull_timelines.py                             # A5: REST timelines of assignee test issues
else
  $PY $S/pull_timelines.py --extract-only
fi
$PY $S/assignee_check.py                               # A9: assignee follow-through check (cached timelines only)
$PY $S/archive.py                                      # A6: snapshot manifest + checksums
$PY $S/review_sheet.py                                 # A9: review sample + sheet (assignee section no longer used)
$PY $S/ai_review_summary.py                            # A9: counts + Wilson CI of the AI review CSVs in review/
$PY $S/report.py                                       # A9: data report
