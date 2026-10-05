"""D1 (part D): collect `Issue.duplicateOf` for every issue created 2023-01..2026-09.

Why a separate pass: the MARKED_AS_DUPLICATE timeline event is only listed on
the *canonical* issue's timeline (checked on #278739 / #248054, REST and
GraphQL), so the duplicate side cannot be read from its own timeline. The
GraphQL field `Issue.duplicateOf` ("A reference to the original issue that
this issue has been marked as a duplicate of", from schema introspection)
gives it directly.

Raw pages: data/d1_dupof_graphql/page_NNNNN.json.gz (resumable via state.json).
Output   : data/d1_duplicate_of.json  {issue_number: {"canonical": n, "repo": "owner/name"}}

Usage: python3 scripts/d1_duplicate_of.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time

from common import DATA, gh, write_json

OUT = DATA / "d1_dupof_graphql"
START, END = "2023-01-01T00:00:00Z", "2026-10-01T00:00:00Z"
QUERY = """
query($cursor:String){ rateLimit{cost remaining resetAt}
 repository(owner:"microsoft",name:"vscode"){ issues(first:100, orderBy:{field:CREATED_AT,direction:DESC}, after:$cursor){
  pageInfo{hasNextPage endCursor}
  nodes{ number createdAt duplicateOf{ number repository{nameWithOwner} } } }}}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "state.json"
    st = json.load(open(sf)) if sf.exists() else {"page": 0, "cursor": None, "done": False}
    while not st["done"]:
        args = ["api", "graphql", "-f", f"query={QUERY}"] + (["-f", f"cursor={st['cursor']}"] if st["cursor"] else [])
        raw = gh(args)
        d = json.loads(raw)
        if d.get("errors"):
            print(d["errors"], file=sys.stderr)
            time.sleep(30)
            continue
        with gzip.open(OUT / f"page_{st['page']:05d}.json.gz", "wt") as f:
            f.write(raw)
        c = d["data"]["repository"]["issues"]
        st["page"] += 1
        st["cursor"] = c["pageInfo"]["endCursor"]
        st["done"] = not c["pageInfo"]["hasNextPage"] or c["nodes"][-1]["createdAt"] < START
        write_json(sf, st)
        if d["data"]["rateLimit"]["remaining"] < 50:
            time.sleep(600)
        if st["page"] % 50 == 0:
            print(st["page"], c["nodes"][-1]["createdAt"], d["data"]["rateLimit"], flush=True)
    res = {}
    for p in sorted(OUT.glob("page_*.json.gz")):
        for n in json.load(gzip.open(p, "rt"))["data"]["repository"]["issues"]["nodes"]:
            if START <= n["createdAt"] < END and n["duplicateOf"]:
                res[n["number"]] = {"canonical": n["duplicateOf"]["number"],
                                    "repo": n["duplicateOf"]["repository"]["nameWithOwner"]}
    write_json(DATA / "d1_duplicate_of.json", res)
    print("done", len(res))


if __name__ == "__main__":
    main()
