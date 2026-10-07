"""A9: human review sheet - 30 random TEST issues per judgment (fixed seed).

Output: review/review_sample.json  sampled issue numbers + ground truth (tracked)
        review/review_sheet.html   the sheet; verdicts are left empty for the reviewer.
Verdict choices: 对 / 错 / 无法判断, plus a free-text note. Inputs persist in
localStorage (wrapped in try/catch); each section has an "导出 CSV" button.

Usage: .venv/bin/python scripts/pipeline/review_sheet.py
"""
from __future__ import annotations

import html
import json
import random
import re

from common import JUDGMENTS, REVIEW, SEED, SETS, load_issues, write_json
from page_style import CSS, THEME_JS

N_REVIEW = 30
EXCERPT = 400
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
NAMES = {"type": "类型", "assignee": "负责人", "duplicate": "查重"}
# Spec A9 revised 2026-10-07: type/duplicate are checked by an AI model, assignee is checked
# automatically from timelines. The sheet is kept for reference.
NOTES = {
    "type": '<p class="muted">2026-10-07 起，这一节由 AI 模型（Claude Opus 5.5）核对，结果见 '
            '<code>review/ai_review_type.csv</code>，未经人工核实。</p>',
    "duplicate": '<p class="muted">2026-10-07 起，这一节由 AI 模型（Claude Opus 5.5）核对，结果见 '
                 '<code>review/ai_review_duplicate.csv</code>，未经人工核实。</p>',
    "assignee": '<p class="muted">这一节已不再使用。按 2026-10-07 修订的实验说明 A9，负责人不做主观核对，'
                '改为用操作记录自动检查最终负责人是否实际处理过 issue，结果见 '
                '<code>manifest/assignee_check.json</code> 和数据报告。</p>',
}


def excerpt(body: str) -> str:
    text = " ".join(_HTML_COMMENT.sub("", body or "").split())
    return text[:EXCERPT] + ("…" if len(text) > EXCERPT else "")


def link(n: int) -> str:
    return f'<a href="https://github.com/microsoft/vscode/issues/{n}" target="_blank" rel="noopener">#{n}</a>'


def main() -> None:
    sets = json.load(open(SETS))
    issues = load_issues()
    sample = {}
    for j in JUDGMENTS:
        test = sorted(sets[j]["test"], key=lambda it: it["number"])
        rng = random.Random(SEED)
        sample[j] = sorted(rng.sample(test, min(N_REVIEW, len(test))), key=lambda it: it["number"])
    write_json(REVIEW / "review_sample.json",
               {"seed": SEED, "n_per_judgment": N_REVIEW,
                "sample": {j: [{"number": it["number"], "truth": it["truth"]} for it in sample[j]]
                           for j in JUDGMENTS}})

    sections = []
    for j in JUDGMENTS:
        rows = []
        for i, it in enumerate(sample[j], 1):
            r = issues[it["number"]]
            n = it["number"]
            if j == "type":
                truth = f"<code>{html.escape(it['truth'])}</code>"
            elif j == "assignee":
                truth = ", ".join(html.escape(a) for a in it["truth"])
            else:
                c = issues.get(it["truth"])
                ctitle = html.escape(c["title"]) if c else "(不在拉取范围内)"
                truth = f"重复于 {link(it['truth'])}<br><span class=\"ct\">{ctitle}</span>"
            key = f"{j}:{n}"
            radios = "".join(
                f'<label><input type="radio" name="{key}" value="{v}"> {v}</label>'
                for v in ("对", "错", "无法判断"))
            rows.append(
                f'<tr data-key="{key}" data-num="{n}" data-truth="{html.escape(json.dumps(it["truth"], ensure_ascii=False))}">'
                f'<td class="num">{i}</td>'
                f'<td>{link(n)} <b>{html.escape(r["title"])}</b><div class="ex">{html.escape(excerpt(r["body"])) or "<i>(正文为空)</i>"}</div></td>'
                f'<td>{truth}</td>'
                f'<td class="verdict">{radios}<textarea placeholder="备注（可选）"></textarea></td></tr>')
        sections.append(f"""
<section data-judgment="{j}">
 <h2>{NAMES[j]}（{len(sample[j])} 条）</h2>{NOTES.get(j, "")}
 <p class="bar"><span class="prog"></span> <button class="exp">导出 CSV</button></p>
 <div class="scroll"><table>
  <thead><tr><th class="num">#</th><th>issue（标题与正文前 {EXCERPT} 字）</th><th>正确答案</th><th>判断</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
</section>""")

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Usher 人工核对表</title>
<style>{CSS}
.ex{{color:var(--muted);font-size:13px;margin-top:4px;word-break:break-word}}
.ct{{font-size:13px;color:var(--muted)}}
td.verdict{{min-width:190px}}td.verdict label{{display:inline-block;margin-right:8px;white-space:nowrap}}
textarea{{display:block;width:100%;min-height:44px;margin-top:6px;font:inherit;font-size:13px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}}
.bar{{display:flex;gap:12px;align-items:center}}
</style></head><body>
<button id="themeBtn" class="ghost theme">切换深浅色</button>
<main>
<h1>Usher 第一阶段：正确答案人工核对表</h1>
<p class="sub">每项判断从测试集里随机抽 30 条（随机种子 {SEED}）。</p>
<div class="card">
<p><b>2026-10-07 修订说明：</b>按修订后的实验说明 A9，类型和查重两节改由 AI 模型（Claude Opus 5.5）根据 issue 的公开内容逐条核对，结果保存在 <code>review/ai_review_type.csv</code> 和 <code>review/ai_review_duplicate.csv</code>，未经人工核实；负责人一节已不再使用，改为自动检查最终负责人是否实际处理过 issue。本页保留作参考，仍可用于人工复核。</p>
</div>
<div class="card">
<p>请逐条判断右侧列出的“正确答案”是否确实正确，并在“判断”一栏选择 <b>对</b>、<b>错</b> 或 <b>无法判断</b>，需要说明时可以写备注。点 issue 编号会在 GitHub 上打开原 issue，方便查看完整内容和讨论。</p>
<p>类型一项的正确答案是维护者最后留下的那个类型标签；负责人一项的正确答案是 issue 当前的全部负责人；查重一项的正确答案是 GitHub 记录的“这条 issue 是哪条更早 issue 的重复”（duplicateOf 字段），下面同时列出那条原 issue 的标题。</p>
<p>你的输入会自动保存在这个浏览器里（localStorage），关掉页面再打开也还在；但换浏览器或清除网站数据后会丢失，所以核对完请用每一节的“导出 CSV”按钮保存结果。</p>
</div>
{''.join(sections)}
</main>
<script>
(function(){{
var KEY='usher-review-v1',state={{}};
try{{state=JSON.parse(localStorage.getItem(KEY)||'{{}}')||{{}};}}catch(e){{state={{}};}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(state));}}catch(e){{}}}}
function prog(sec){{var rows=sec.querySelectorAll('tr[data-key]'),d=0;
 rows.forEach(function(r){{var s=state[r.dataset.key];if(s&&s.v)d++;}});
 sec.querySelector('.prog').textContent='已判断 '+d+' / '+rows.length;}}
document.querySelectorAll('section[data-judgment]').forEach(function(sec){{
 sec.querySelectorAll('tr[data-key]').forEach(function(r){{
  var k=r.dataset.key,s=state[k]||{{}},ta=r.querySelector('textarea');
  r.querySelectorAll('input[type=radio]').forEach(function(i){{
   if(s.v===i.value)i.checked=true;
   i.addEventListener('change',function(){{state[k]=state[k]||{{}};state[k].v=i.value;save();prog(sec);}});}});
  ta.value=s.n||'';
  ta.addEventListener('input',function(){{state[k]=state[k]||{{}};state[k].n=ta.value;save();}});}});
 prog(sec);
 sec.querySelector('.exp').addEventListener('click',function(){{
  var q=function(x){{x=String(x==null?'':x);return '"'+x.replace(/"/g,'""')+'"';}};
  var out=['judgment,number,truth,verdict,note'];
  sec.querySelectorAll('tr[data-key]').forEach(function(r){{var s=state[r.dataset.key]||{{}};
   var t=JSON.parse(r.dataset.truth);if(Array.isArray(t))t=t.join(';');
   out.push([sec.dataset.judgment,r.dataset.num,t,s.v||'',s.n||''].map(q).join(','));}});
  var blob=new Blob(['\\ufeff'+out.join('\\n')+'\\n'],{{type:'text/csv;charset=utf-8'}});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='review_'+sec.dataset.judgment+'.csv';document.body.appendChild(a);a.click();
  setTimeout(function(){{URL.revokeObjectURL(a.href);a.remove();}},0);}});
}});
}})();
{THEME_JS}
</script></body></html>
"""
    (REVIEW / "review_sheet.html").write_text(page)
    print("review sheet:", {j: len(sample[j]) for j in JUDGMENTS})


if __name__ == "__main__":
    main()
