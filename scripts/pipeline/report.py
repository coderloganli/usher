"""A9: write review/data_report.html (Chinese) from the result files. No number is typed by hand.

Inputs: manifest/counts.json, manifest/snapshot.json, manifest/module_labels.json,
        data/timeline_meta.json, review/review_sample.json,
        review/ai_review_{type,module,duplicate}.csv, review/ai_review_summary.json,
        manifest/assignee_check.json
Usage : .venv/bin/python scripts/pipeline/report.py
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime

from common import DATA, DISCONTINUED, JUDGMENTS, MANIFEST, REVIEW, SEED, SPLITS, TYPE_LABELS
from page_style import CSS, THEME_JS

JN = {"type": "类型", "module": "模块", "duplicate": "查重", "assignee": "负责人"}
REVIEWED = ["type", "module", "duplicate"]


def jname(j: str) -> str:
    """Judgment name; the discontinued assignee line is marked wherever it is listed."""
    return f"{JN[j]}（{DISCONTINUED[j]['date']} 起停用，仅作记录）" if j in DISCONTINUED else JN[j]
SN = {"train": "训练集", "tune": "调参集", "test": "测试集"}
SPLIT_DESC = {"train": "2025-01-01 至 2026-05-31", "tune": "2026-06-01 至 2026-06-30",
              "test": "2026-07-01 至 2026-09-30"}
RULE_NAMES = {
    "bot_author": "提交者是 bot",
    "empty_body": "正文为空",
    "no_type_label": "没有任何类型标签",
    "multiple_type_labels": "带两个及以上类型标签",
    "no_assignee": "没有负责人",
    "no_module_label": "没有任何模块标签（classifier.json 中列出的标签）",
    "no_duplicate_of": "没有被标为重复（duplicateOf 为空）",
    "canonical_other_repo": "原 issue 不在 microsoft/vscode 仓库",
    "canonical_not_earlier": "原 issue 不比这条 issue 更早创建",
    "canonical_before_2023": "原 issue 创建于 2023-01-01 之前",
}


def f(n: int) -> str:
    return f"{n:,}"


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "—"


def dur(a: str | None, b: str | None) -> str:
    if not a or not b:
        return "—"
    s = (datetime.fromisoformat(b.replace("Z", "+00:00")) - datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds()
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h} 小时 {m} 分钟" if h else f"{m} 分钟"


def link(n: int) -> str:
    return f'<a href="https://github.com/microsoft/vscode/issues/{n}">#{n}</a>'


def pct1(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def ai_review_section(rs: dict) -> str:
    summ = json.load(open(REVIEW / "ai_review_summary.json"))
    std = summ["a4_standard"]
    rows, notes, verdicts = [], [], []
    for j in REVIEWED:
        v = summ["judgments"][j]
        c = v["counts"]
        if v["meets_a4_point_estimate"] and v["ci95_low_meets_a4"]:
            verdict = f"达标：正确率和 95% 置信区间的下限都不低于 {pct1(std)}。"
        elif v["meets_a4_point_estimate"]:
            verdict = (f"按正确率本身达标（不低于 {pct1(std)}），但 95% 置信区间的下限是 {pct1(v['ci95_low'])}，"
                       f"低于 {pct1(std)}，也就是说，只凭这 {c['对'] + c['错']} 条样本还不能排除真实正确率略低于标准的可能。")
        else:
            verdict = f"未达标：正确率低于 {pct1(std)}。"
        verdicts.append(f"<li><b>{JN[j]}：</b>{verdict}</li>")
        rows.append(
            f"<tr><td>{JN[j]}</td><td class=\"num\">{v['reviewed']}</td><td class=\"num\">{c['对']}</td>"
            f"<td class=\"num\">{c['错']}</td><td class=\"num\">{c['无法判断']}</td>"
            f"<td class=\"num\">{pct1(v['cannot_judge_share'])}</td><td class=\"num\"><b>{pct1(v['accuracy'])}</b></td>"
            f"<td class=\"num\">{pct1(v['ci95_low'])} – {pct1(v['ci95_high'])}</td></tr>")
        with open(REVIEW / f"ai_review_{j}.csv", newline="") as fh:
            items = [r for r in csv.DictReader(fh) if r["verdict"] != "对"]
        for r in items:  # noqa: B007
            truth = (f"重复于 {link(int(r['truth']))}" if j == "duplicate"
                     else "、".join(f"<code>{html.escape(t)}</code>" for t in r["truth"].split(";")))
            notes.append(f"<tr><td>{JN[j]}</td><td>{link(int(r['number']))}</td><td>{truth}</td>"
                         f"<td>{html.escape(r['verdict'])}</td><td>{html.escape(r['reason'])}</td></tr>")
    mm = summ["judgments"]["module"].get("multi_module_items", {"count": 0, "counts": {}})
    with open(REVIEW / "ai_review_module.csv", newline="") as fh:
        mod_rows = "".join(
            f"<tr><td>{link(int(r['number']))}</td><td>{'、'.join(f'<code>{html.escape(t)}</code>' for t in r['truth'].split(';'))}</td>"
            f"<td>{html.escape(r['verdict'])}</td><td>{html.escape(r['reason'])}</td></tr>" for r in csv.DictReader(fh))
    return f"""
<h2>8. 类型、模块和查重：正确答案核对</h2>
<div class="card"><p><b>本节的核对由 AI 模型（Claude Opus 5.5）完成，未经人工核实。</b></p></div>
<p><b>为什么改由 AI 核对。</b>实验说明 A9 在 2026-10-07 修订：类型、模块和查重三项的“正确答案”是否正确，只需要根据 issue 的公开内容就能判断，人和模型掌握的信息相同，所以改由 AI 模型逐条核对，并在结果中注明。原先的人工核对表保留在 <a href="review_sheet.html">review_sheet.html</a>（已加入模块一节），可以随时用来做人工复核。</p>
<p><b>样本。</b>每项从测试集中随机抽 {rs['n_per_judgment']} 条（随机种子 {rs['seed']}），编号保存在 <code>review/review_sample.json</code>，抽样在核对之前就已固定。每项判断各用一个以同一种子初始化的随机数生成器抽样，所以加入模块一项后，类型和查重的样本没有变化。</p>
<p><b>核对方法。</b>对每条样本，模型阅读快照中的标题和正文；查重一项同时阅读原 issue 的标题和正文。正文不足以判断时再参考 GitHub 上的评论。为了做法一致，全部样本以及查重样本对应的原 issue 的评论都通过 GitHub 接口统一拉取了一次，缓存在 <code>data/review_comments/</code>。这些评论是在快照之后拉取的，不属于快照，也不计入快照的校验值。判断标准如下，对每一条都按同样的标准执行：</p>
<ul>
<li><b>类型</b>：如果维护者最后打上的类型标签符合 issue 的内容，判为“对”，例如真实的缺陷对应 <code>bug</code>，请求新的行为对应 <code>feature-request</code>；如果内容明显属于另一种类型，判为“错”；如果内容太少，或者确实可以归入不止一种类型，判为“无法判断”。</li>
<li><b>模块</b>：如果 issue 当前的模块标签所代表的代码区域确实与 issue 描述的问题或需求有关，判为“对”；带多个模块标签时，只有每个标签都与 issue 有关才判为“对”；如果某个标签明显与 issue 无关，判为“错”；如果看不清楚，判为“无法判断”。标签“有关”但不是最具体的选择，或者 issue 的主体落在 classifier.json 没有列出的区域（例如聊天、智能体）时，仍判为“对”，并在理由中写明。</li>
<li><b>查重</b>：如果这条 issue 和它被标为重复的原 issue 描述的是同一个根本问题或同一个需求，判为“对”；如果明显是不同的问题，判为“错”；如果看不清楚，判为“无法判断”。</li>
</ul>
<p>每一条都写了一句理由，完整结果在 <code>review/ai_review_type.csv</code>、<code>review/ai_review_module.csv</code> 和 <code>review/ai_review_duplicate.csv</code>，汇总数字在 <code>review/ai_review_summary.json</code>。模块样本中带多个模块标签的有 {mm['count']} 条，其中判为“对”的 {mm['counts'].get('对', 0)} 条。</p>
<p><b>正确率的算法。</b>正确率 = “对”的条数 ÷（“对”的条数 + “错”的条数），“无法判断”的条目不计入分母，但单独报告它占全部样本的比例。95% 置信区间用 Wilson 方法计算。置信区间表示：考虑到只抽了 30 条，真实的正确率大致落在这个范围内。Wilson 方法在样本少、正确率接近 100% 时仍然给出合理的区间，而常见的“正确率 ± 1.96 × 标准误”在这种情况下会给出超过 100% 或宽度为零的区间。</p>
<div class="scroll"><table><thead><tr><th>判断</th><th class="num">核对条数</th><th class="num">对</th><th class="num">错</th><th class="num">无法判断</th><th class="num">无法判断占比</th><th class="num">正确率</th><th class="num">95% 置信区间</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><b>与 A4 质量标准（正确答案正确的比例不低于 {pct1(std)}）的对比：</b></p>
<ul>{''.join(verdicts)}</ul>
<p><b>判为“错”或“无法判断”的条目。</b>下表列出全部这类条目和理由，便于检查错误集中在哪类情况。</p>
<div class="scroll"><table><thead><tr><th>判断</th><th>issue</th><th>正确答案</th><th>核对结果</th><th>理由</th></tr></thead><tbody>{''.join(notes) or '<tr><td colspan="5">无</td></tr>'}</tbody></table></div>
<p><b>模块一项的全部核对结果。</b>模块是新加入的判断，下表列出全部 {rs['n_per_judgment']} 条样本和理由，其中也能看出哪些 issue 的主体落在 classifier.json 没有列出的区域。</p>
<div class="scroll"><table><thead><tr><th>issue</th><th>模块标签</th><th>核对结果</th><th>理由</th></tr></thead><tbody>{mod_rows}</tbody></table></div>
"""


def module_section(c: dict) -> str:
    ml = json.load(open(MANIFEST / "module_labels.json"))
    diag = c["diagnostics"]
    multi = diag["module_multi_label"]
    k = c["kept"]["module"]
    multi_rows = "".join(
        f"<tr><td>{SN[s]}</td><td class=\"num\">{f(v['issues'])}</td><td class=\"num\">{f(v['multi_module'])}</td>"
        f"<td class=\"num\">{pct(v['multi_module'], v['issues'])}</td>"
        f"<td>{'；'.join(f'{n} 个：{f(m)} 条' for n, m in v['modules_per_issue'].items())}</td></tr>"
        for s, v in multi.items())
    dist_te = diag["module_label_distribution_test"]
    dist_tr = diag["module_label_distribution_train"]
    top = list(dist_te.items())[:20]
    top_rows = "".join(f"<tr><td><code>{html.escape(m)}</code></td><td class=\"num\">{f(dist_tr.get(m, 0))}</td>"
                       f"<td class=\"num\">{f(n)}</td></tr>" for m, n in top)
    unseen = diag["module_test_labels_not_in_train"]
    unseen_txt = ("、".join(f"<code>{html.escape(m)}</code>" for m in unseen) if unseen else "无")
    dropped = diag["module_dropped_test_issues"]
    nolab = diag["module_dropped_test_without_any_label"]
    dl = diag["module_dropped_test_top_labels"]
    dl_txt = "、".join(f"<code>{html.escape(l)}</code>（{f(n)}）" for l, n in list(dl.items())[:20])
    drop_names = ", ".join(ml["dropped_never_used"])
    before_test = c["issues_per_split_before_filter"]["test"]
    return f"""
<h2>7. 模块标签（2026-10-07 新增）</h2>
<p><b>模块判断是什么。</b>模块判断回答“这条 issue 属于哪个代码模块”，例如终端（<code>terminal</code>）、编辑器标签页（<code>workbench-tabs</code>）。它在 2026-10-07 的修订中取代了负责人判断。正确答案是 issue 当前带有的全部模块标签；一条 issue 带多个模块标签时，这些标签都算正确答案，模型猜中其中任何一个都算命中（实验说明 B2 用 top-1 和 top-3 命中率打分）。</p>
<p><b>哪些标签算模块标签。</b>vscode 仓库的 <code>.github/classifier.json</code> 在 <code>labels</code> 下列出了 {f(ml['classifier_keys'])} 个标签，它们是分诊机器人按模块自动分派时使用的标签。快照里没有仓库的完整标签列表，所以我们用快照本身检查这些标签是否真的在使用：只有在拉取到的 {f(c['issues_pulled'])} 条 issue（2023-01 至 2026-09）中至少出现过一次的标签才算模块标签。按这条规则保留 {f(ml['module_labels_count'])} 个，去掉 {f(ml['dropped_never_used_count'])} 个从未出现过的标签：{html.escape(drop_names)}。从未出现过的标签不可能成为正确答案，去掉它们只会缩小基线模型的候选范围。完整列表在 <code>manifest/module_labels.json</code>。</p>
<p><b>筛选。</b>先执行各项判断共用的两条规则（去掉 bot 提交的 issue、去掉正文为空的 issue），再去掉没有任何模块标签的 issue。筛选后训练集 {f(k['train'])} 条、调参集 {f(k['tune'])} 条、测试集 {f(k['test'])} 条，各步去掉的数量见第 3 节。</p>
<p><b>带多个模块标签的 issue。</b>实验说明 B2 要求单独报告这类 issue 的占比。下表列出每个集合中带两个及以上模块标签的 issue 数和占比，以及“每条 issue 带几个模块标签”的分布。</p>
<div class="scroll"><table><thead><tr><th>集合</th><th class="num">issue 数</th><th class="num">带多个模块标签</th><th class="num">占比</th><th>每条 issue 的模块标签数</th></tr></thead><tbody>{multi_rows}</tbody></table></div>
<p><b>最常见的模块。</b>测试集中一共出现了 {f(len(dist_te))} 个不同的模块标签，训练集中出现了 {f(len(dist_tr))} 个。测试集里出现、但训练集里从未出现的模块标签是：{unseen_txt}，涉及 {f(diag['module_test_issues_with_label_not_in_train'])} 条测试 issue。下表是测试集中最常见的 20 个模块标签（一条 issue 有多个标签时每个标签各计一次）。</p>
<div class="scroll"><table><thead><tr><th>模块标签</th><th class="num">训练集</th><th class="num">测试集</th></tr></thead><tbody>{top_rows}</tbody></table></div>
<p><b>已知局限：classifier.json 已经过时。</b>截至查询时，classifier.json 最近一次修改是 {ml['source']['commit_date'][:10]}。它没有收录 2025–2026 年新增的大部分模块，尤其是聊天和智能体相关的区域；其中只有 <code>chat-terminal</code> 等个别与聊天有关的标签。因此模块测试集只覆盖 classifier.json 列出的模块，不代表全部 issue：测试集时间段内共有 {f(before_test)} 条 issue，按共用规则筛选后仍有 {f(dropped)} 条因为没有模块标签被去掉，只保留了 {f(k['test'])} 条。被去掉的 {f(dropped)} 条中，{f(nolab)} 条没有任何标签；其余 issue 上最常见的标签是（括号内为条数）：{dl_txt}。其中 <code>agents-window</code>、<code>agent-host</code>、<code>chat</code> 等是近年大量使用、但 classifier.json 中没有的模块类标签；<code>bug</code>、<code>insiders-released</code> 等则是类型或流程标签，不表示模块。基于这个测试集得到的分数，只能说明模型在这些“老模块”上的表现，不能推广到聊天、智能体等新区域。</p>
"""


def assignee_section(tl: dict) -> str:
    a = json.load(open(MANIFEST / "assignee_check.json"))
    o = a["overall"]
    cats = a["categories"]
    subsets = a["by_bot_subset"]

    def cell(v: dict, k: str) -> str:
        return f'<td class="num">{f(v["counts"][k])}（{pct(v["counts"][k], v["issues"])}）</td>'

    def_rows = "".join(f"<tr><td>{c['name_zh']}</td><td>{c['definition_zh']}</td>{cell(o, c['key'])}</tr>" for c in cats)
    sub_head = "".join(f'<th class="num">{v["name_zh"]}</th>' for v in subsets.values())
    sub_rows = "".join(
        f"<tr><td>{c['name_zh']}</td>{cell(o, c['key'])}" + "".join(cell(v, c["key"]) for v in subsets.values()) + "</tr>"
        for c in cats)
    sub_rows += (f"<tr><td><b>合计</b></td><td class=\"num\"><b>{f(o['issues'])}</b></td>"
                 + "".join(f'<td class="num"><b>{f(v["issues"])}</b></td>' for v in subsets.values()) + "</tr>")
    month_rows = "".join(
        f"<tr><td>{m}</td><td class=\"num\">{f(v['issues'])}</td><td class=\"num\">{f(v['counts']['none'])}（{pct(v['counts']['none'], v['issues'])}）</td></tr>"
        for m, v in a["by_month"].items())
    none = o["counts"]["none"]
    s_in, s_out, s_no = subsets["bot_in_final"], subsets["bot_not_in_final"], subsets["no_bot_assignment"]
    share = {k: v["counts"]["none"] / v["issues"] for k, v in subsets.items() if v["issues"]}
    if len(share) == 3 and share["bot_in_final"] > max(share["bot_not_in_final"], share["no_bot_assignment"]):
        subset_note = ("也就是说，最终负责人就是 bot 首次分派的人时，没有任何操作的比例最高；"
                       "负责人经过人工改派或由人工直接指定时，这个比例更低。")
    else:
        subset_note = ""
    return f"""
<h2>9. 负责人（已停用）：最终决定是否落实</h2>
<div class="card"><p><b>负责人判断已于 {DISCONTINUED['assignee']['date']} 停止评估，由模块判断取代。</b>本节的检查是停用前完成的，结果保留作为记录，也是停用的依据之一：“该分给谁”取决于团队内部分工，无法用公开信息评估，而且下面的数字显示，很多最终负责人从未在 issue 上做过任何操作。之后 agent 给出的建议负责人由我们假定的“模块 → 负责人”对应表给出，只作演示，不报告准确率。</p></div>
<p><b>为什么不做主观核对。</b>一条 issue 该分给谁，取决于 vscode 团队内部的分工，从公开信息无法判断对错。因此按 2026-10-07 修订的实验说明 A9，负责人的正确答案就定义为维护者最后的决定，不再核对它“对不对”，而是检查这个决定是否落实：最终负责人有没有在这条 issue 上实际做过操作。这一项不适用 A4 中“正确率不低于 85%”的标准，只报告落实的比例。</p>
<p><b>怎么检查。</b>脚本 <code>scripts/pipeline/assignee_check.py</code> 读取负责人测试集全部 {f(o['issues'])} 条 issue 已缓存的时间线（时间线是 GitHub 对一条 issue 上每一次操作的记录，包括评论、关闭、加标签、分派等，每条记录都写明是谁做的），找出由任何一位最终负责人执行的操作。以下几类记录虽然也写着这个人的名字，但不是他本人的操作，所以不计入：被别人分派或取消分派（assigned/unassigned，执行者是别人）；被别人在评论里提到（mentioned）；被 GitHub 自动加入或移出关注（subscribed/unsubscribed）。打开 issue 本身不是时间线里的记录，所以提交 issue 也不算一次操作。</p>
<p>每条 issue 只归入一类：按下表从上到下的顺序，取任何一位最终负责人做过的最“强”的那种操作。例如一条 issue 的负责人既评论过、又关闭了它，就归入“关闭了 issue”。时间线缺失的 issue 有 {f(a['timelines_missing'])} 条。逐条结果保存在 <code>manifest/assignee_check.tsv</code>，汇总在 <code>manifest/assignee_check.json</code>。</p>
<div class="scroll"><table><thead><tr><th>类别</th><th>定义</th><th class="num">issue 数（占比）</th></tr></thead><tbody>{def_rows}</tbody></table></div>
<p><b>结果。</b>在 {f(o['issues'])} 条负责人测试 issue 中，有 {f(o['any_action'])} 条（{pct(o['any_action'], o['issues'])}）至少有一位最终负责人做过操作；如果不把“只做了自我分派”算作实际处理，这个数字是 {f(o['substantive_action'])} 条（{pct(o['substantive_action'], o['issues'])}）。没有任何最终负责人操作的有 <b>{f(none)} 条（{pct(none, o['issues'])}）</b>。在这 {f(none)} 条中，拉取时间线时仍然打开的有 {f(o['none_still_open'])} 条，被分诊 bot 关闭的有 {f(o['none_closed_by_bot'])} 条（例如信息不足或被标为重复后自动关闭），被负责人以外的其他人关闭的有 {f(o['none_closed_by_other_person'])} 条。</p>
<p>另外，有 {f(o['authored_by_final_assignee'])} 条 issue 是由它的某位最终负责人自己提交的；其中只有 {f(o['authored_by_final_assignee_among_none'])} 条落在“没有任何操作”一类。</p>
<p><b>按 bot 的首次分派分组。</b>第 5 节介绍了分诊 bot 的首次分派。下表把负责人测试集分成三组：bot 首次分派的人在最终负责人中（{f(s_in['issues'])} 条），bot 首次分派的人不在最终负责人中、也就是后来被人改派了（{f(s_out['issues'])} 条），以及 bot 没有分派过、负责人是人工指定的（{f(s_no['issues'])} 条）。</p>
<div class="scroll"><table><thead><tr><th>类别</th><th class="num">全部</th>{sub_head}</tr></thead><tbody>{sub_rows}</tbody></table></div>
<p>三组中“没有任何操作”的比例分别是 {pct(s_in['counts']['none'], s_in['issues'])}、{pct(s_out['counts']['none'], s_out['issues'])} 和 {pct(s_no['counts']['none'], s_no['issues'])}。{subset_note}</p>
<p><b>按创建月份。</b>时间线在 {tl['first_fetched_at']} 到 {tl['last_fetched_at']}（UTC）之间拉取，9 月创建的 issue 到拉取时最多只有一个月，负责人还没来得及处理的可能性更大。下表按创建月份列出“没有任何操作”的比例。</p>
<div class="scroll"><table><thead><tr><th>创建月份</th><th class="num">issue 数</th><th class="num">没有任何操作（占比）</th></tr></thead><tbody>{month_rows}</tbody></table></div>
<p><b>怎么理解这些数字。</b>“没有任何操作”不等于分错了人：负责人可能只是还没处理、可能在 GitHub 以外处理（例如在其他仓库提交修复而没有引用这条 issue），也可能认为 issue 不需要处理而由 bot 自动关闭。反过来，“有操作”也不能证明分派是对的。这项检查只说明最终决定在 GitHub 上有没有留下被执行的痕迹；后续用负责人作为正确答案打分时，应当把这个比例一并报告。</p>
<p>需要特别注意的是第一组中没有任何操作的 {f(s_in['counts']['none'])} 条（占负责人测试集的 {pct(s_in['counts']['none'], o['issues'])}）：它们的正确答案就是 bot 当初的分派，之后没有任何负责人在 GitHub 上处理过，因此这个答案并没有经过人的确认。基线阶段拿 bot 的首次分派做对照时，bot 在这些 issue 上必然“答对”，所以除了全部测试集的分数，还应报告去掉这部分 issue 之后的分数。</p>
"""


def a4_section() -> str:
    summ = json.load(open(REVIEW / "ai_review_summary.json"))
    a = json.load(open(MANIFEST / "assignee_check.json"))["overall"]
    std = summ["a4_standard"]
    rows = []
    for j in REVIEWED:
        v = summ["judgments"][j]
        ok = "达标" if v["meets_a4_point_estimate"] else "未达标"
        low = "下限也不低于标准" if v["ci95_low_meets_a4"] else "下限低于标准"
        rows.append(f"<tr><td>{JN[j]}的正确答案正确的比例不低于 {pct1(std)}（AI 核对）</td>"
                    f"<td>{pct1(v['accuracy'])}，95% 置信区间 {pct1(v['ci95_low'])} – {pct1(v['ci95_high'])}（{low}）</td><td>{ok}</td></tr>")
    rows.append(f"<tr><td>负责人（已停用，仅作记录）</td><td>不适用正确率标准。至少有一位最终负责人做过操作的比例是 "
                f"{pct(a['any_action'], a['issues'])}，没有任何操作的比例是 {pct(a['counts']['none'], a['issues'])}。</td><td>只报告</td></tr>")
    rows.append("<tr><td>不包含 bot 自动提交的 issue</td><td>由筛选规则保证，见第 3 节。</td><td>达标</td></tr>")
    rows.append("<tr><td>每条 issue 都有正文和创建时间</td><td>正文为空的 issue 已在筛选中去掉，创建时间由 GitHub 记录，每条都有。</td><td>达标</td></tr>")
    return f"""
<h2>10. 与 A4 质量标准的对比</h2>
<p>实验说明 A4 规定，数据必须同时满足下表中的条件才能用来出结果。类型、模块和查重三项的核对由 AI 模型（Claude Opus 5.5）完成，未经人工核实。</p>
<div class="scroll"><table><thead><tr><th>条件</th><th>结果</th><th>结论</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>这里“达标”按正确率本身是否不低于 {pct1(std)} 判断。由于每项只核对了 30 条，置信区间较宽，表中同时注明了区间下限是否也不低于标准。</p>
"""


def main() -> None:
    c = json.load(open(MANIFEST / "counts.json"))
    snap = json.load(open(MANIFEST / "snapshot.json"))
    tl = json.load(open(DATA / "timeline_meta.json"))
    rs = json.load(open(REVIEW / "review_sample.json"))
    ip, tp = snap["issue_pull"], snap["timeline_pull"]
    diag = c["diagnostics"]
    ml = json.load(open(MANIFEST / "module_labels.json"))

    kept_rows = "".join(
        f"<tr><td>{jname(j)}</td>" + "".join(f'<td class="num">{f(c["kept"][j][s])}</td>' for s in SPLITS)
        + f'<td class="num">{f(sum(c["kept"][j].values()))}</td></tr>' for j in JUDGMENTS)

    removal_tables = []
    for j in JUDGMENTS:
        rows = [f"<tr><td>筛选前（该时间段内拉到的全部 issue）</td>" + "".join(
            f'<td class="num">{f(c["issues_per_split_before_filter"][s])}</td>' for s in SPLITS) + "</tr>"]
        for i, rule in enumerate(c["rule_order"][j], 1):
            rows.append(f"<tr><td>第 {i} 步去掉：{RULE_NAMES[rule]}</td>" + "".join(
                f'<td class="num">{f(c["removed"][j][s][rule])}</td>' for s in SPLITS) + "</tr>")
        rows.append(f"<tr><td><b>保留</b></td>" + "".join(
            f'<td class="num"><b>{f(c["kept"][j][s])}</b></td>' for s in SPLITS) + "</tr>")
        removal_tables.append(
            f"<h3>{jname(j)}</h3><div class=\"scroll\"><table><thead><tr><th>筛选步骤</th>"
            + "".join(f'<th class="num">{SN[s]}</th>' for s in SPLITS) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")

    flag_rows = []
    for j in JUDGMENTS:
        n, tot = c["exact_dup_flagged_test"][j], c["kept"][j]["test"]
        ex = c["exact_dup_flags"][j][:10]
        exs = "、".join(f"{link(x['number'])}（与训练集 {link(x['train_match'])} 相同）" for x in ex) or "无"
        more = f"，另有 {f(n - len(ex))} 条未列出" if n > len(ex) else ""
        flag_rows.append(f"<tr><td>{jname(j)}</td><td class=\"num\">{f(n)}</td><td class=\"num\">{f(tot)}</td>"
                         f"<td class=\"num\">{pct(n, tot)}</td><td>{exs}{more}</td></tr>")

    type_dist = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td class=\"num\">{f(diag['type_label_distribution_train'].get(k, 0))}</td>"
        f"<td class=\"num\">{f(diag['type_label_distribution_test'].get(k, 0))}</td></tr>" for k in TYPE_LABELS)
    trunc = ip["truncated_connections"]
    module_html = module_section(c)

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Usher 数据报告</title><style>{CSS}</style></head><body>
<button id="themeBtn" class="ghost theme">切换深浅色</button>
<main>
<h1>Usher 第一阶段数据报告（A5–A9）</h1>
<p class="sub">本页由 <code>scripts/pipeline/report.py</code> 根据结果文件自动生成，页面里的数字都来自脚本输出，没有手工填写。</p>

<div class="card">
<p>这份报告对应实验说明（stage1_spec.html）里 A5 到 A9 的数据部分：从 GitHub 拉取 microsoft/vscode 的 issue，固定成一个快照，按规则筛选，按创建时间切成训练集、调参集和测试集，并评估正确答案的质量。2026-10-07 的修订把三项判断定为类型、模块和查重：类型、模块和查重的正确答案由 AI 模型核对；原来的负责人判断停止评估，由模块判断取代，已完成的负责人数据和检查保留作为记录，在本页中都标为“已停用”。基线模型（B 部分）不在本页范围内，会在下一步完成。</p>
<p>报告中的“issue”指 GitHub 上的问题报告；“正确答案”指我们从维护者的操作中提取、用来给模型打分的标准答案；“bot”指自动操作的账号，例如 vscode 团队自己的分诊机器人 <code>vs-code-engineering[bot]</code>；“模块”指 issue 涉及的代码区域，例如终端、编辑器标签页，在 vscode 仓库里用标签表示。</p>
</div>

<h2>1. 快照信息</h2>
<p>issue 数据通过 GitHub 的 GraphQL 接口一次性批量拉取，每页 100 条，按创建时间从新到旧翻页。拉取从 <b>{ip['started_at']}</b>（UTC）开始，到 <b>{ip['finished_at']}</b> 结束，用时 {dur(ip['started_at'], ip['finished_at'])}，共请求 {f(ip['graphql_pages'])} 页。创建时间在 {ip['range_utc'][0][:10]} 到 2026-09-30 之间的 issue 共有 <b>{f(ip['issues'])}</b> 条。每条 issue 记录了编号、创建时间、标题、正文、提交者、最终标签、最终负责人，以及 GitHub 的 duplicateOf 字段（这条 issue 被维护者标为哪条 issue 的重复，以及那条原 issue 的创建时间和所在仓库）。</p>
<p>标签每条最多取 {ip['limits']['labels']} 个，负责人每条最多取 {ip['limits']['assignees']} 位。实际超出上限、因而可能被截断的 issue 数量是：标签 {f(trunc['labels'])} 条，负责人 {f(trunc['assignees'])} 条。</p>
<p>测试集的操作记录（时间线）通过 REST 接口逐条拉取，第一条在 {tp['first_fetched_at']} 拉取，最后一条在 {tp['last_fetched_at']} 拉取，前后历时 {dur(tp['first_fetched_at'], tp['last_fetched_at'])}，共 {f(tp['rest_pages'] or 0)} 页请求（时间线超过 100 条事件的 issue 需要翻页）。时间线和 issue 不是在同一时刻拉取的，两者之间 issue 本身可能又有变化，但我们只从时间线里读 bot 的第一次分派，这一事件在之后不会改变。</p>
<p>模块标签列表来自 vscode 仓库的 <code>.github/classifier.json</code>（vscode 分诊机器人用来按模块自动分派 issue 的配置文件）。这个文件于 {ml['source']['fetched_on']} 从 vscode 默认分支取得，保存为快照中的 <code>{ml['source']['snapshot_file']}</code>，并和其他数据文件一样计算了校验值。它的 git 对象编号与 vscode 提交 <code>{ml['source']['commit'][:12]}</code>（{ml['source']['commit_date'][:10]}）中的版本一致，这个提交也是截至查询时最后一次修改该文件的提交。详细来源记录在 <code>manifest/module_labels.json</code>。</p>
<p>快照的存档放在 <code>manifest/</code> 目录：<code>snapshot.json</code> 记录拉取时间和查询语句；<code>issue_numbers.txt</code> 是全部 {f(ip['issues'])} 个 issue 编号；<code>issue_hashes.tsv</code> 给每条 issue 的内容（标题、正文、标签、负责人、重复关系等）算了一个校验值，别人重新拉取后可以逐条比对哪些 issue 和我们的不同；<code>checksums.sha256</code> 记录了 <code>data/</code> 下全部 {f(snap['files_checksummed'])} 个数据文件的 SHA-256 校验值（一种文件“指纹”，内容只要改动一个字节，校验值就会完全不同）。存档时间是 {snap['archived_at']}。</p>

<h2>2. 筛选后每项判断的数量</h2>
<p>下表是筛选之后，各项判断在训练集（{SPLIT_DESC['train']}）、调参集（{SPLIT_DESC['tune']}）和测试集（{SPLIT_DESC['test']}）中各有多少条 issue。切分按 issue 的创建时间（UTC）进行。每个集合的编号和正确答案保存在 <code>manifest/sets/&lt;判断&gt;_&lt;集合&gt;.tsv</code>。</p>
<div class="scroll"><table><thead><tr><th>判断</th>{''.join(f'<th class="num">{SN[s]}</th>' for s in SPLITS)}<th class="num">合计</th></tr></thead><tbody>{kept_rows}</tbody></table></div>
<p>查重这一项的检索范围是拉取到的全部 {f(diag['duplicate_corpus_size'])} 条 issue（2023-01-01 起），每条查询只能在比它更早创建的 issue 里找。这个检索范围包括 bot 提交的 issue，因为原 issue 本身可能是 bot 提交的，而真实使用时这些 issue 也存在于仓库中；bot 提交的 issue 只是不作为查询。在全部 {f(sum(c['kept']['duplicate'].values()))} 条查重查询中，原 issue 由 bot 提交的有 {f(diag['duplicate_canonical_bot_authored'])} 条，原 issue 不在拉取结果里的有 {f(diag['duplicate_canonical_missing_from_pull'])} 条。</p>

<h2>3. 每条筛选规则去掉了多少</h2>
<p>筛选规则按表中的顺序依次执行，每条 issue 只计入第一条把它去掉的规则。因此“筛选前”减去各步去掉的数量，正好等于“保留”的数量。2023 年和 2024 年创建的 issue 不属于任何一个集合，只作为查重的检索范围，所以不出现在这些表里。</p>
{''.join(removal_tables)}

<h2>4. 测试集中与训练集文本完全相同的 issue</h2>
<p>按照说明 A7，这类 issue 保留在测试集里，但做了标记，以后报告分数时会给出两个数字：全部测试数据的分数，以及去掉这些 issue 之后的分数。判断方法是：取 issue 的标题加正文，去掉 HTML 注释（多为 issue 模板里的提示文字），正文只取前 2,000 个字符（和 PoC 的 <code>issue_text</code> 一致），再把连续的空白合并成一个空格并统一转成小写；如果测试集某条 issue 处理后的文本和同一项判断的训练集里某条 issue 完全一样，就标记它。</p>
<div class="scroll"><table><thead><tr><th>判断</th><th class="num">被标记的测试 issue</th><th class="num">测试集总数</th><th class="num">占比</th><th>例子（最多列 10 条）</th></tr></thead><tbody>{''.join(flag_rows)}</tbody></table></div>
<p>完整的标记列表在 <code>manifest/counts.json</code> 的 <code>exact_dup_flags</code> 字段，以及各测试集 tsv 文件的 <code>exact_dup_of_train</code> 列。</p>

<h2>5. bot 第一次分派的覆盖情况（负责人，已停用）</h2>
<p>这一节属于已停用的负责人判断，结果保留作为记录。说明 A2 要求找出 vscode 的分诊 bot（<code>vs-code-engineering[bot]</code>）第一次把 issue 分给了谁，留到基线阶段和我们的负责人判断做对照。做法是：拉取负责人测试集每条 issue 的时间线，取第一个由这个 bot 发出的 <code>assigned</code>（分派）事件，记下被分派的人。</p>
<p>负责人测试集共有 {f(tl['issues'])} 条 issue，其中 {f(tl['timelines_available'])} 条成功拉到了时间线，{f(tl['timelines_missing'])} 条拉不到（例如 issue 已被删除或转移）。在全部 {f(tl['issues'])} 条中，有 <b>{f(tl['with_bot_first_assignment'])}</b> 条（{pct(tl['with_bot_first_assignment'], tl['issues'])}）找到了 bot 的第一次分派；其余 issue 的时间线里没有这个 bot 发出的分派事件（通常是维护者直接分派的）。bot 分派得对不对，留到基线阶段再统计。结果保存在 <code>manifest/bot_first_assignment.tsv</code>。</p>

<h2>6. 其他检查</h2>
<p><b>正文为空的 issue。</b>说明 A4 要求每条 issue 都有正文。正文为空（去掉空白后没有任何字符）的 issue 在筛选中直接去掉，去掉的数量见第 3 节各表中“正文为空”一行。</p>
<p><b>bot 判定。</b>提交者满足以下任一条件，就认为是 bot：登录名在 PoC 的 bot 账号名单中，或以 <code>[bot]</code> 结尾；或者 GitHub 标注的账号类型是 Bot。在 2025-01 到 2026-09 的 issue 中，只靠账号类型才识别出来的 bot issue 有 {f(diag['bot_by_typename_only'])} 条。</p>
<p><b>类型标签的分布。</b>下表是筛选后训练集和测试集中各类型标签的数量，供以后看 macro-F1 时参考。</p>
<div class="scroll"><table><thead><tr><th>类型标签</th><th class="num">训练集</th><th class="num">测试集</th></tr></thead><tbody>{type_dist}</tbody></table></div>

{module_html}
{ai_review_section(rs)}
{assignee_section(tl)}
{a4_section()}

<h2>11. 如何重新运行</h2>
<p>在 worktree 根目录执行下面这条命令，就会依次完成拉取、筛选与切分、时间线拉取、负责人落实检查、存档、核对表、AI 核对结果的汇总和本报告。拉取脚本可以中断后续跑：已经拉到的页面和时间线会缓存在 <code>data/raw/</code> 里，不会重复请求。这次快照拉取 issue 用了 {dur(ip['started_at'], ip['finished_at'])}，拉取时间线用了 {dur(tp['first_fetched_at'], tp['last_fetched_at'])}（REST 接口每小时限 5,000 次请求，脚本在额度不足时会自动等待到额度重置）。</p>
<p>因为拉取结果会被缓存，在已有 <code>data/raw/</code> 的情况下再次运行这条命令不会重新拉取，而是在同一个快照上重新计算，结果和本页完全一致（已核对过两次运行的校验值相同）。如果要拉一个新的快照，需要先删掉或移走 <code>data/raw/</code> 目录。</p>
<pre>./run_data.sh</pre>
<p>如果只想在已有数据上重新筛选和生成报告，不再请求 GitHub，可以运行 <code>./run_data.sh --offline</code>。模块标签列表只读取快照中的 <code>data/raw/classifier.json</code>，并核对它和记录的提交版本一致；负责人落实检查只读取已缓存的时间线，AI 核对的汇总只读取 <code>review/</code> 下已保存的核对结果，两者在离线模式下都可以重新运行；AI 核对本身（逐条阅读和判断）不是脚本的一部分，重新运行不会改变核对结果。</p>

<h2>12. 变更记录</h2>
<ul>
<li><b>2026-10-06：</b>筛选规则增加两条：去掉正文为空的 issue；识别 bot 时加上 GitHub 标注的账号类型。数量和核对样本随之更新。</li>
<li><b>2026-10-07（负责人改为模块）：</b>按修订后的实验说明，负责人判断停止评估，改为模块判断。新增模块标签列表（第 7 节）和模块的筛选与切分，模块正确答案的 AI 核对结果并入第 8 节，A4 对比（第 10 节）加入模块一行。负责人的集合、时间线和落实检查保留作为记录，在各表中标为“已停用”。原第 7 至第 11 节依次改为第 8 至第 12 节。</li>
<li><b>2026-10-07：</b>按修订后的实验说明 A9 完成数据质量评估。类型和查重的正确答案改由 AI 模型（Claude Opus 5.5）根据公开内容逐条核对，结果见第 7 节，未经人工核实；负责人不再做主观核对，改为自动检查最终负责人是否在 issue 上实际做过操作，结果见第 8 节；第 9 节把结果和 A4 的质量标准做了对比。人工核对表保留，但其中负责人一节已不再使用。</li>
</ul>
</main><script>{THEME_JS}</script></body></html>
"""
    (REVIEW / "data_report.html").write_text(page)
    print("report written")


if __name__ == "__main__":
    main()
