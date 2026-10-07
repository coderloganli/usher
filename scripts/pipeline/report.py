"""A9: write review/data_report.html (Chinese) from the result files. No number is typed by hand.

Inputs: manifest/counts.json, manifest/snapshot.json, data/timeline_meta.json,
        review/review_sample.json
Usage : .venv/bin/python scripts/pipeline/report.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime

from common import DATA, JUDGMENTS, MANIFEST, REVIEW, SEED, SPLITS, TYPE_LABELS
from page_style import CSS, THEME_JS

JN = {"type": "类型", "assignee": "负责人", "duplicate": "查重"}
SN = {"train": "训练集", "tune": "调参集", "test": "测试集"}
SPLIT_DESC = {"train": "2025-01-01 至 2026-05-31", "tune": "2026-06-01 至 2026-06-30",
              "test": "2026-07-01 至 2026-09-30"}
RULE_NAMES = {
    "bot_author": "提交者是 bot",
    "empty_body": "正文为空",
    "no_type_label": "没有任何类型标签",
    "multiple_type_labels": "带两个及以上类型标签",
    "no_assignee": "没有负责人",
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


def main() -> None:
    c = json.load(open(MANIFEST / "counts.json"))
    snap = json.load(open(MANIFEST / "snapshot.json"))
    tl = json.load(open(DATA / "timeline_meta.json"))
    rs = json.load(open(REVIEW / "review_sample.json"))
    ip, tp = snap["issue_pull"], snap["timeline_pull"]
    diag = c["diagnostics"]

    kept_rows = "".join(
        f"<tr><td>{JN[j]}</td>" + "".join(f'<td class="num">{f(c["kept"][j][s])}</td>' for s in SPLITS)
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
            f"<h3>{JN[j]}</h3><div class=\"scroll\"><table><thead><tr><th>筛选步骤</th>"
            + "".join(f'<th class="num">{SN[s]}</th>' for s in SPLITS) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")

    flag_rows = []
    for j in JUDGMENTS:
        n, tot = c["exact_dup_flagged_test"][j], c["kept"][j]["test"]
        ex = c["exact_dup_flags"][j][:10]
        exs = "、".join(f"{link(x['number'])}（与训练集 {link(x['train_match'])} 相同）" for x in ex) or "无"
        more = f"，另有 {f(n - len(ex))} 条未列出" if n > len(ex) else ""
        flag_rows.append(f"<tr><td>{JN[j]}</td><td class=\"num\">{f(n)}</td><td class=\"num\">{f(tot)}</td>"
                         f"<td class=\"num\">{pct(n, tot)}</td><td>{exs}{more}</td></tr>")

    type_dist = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td class=\"num\">{f(diag['type_label_distribution_train'].get(k, 0))}</td>"
        f"<td class=\"num\">{f(diag['type_label_distribution_test'].get(k, 0))}</td></tr>" for k in TYPE_LABELS)
    trunc = ip["truncated_connections"]

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Usher 数据报告</title><style>{CSS}</style></head><body>
<button id="themeBtn" class="ghost theme">切换深浅色</button>
<main>
<h1>Usher 第一阶段数据报告（A5–A9）</h1>
<p class="sub">本页由 <code>scripts/pipeline/report.py</code> 根据结果文件自动生成，页面里的数字都来自脚本输出，没有手工填写。</p>

<div class="card">
<p>这份报告对应实验说明（stage1_spec.html）里 A5 到 A9 的数据部分：从 GitHub 拉取 microsoft/vscode 的 issue，固定成一个快照，按规则筛选，按创建时间切成训练集、调参集和测试集，并准备人工核对表。基线模型（B 部分）不在本页范围内，会在下一步完成。</p>
<p>报告中的“issue”指 GitHub 上的问题报告；“正确答案”指我们从维护者的操作中提取、用来给模型打分的标准答案；“bot”指自动操作的账号，例如 vscode 团队自己的分诊机器人 <code>vs-code-engineering[bot]</code>。</p>
</div>

<h2>1. 快照信息</h2>
<p>issue 数据通过 GitHub 的 GraphQL 接口一次性批量拉取，每页 100 条，按创建时间从新到旧翻页。拉取从 <b>{ip['started_at']}</b>（UTC）开始，到 <b>{ip['finished_at']}</b> 结束，用时 {dur(ip['started_at'], ip['finished_at'])}，共请求 {f(ip['graphql_pages'])} 页。创建时间在 {ip['range_utc'][0][:10]} 到 2026-09-30 之间的 issue 共有 <b>{f(ip['issues'])}</b> 条。每条 issue 记录了编号、创建时间、标题、正文、提交者、最终标签、最终负责人，以及 GitHub 的 duplicateOf 字段（这条 issue 被维护者标为哪条 issue 的重复，以及那条原 issue 的创建时间和所在仓库）。</p>
<p>标签每条最多取 {ip['limits']['labels']} 个，负责人每条最多取 {ip['limits']['assignees']} 位。实际超出上限、因而可能被截断的 issue 数量是：标签 {f(trunc['labels'])} 条，负责人 {f(trunc['assignees'])} 条。</p>
<p>测试集的操作记录（时间线）通过 REST 接口逐条拉取，第一条在 {tp['first_fetched_at']} 拉取，最后一条在 {tp['last_fetched_at']} 拉取，前后历时 {dur(tp['first_fetched_at'], tp['last_fetched_at'])}，共 {f(tp['rest_pages'] or 0)} 页请求（时间线超过 100 条事件的 issue 需要翻页）。时间线和 issue 不是在同一时刻拉取的，两者之间 issue 本身可能又有变化，但我们只从时间线里读 bot 的第一次分派，这一事件在之后不会改变。</p>
<p>快照的存档放在 <code>manifest/</code> 目录：<code>snapshot.json</code> 记录拉取时间和查询语句；<code>issue_numbers.txt</code> 是全部 {f(ip['issues'])} 个 issue 编号；<code>issue_hashes.tsv</code> 给每条 issue 的内容（标题、正文、标签、负责人、重复关系等）算了一个校验值，别人重新拉取后可以逐条比对哪些 issue 和我们的不同；<code>checksums.sha256</code> 记录了 <code>data/</code> 下全部 {f(snap['files_checksummed'])} 个数据文件的 SHA-256 校验值（一种文件“指纹”，内容只要改动一个字节，校验值就会完全不同）。存档时间是 {snap['archived_at']}。</p>

<h2>2. 筛选后每项判断的数量</h2>
<p>下表是筛选之后，三项判断在训练集（{SPLIT_DESC['train']}）、调参集（{SPLIT_DESC['tune']}）和测试集（{SPLIT_DESC['test']}）中各有多少条 issue。切分按 issue 的创建时间（UTC）进行。每个集合的编号和正确答案保存在 <code>manifest/sets/&lt;判断&gt;_&lt;集合&gt;.tsv</code>。</p>
<div class="scroll"><table><thead><tr><th>判断</th>{''.join(f'<th class="num">{SN[s]}</th>' for s in SPLITS)}<th class="num">合计</th></tr></thead><tbody>{kept_rows}</tbody></table></div>
<p>查重这一项的检索范围是拉取到的全部 {f(diag['duplicate_corpus_size'])} 条 issue（2023-01-01 起），每条查询只能在比它更早创建的 issue 里找。这个检索范围包括 bot 提交的 issue，因为原 issue 本身可能是 bot 提交的，而真实使用时这些 issue 也存在于仓库中；bot 提交的 issue 只是不作为查询。在全部 {f(sum(c['kept']['duplicate'].values()))} 条查重查询中，原 issue 由 bot 提交的有 {f(diag['duplicate_canonical_bot_authored'])} 条，原 issue 不在拉取结果里的有 {f(diag['duplicate_canonical_missing_from_pull'])} 条。</p>

<h2>3. 每条筛选规则去掉了多少</h2>
<p>筛选规则按表中的顺序依次执行，每条 issue 只计入第一条把它去掉的规则。因此“筛选前”减去各步去掉的数量，正好等于“保留”的数量。2023 年和 2024 年创建的 issue 不属于任何一个集合，只作为查重的检索范围，所以不出现在这些表里。</p>
{''.join(removal_tables)}

<h2>4. 测试集中与训练集文本完全相同的 issue</h2>
<p>按照说明 A7，这类 issue 保留在测试集里，但做了标记，以后报告分数时会给出两个数字：全部测试数据的分数，以及去掉这些 issue 之后的分数。判断方法是：取 issue 的标题加正文，去掉 HTML 注释（多为 issue 模板里的提示文字），正文只取前 2,000 个字符（和 PoC 的 <code>issue_text</code> 一致），再把连续的空白合并成一个空格并统一转成小写；如果测试集某条 issue 处理后的文本和同一项判断的训练集里某条 issue 完全一样，就标记它。</p>
<div class="scroll"><table><thead><tr><th>判断</th><th class="num">被标记的测试 issue</th><th class="num">测试集总数</th><th class="num">占比</th><th>例子（最多列 10 条）</th></tr></thead><tbody>{''.join(flag_rows)}</tbody></table></div>
<p>完整的标记列表在 <code>manifest/counts.json</code> 的 <code>exact_dup_flags</code> 字段，以及各测试集 tsv 文件的 <code>exact_dup_of_train</code> 列。</p>

<h2>5. bot 第一次分派的覆盖情况</h2>
<p>说明 A2 要求找出 vscode 的分诊 bot（<code>vs-code-engineering[bot]</code>）第一次把 issue 分给了谁，留到基线阶段和我们的负责人判断做对照。做法是：拉取负责人测试集每条 issue 的时间线，取第一个由这个 bot 发出的 <code>assigned</code>（分派）事件，记下被分派的人。</p>
<p>负责人测试集共有 {f(tl['issues'])} 条 issue，其中 {f(tl['timelines_available'])} 条成功拉到了时间线，{f(tl['timelines_missing'])} 条拉不到（例如 issue 已被删除或转移）。在全部 {f(tl['issues'])} 条中，有 <b>{f(tl['with_bot_first_assignment'])}</b> 条（{pct(tl['with_bot_first_assignment'], tl['issues'])}）找到了 bot 的第一次分派；其余 issue 的时间线里没有这个 bot 发出的分派事件（通常是维护者直接分派的）。bot 分派得对不对，留到基线阶段再统计。结果保存在 <code>manifest/bot_first_assignment.tsv</code>。</p>

<h2>6. 其他检查</h2>
<p><b>正文为空的 issue。</b>说明 A4 要求每条 issue 都有正文。正文为空（去掉空白后没有任何字符）的 issue 在筛选中直接去掉，去掉的数量见第 3 节各表中“正文为空”一行。</p>
<p><b>bot 判定。</b>提交者满足以下任一条件，就认为是 bot：登录名在 PoC 的 bot 账号名单中，或以 <code>[bot]</code> 结尾；或者 GitHub 标注的账号类型是 Bot。在 2025-01 到 2026-09 的 issue 中，只靠账号类型才识别出来的 bot issue 有 {f(diag['bot_by_typename_only'])} 条。</p>
<p><b>类型标签的分布。</b>下表是筛选后训练集和测试集中各类型标签的数量，供以后看 macro-F1 时参考。</p>
<div class="scroll"><table><thead><tr><th>类型标签</th><th class="num">训练集</th><th class="num">测试集</th></tr></thead><tbody>{type_dist}</tbody></table></div>

<h2>7. 人工核对</h2>
<p>核对表在 <a href="review_sheet.html">review_sheet.html</a>。每项判断从测试集里随机抽了 {rs['n_per_judgment']} 条（随机种子 {rs['seed']}），抽到的编号保存在 <code>review/review_sample.json</code>。表里已经列好每条 issue 的链接、标题、正文开头约 400 个字符和正确答案，判断一栏留空，需要你逐条选择“对”“错”或“无法判断”。输入会自动保存在浏览器里，核对完请用每一节的“导出 CSV”按钮导出结果。核对结果出来之后，再计算每项的正确率和置信区间，并和 A4 的 85% 标准对比。</p>
<p>注意：核对表里含有 issue 正文的片段。实验说明规定公开 repo 里不放 issue 正文，所以发布前需要把核对表换成只含编号和核对结果的版本。</p>

<h2>8. 如何重新运行</h2>
<p>在 worktree 根目录执行下面这条命令，就会依次完成拉取、筛选与切分、时间线拉取、存档、核对表和本报告。拉取脚本可以中断后续跑：已经拉到的页面和时间线会缓存在 <code>data/raw/</code> 里，不会重复请求。这次快照拉取 issue 用了 {dur(ip['started_at'], ip['finished_at'])}，拉取时间线用了 {dur(tp['first_fetched_at'], tp['last_fetched_at'])}（REST 接口每小时限 5,000 次请求，脚本在额度不足时会自动等待到额度重置）。</p>
<p>因为拉取结果会被缓存，在已有 <code>data/raw/</code> 的情况下再次运行这条命令不会重新拉取，而是在同一个快照上重新计算，结果和本页完全一致（已核对过两次运行的校验值相同）。如果要拉一个新的快照，需要先删掉或移走 <code>data/raw/</code> 目录。</p>
<pre>./run_data.sh</pre>
<p>如果只想在已有数据上重新筛选和生成报告，不再请求 GitHub，可以运行 <code>./run_data.sh --offline</code>。</p>
</main><script>{THEME_JS}</script></body></html>
"""
    (REVIEW / "data_report.html").write_text(page)
    print("report written")


if __name__ == "__main__":
    main()
