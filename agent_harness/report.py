"""Single-run reports: Markdown (for terminals / PRs) and a self-contained HTML trace viewer."""

from __future__ import annotations

import html
import json

from .metrics import summarize
from .schema import RunResult


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def to_markdown(run: RunResult) -> str:
    s = summarize(run)
    lo, hi = s["success_ci95"]
    lines = [
        f"# Agent reliability report — `{run.suite}` / `{run.label}`",
        "",
        (f"- Generated {run.created_at} · git `{run.git_sha or 'n/a'}` · "
         f"{s['tasks']} tasks × {run.trials_per_task} trial(s) = {s['trials']} runs"),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success (pass@1) | **{s['success_rate']:.0%}** (95% CI {lo:.0%}–{hi:.0%}) |",
        f"| Passed every trial (pass^{run.trials_per_task}) | {s['pass_all_k']:.0%} |",
        f"| Tool-use checks passed | {_pct(s['tool_use_accuracy'])} |",
        f"| Final-state checks passed | {_pct(s['state_accuracy'])} |",
        f"| Quality (judge) checks passed | {_pct(s['quality_pass_rate'])} |",
        f"| Budget checks passed | {_pct(s['budget_pass_rate'])} |",
        f"| Mean cost / task | ${s['mean_cost_usd']:.4f} (judge: ${s['judge_cost_usd']:.2f} total) |",
        f"| Latency p50 / p95 | {s['latency_p50_s']:.1f}s / {s['latency_p95_s']:.1f}s |",
        f"| Mean steps | {s['mean_steps']:.1f} ({s['mean_llm_calls']:.1f} LLM, {s['mean_tool_calls']:.1f} tool) |",
        "",
    ]
    if s["flaky_tasks"]:
        lines += [f"**Flaky tasks** (mixed results across trials): {', '.join(f'`{t}`' for t in s['flaky_tasks'])}", ""]
    if s["top_failures"]:
        lines += ["## Most common failing checks", "", "| Check | Failures |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in s["top_failures"].items()] + [""]

    lines += ["## Per task", "", "| Task | Passed | Mean cost | Mean latency | Mean steps |", "|---|---|---|---|---|"]
    for tid, t in s["per_task"].items():
        lines.append(f"| `{tid}` | {t['passes']}/{t['trials']} | ${t['mean_cost_usd']:.4f} | "
                     f"{t['mean_latency_s']:.1f}s | {t['mean_steps']:.1f} |")
    lines.append("")

    failed = [t for t in run.trials if not t.success]
    if failed:
        lines += ["## Failure details", ""]
        for t in failed:
            lines.append(f"### `{t.task_id}` trial {t.trial} — path: {' → '.join(t.node_path) or 'n/a'}")
            if t.crashed:
                lines.append(f"- crashed: `{t.crashed}`")
            for c in t.checks:
                if c.required and not c.passed:
                    lines.append(f"- ❌ **{c.name}** ({c.category}): {c.detail}")
            lines.append("")
    return "\n".join(lines)


def to_html(run: RunResult) -> str:
    s = summarize(run)
    payload = {
        "suite": run.suite, "label": run.label, "created_at": run.created_at, "git_sha": run.git_sha,
        "summary": {k: v for k, v in s.items() if k != "per_task"}, "per_task": s["per_task"],
        "trials": [
            {
                "task_id": t.task_id, "trial": t.trial, "success": t.success, "latency_s": t.latency_s,
                "cost_usd": t.cost_usd, "steps": t.steps, "node_path": t.node_path, "crashed": t.crashed,
                "checks": [c.model_dump(include={"name", "category", "passed", "required", "detail", "score"})
                           for c in t.checks],
                "spans": [
                    {"kind": sp.kind, "name": sp.name, "node": sp.node, "start": sp.start, "end": sp.end,
                     "model": sp.model, "in": sp.input_tokens, "out": sp.output_tokens, "cost": sp.cost_usd,
                     "error": sp.error, "input": (sp.input or "")[:600], "output": (sp.output or "")[:600]}
                    for sp in t.trace.spans
                ],
            }
            for t in run.trials
        ],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    title = html.escape(f"{run.suite} · {run.label}")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1917;--muted:#78716c;--line:#e7e5e4;--pass:#15803d;--fail:#b91c1c;
--node:#a8a29e;--llm:#2563eb;--tool:#d97706;--fault:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--panel:#1c1917;--ink:#f5f5f4;--muted:#a8a29e;--line:#292524;
--pass:#4ade80;--fail:#f87171;--node:#57534e;--llm:#60a5fa;--tool:#fbbf24;--fault:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:24px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:12px}
.task{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:8px}
.task>summary{cursor:pointer;padding:10px 14px;display:flex;gap:12px;align-items:center;list-style:none}
.task>summary::-webkit-details-marker{display:none}.badge{font-weight:600;font-variant-numeric:tabular-nums;min-width:40px}
.ok{color:var(--pass)}.bad{color:var(--fail)}.meta{color:var(--muted);margin-left:auto;font-size:12px;font-variant-numeric:tabular-nums}
.trial{border-top:1px solid var(--line);padding:12px 14px}.checks{margin:6px 0 10px;padding:0;list-style:none}
.checks li{padding:2px 0;font-size:13px}.checks code{font-size:12px}.path{color:var(--muted);font-size:12px}
.wf{position:relative;border-left:1px solid var(--line)}.row{display:flex;align-items:center;height:20px;font-size:11px}
.row .lbl{width:180px;flex:none;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:var(--muted);padding-right:8px}
.row .track{position:relative;flex:1;height:12px}.bar{position:absolute;height:12px;border-radius:3px;min-width:2px;cursor:default}
.bar.node{background:var(--node)}.bar.llm{background:var(--llm)}.bar.tool{background:var(--tool)}.bar.fault{background:var(--fault);width:4px!important}
.bar.err{outline:2px solid var(--fail)}#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;max-width:420px;font-size:12px;white-space:pre-wrap;word-break:break-word;display:none;z-index:9;box-shadow:0 4px 16px #0003}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);margin:4px 0 16px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
@media (max-width:600px){.row .lbl{width:90px}}
</style></head><body><main>
<h1 id="h"></h1><div class="sub" id="sub"></div><div class="tiles" id="tiles"></div>
<div class="legend"><span><i style="background:var(--node)"></i>graph node</span><span><i style="background:var(--llm)"></i>LLM call</span>
<span><i style="background:var(--tool)"></i>tool call</span><span><i style="background:var(--fault)"></i>injected fault</span></div>
<div id="tasks"></div></main><div id="tip"></div>
<script>
const D=__DATA__;const S=D.summary;const pct=x=>x==null?'–':Math.round(x*100)+'%';const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('h').textContent=`${D.suite} — ${D.label}`;
document.getElementById('sub').textContent=`${D.created_at} · git ${D.git_sha||'n/a'} · ${S.tasks} tasks × ${S.trials_per_task} trial(s)`;
const tiles=[[pct(S.success_rate),`task success (95% CI ${pct(S.success_ci95[0])}–${pct(S.success_ci95[1])})`],[pct(S.pass_all_k),`pass^${S.trials_per_task}`],
[pct(S.tool_use_accuracy),'tool-use checks'],[pct(S.state_accuracy),'final-state checks'],[pct(S.quality_pass_rate),'quality checks'],
['$'+S.mean_cost_usd.toFixed(4),'mean cost / task'],[S.latency_p95_s.toFixed(1)+'s','p95 latency'],[S.mean_steps.toFixed(1),'mean steps']];
document.getElementById('tiles').innerHTML=tiles.map(([v,l])=>`<div class="tile"><b>${v}</b><span>${l}</span></div>`).join('');
const tip=document.getElementById('tip');const spans=[];
function wf(t){const end=Math.max(t.latency_s,...t.spans.map(s=>s.end||0))||1;return '<div class="wf">'+t.spans.map(s=>{const i=spans.push(s)-1;
const l=(s.start/end*100).toFixed(2),w=(((s.end||s.start)-s.start)/end*100).toFixed(2);const indent=s.kind==='node'?'':'  ';
return `<div class="row"><div class="lbl">${indent}${esc(s.kind==='node'?s.name:s.kind+': '+s.name)}</div><div class="track"><div class="bar ${s.kind}${s.error?' err':''}" data-i="${i}" style="left:${l}%;width:${w}%"></div></div></div>`}).join('')+'</div>'}
const byTask={};D.trials.forEach(t=>(byTask[t.task_id]=byTask[t.task_id]||[]).push(t));
document.getElementById('tasks').innerHTML=Object.entries(D.per_task).map(([id,p])=>{const ok=p.passes===p.trials;
return `<details class="task"><summary><span class="badge ${ok?'ok':'bad'}">${p.passes}/${p.trials}</span><code>${esc(id)}</code>
<span class="meta">$${p.mean_cost_usd.toFixed(4)} · ${p.mean_latency_s.toFixed(1)}s · ${p.mean_steps.toFixed(1)} steps</span></summary>`+
byTask[id].map(t=>`<div class="trial"><b class="${t.success?'ok':'bad'}">Trial ${t.trial} ${t.success?'passed':'failed'}</b>
<span class="path"> · ${esc(t.node_path.join(' → ')||'no nodes')} · ${t.latency_s.toFixed(1)}s · $${t.cost_usd.toFixed(4)}</span>
${t.crashed?`<div class="bad">crashed: ${esc(t.crashed)}</div>`:''}
<ul class="checks">${t.checks.map(c=>`<li><span class="${c.passed?'ok':'bad'}">${c.passed?'✓':'✗'}</span> <code>${esc(c.name)}</code> <span class="path">${esc(c.category)}${c.required?'':' · optional'}</span> — ${esc(c.detail)}</li>`).join('')}</ul>
${wf(t)}</div>`).join('')+'</details>'}).join('');
document.addEventListener('mousemove',e=>{const b=e.target.closest('.bar');if(!b){tip.style.display='none';return}const s=spans[+b.dataset.i];
let txt=`${s.kind}: ${s.name}${s.node&&s.kind!=='node'?'  (in '+s.node+')':''}\\n${((s.end||s.start)-s.start).toFixed(2)}s`;
if(s.kind==='llm')txt+=` · ${s.in} in / ${s.out} out · $${s.cost.toFixed(4)}\\nmodel: ${s.model}`;if(s.error)txt+=`\\nERROR: ${s.error}`;
if(s.input)txt+=`\\n\\n→ ${s.input.slice(0,300)}`;if(s.output)txt+=`\\n\\n← ${s.output.slice(0,300)}`;
tip.textContent=txt;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,innerWidth-440)+'px';tip.style.top=(e.clientY+12)+'px'});
</script></body></html>
"""
