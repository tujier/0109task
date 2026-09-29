import json
import statistics
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from eval.config import EvalConfig

console = Console()

METRIC_LABELS = {
    "answer_correctness": "准确性 (0.30)",
    "helpfulness": "有用性 (0.30)",
    "no_hallucination": "不瞎编 (0.20)",
    "emotion_alignment": "语气好 (0.15)",
    "specificity": "针对性 (0.05)",
}


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), 4)


def print_report(results: List[Dict[str, Any]], config: EvalConfig) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📊 客服自动回复质量评估报告[/bold cyan]",
        border_style="cyan",
    ))

    total_scores = [r["total_score"] for r in results]
    avg_total = _safe_mean(total_scores)
    min_total = min(total_scores)
    max_total = max(total_scores)
    std_total = round(statistics.stdev(total_scores), 4) if len(total_scores) > 1 else 0.0

    summary_table = Table(title="整体概况", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    summary_table.add_column("指标", style="cyan")
    summary_table.add_column("数值", justify="right")

    summary_table.add_row("评估 case 总数", str(len(results)))
    summary_table.add_row("综合得分 (加权平均)", f"[bold green]{avg_total:.4f}[/bold green] (满分 1.0)")
    summary_table.add_row("最低分", f"{min_total:.4f}")
    summary_table.add_row("最高分", f"{max_total:.4f}")
    summary_table.add_row("标准差", f"{std_total:.4f}")
    console.print(summary_table)

    console.print()

    metric_stats: Dict[str, List[float]] = {m: [] for m in config.weights.keys()}
    for r in results:
        for m in config.weights.keys():
            score = r["metrics"].get(m, {}).get("score")
            if score is not None:
                metric_stats[m].append(float(score))

    metric_table = Table(title="各指标分布 (1-5分)", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    metric_table.add_column("指标", style="cyan")
    metric_table.add_column("权重", justify="right")
    metric_table.add_column("平均分", justify="right")
    metric_table.add_column("最低", justify="right")
    metric_table.add_column("最高", justify="right")
    metric_table.add_column("标准差", justify="right")

    for m, w in config.weights.items():
        scores = metric_stats[m]
        if scores:
            mean_s = _safe_mean(scores)
            min_s = min(scores)
            max_s = max(scores)
            std_s = round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0
        else:
            mean_s = min_s = max_s = std_s = 0.0
        metric_table.add_row(
            METRIC_LABELS.get(m, m),
            f"{w:.2f}",
            f"{mean_s:.2f}",
            f"{min_s:.1f}",
            f"{max_s:.1f}",
            f"{std_s:.2f}",
        )
    console.print(metric_table)

    failed_items = [(r["id"], m) for r in results for m in r.get("failed_metrics", [])]
    if failed_items:
        console.print()
        failed_text = "、".join(f"{cid}·{METRIC_LABELS.get(m, m)}" for cid, m in failed_items)
        console.print(Panel(
            f"[bold red]⚠️  有 {len(failed_items)} 项指标评估失败（未计入统计，已从综合分中剔除）[/bold red]\n[dim]{failed_text}[/dim]",
            border_style="red",
        ))

    console.print()
    console.print(Panel("[bold red]⚠️  最差 3 条 Case 深度分析[/bold red]", border_style="red"))

    sorted_results = sorted(results, key=lambda x: x["total_score"])
    for i, r in enumerate(sorted_results[:3]):
        _print_case_detail(r, rank=i + 1)

    console.print()
    console.print(Panel("[bold green]✅ 评估完成[/bold green]", border_style="green"))


def _print_case_detail(result: Dict[str, Any], rank: int) -> None:
    incomplete_note = " [bold red]⚠️部分指标失败[/bold red]" if result.get("incomplete") else ""
    panel_content = f"""[bold]#{rank}  {result['id']}  综合得分: [red]{result['total_score']:.4f}[/red]{incomplete_note}[/bold]

[cyan]用户问题：[/cyan]{result['user_question']}

[cyan]自动回复：[/cyan]{result['auto_reply']}

[cyan]人工参考：[/cyan]{result.get('human_reference', 'N/A')}

[cyan]标注员分析：[/cyan]{result.get('annotator_notes', 'N/A')}

[bold]各指标得分：[/bold]
"""

    for m, data in result["metrics"].items():
        metric_label = METRIC_LABELS.get(m, m)
        score = data.get("score")
        if score is None:
            panel_content += f"  [dim]{metric_label}[/dim]: [bold red]⚠️ 失败[/bold red]"
            reason = data.get("reasoning", "")
            if reason:
                panel_content += f"  [dim]— {reason[:80]}[/dim]"
            panel_content += "\n"
            continue
        color = "green" if score >= 4 else ("yellow" if score >= 3 else "red")
        panel_content += f"  [dim]{metric_label}[/dim]: [{color}]{score}/5[/{color}]"
        reason = data.get("reasoning", "")
        if reason and reason != "(RAGAS answer_correctness)":
            panel_content += f"  [dim]— {reason[:80]}[/dim]"
        panel_content += "\n"

    console.print(Panel(panel_content, border_style="red", expand=False))


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    serializable_results = []
    for r in results:
        serializable_results.append(r)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    console.print(f"\n💾 完整结果已保存至 [bold]{output_path}[/bold]")