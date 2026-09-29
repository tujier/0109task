import time
from typing import List, Dict, Any

from eval.config import EvalConfig
from eval.data_loader import EvalCase, load_cases
from eval.llm_client import get_llm, get_ragas_llm
from eval.metrics import ragas_metrics
from eval.metrics import helpfulness, no_hallucination, emotion_alignment, specificity

METRIC_ORDER = [
    ("answer_correctness", "ragas", ragas_metrics),
    ("helpfulness", "custom", helpfulness),
    ("no_hallucination", "custom", no_hallucination),
    ("emotion_alignment", "custom", emotion_alignment),
    ("specificity", "custom", specificity),
]

METRIC_NAMES_CN = {
    "answer_correctness": "准确性",
    "helpfulness": "有用性",
    "no_hallucination": "不瞎编",
    "emotion_alignment": "语气好",
    "specificity": "针对性",
}


def run_evaluation(config: EvalConfig) -> List[Dict[str, Any]]:
    print("=== 客服自动回复质量评估流水线 ===\n")

    print("[1/3] 加载数据...")
    cases = load_cases(config.data_dir, config.auto_replies_file, config.human_ref_file)
    print(f"    ✅ 加载了 {len(cases)} 条评估 case\n")

    print("[2/3] 初始化 LLM...")
    llm = get_llm(temperature=config.llm_temperature)
    ragas_llm = get_ragas_llm()
    provider = __import__("os").getenv("LLM_PROVIDER", "deepseek")
    print(f"    ✅ LLM 初始化成功 (provider={provider})\n")

    all_results: List[Dict[str, Any]] = []
    total = len(cases)
    metric_total = len(METRIC_ORDER)

    print(f"[3/3] 评估中：{total} 条 case × {metric_total} 个指标 = {total * metric_total} 步")
    print("-" * 60)

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{total}] {case.id}  「{case.user_question[:30]}...」")

        case_result: Dict[str, Any] = {
            "id": case.id,
            "user_question": case.user_question,
            "auto_reply": case.auto_reply,
            "human_reference": case.human_reference,
            "annotator_notes": case.annotator_notes,
            "metrics": {},
        }

        for j, (metric_name, metric_type, metric_module) in enumerate(METRIC_ORDER, 1):
            cn = METRIC_NAMES_CN.get(metric_name, metric_name)
            print(f"    ({j}/{metric_total}) {cn}... ", end="", flush=True)
            t0 = time.time()

            try:
                if metric_type == "ragas":
                    result = metric_module.evaluate(case, ragas_llm)
                else:
                    result = metric_module.evaluate(case, llm)
                case_result["metrics"][metric_name] = result
                elapsed = time.time() - t0
                score = result.get("score", "?")
                print(f"✅  score={score}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"❌  失败: {e}  ({elapsed:.1f}s)")
                case_result["metrics"][metric_name] = {
                    "score": None,
                    "reasoning": f"评估失败: {e}",
                }
            time.sleep(0.2)

        total_score = 0.0
        weight_sum = 0.0
        failed_metrics: List[str] = []
        for m, w in config.weights.items():
            md = case_result["metrics"].get(m, {})
            score = md.get("score")
            if score is None:
                failed_metrics.append(m)
                continue
            total_score += (float(score) / config.score_scale) * w
            weight_sum += w

        case_result["total_score"] = round(total_score / weight_sum, 4) if weight_sum else 0.0
        case_result["failed_metrics"] = failed_metrics
        case_result["incomplete"] = bool(failed_metrics)
        all_results.append(case_result)

        status = " ⚠️  部分指标失败" if failed_metrics else ""
        print(f"    📊 综合得分: {case_result['total_score']:.4f}{status}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成，共 {len(all_results)} 条 case")
    return all_results