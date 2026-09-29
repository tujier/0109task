import os
import sys

from dotenv import load_dotenv

load_dotenv()

from eval.config import EvalConfig
from eval.evaluator import run_evaluation
from eval.report import print_report, save_results


def main():
    config = EvalConfig()

    try:
        results = run_evaluation(config)
    except Exception as e:
        print(f"\n❌ 评估流水线失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print_report(results, config)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, os.path.join(output_dir, "eval_results.json"))


if __name__ == "__main__":
    main()