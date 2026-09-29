from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EvalConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "answer_correctness": 0.30,
        "helpfulness": 0.30,
        "no_hallucination": 0.20,
        "emotion_alignment": 0.15,
        "specificity": 0.05,
    })

    data_dir: str = "data"
    auto_replies_file: str = "task3_auto_replies.json"
    human_ref_file: str = "task3_human_ref.json"

    llm_temperature: float = 0.0

    score_scale: int = 5

    def total_weight(self) -> float:
        return sum(self.weights.values())