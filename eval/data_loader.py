import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class EvalCase:
    id: str
    user_question: str
    auto_reply: str
    human_reference: Optional[str]
    annotator_notes: Optional[str]

    def to_ragas_row(self) -> Dict[str, Any]:
        return {
            "question": self.user_question,
            "answer": self.auto_reply,
            "reference": self.human_reference or "",
            "contexts": [],
        }


def load_cases(data_dir: str, auto_file: str, human_file: str) -> List[EvalCase]:
    auto_path = os.path.join(data_dir, auto_file)
    human_path = os.path.join(data_dir, human_file)

    if not os.path.exists(auto_path):
        raise FileNotFoundError(f"自动回复文件不存在: {auto_path}")
    if not os.path.exists(human_path):
        raise FileNotFoundError(f"人工参考文件不存在: {human_path}")

    with open(auto_path, "r", encoding="utf-8") as f:
        auto_replies = json.load(f)
    with open(human_path, "r", encoding="utf-8") as f:
        human_refs = json.load(f)

    human_map = {item["id"]: item for item in human_refs}

    cases: List[EvalCase] = []
    for item in auto_replies:
        case_id = item["id"]
        human_item = human_map.get(case_id, {})
        cases.append(EvalCase(
            id=case_id,
            user_question=item["user_question"],
            auto_reply=item["auto_reply"],
            human_reference=human_item.get("human_reference"),
            annotator_notes=human_item.get("annotator_notes"),
        ))

    return cases