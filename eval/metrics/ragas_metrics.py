import sys
import types


def _patch_vertexai():
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    fake = types.ModuleType(mod_name)
    fake.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[mod_name] = fake


_patch_vertexai()

from ragas.metrics.collections.answer_correctness import AnswerCorrectness

from eval.data_loader import EvalCase


def evaluate(case: EvalCase, ragas_llm) -> dict:
    metric = AnswerCorrectness(llm=ragas_llm, weights=[1.0, 0.0])

    result = metric.score(
        user_input=case.user_question,
        response=case.auto_reply,
        reference=case.human_reference or "",
    )

    score = float(result.value) if result.value is not None else 0.0
    score_5scale = round(score * 5, 2)

    return {
        "score": score_5scale,
        "reasoning": "(RAGAS answer_correctness)",
    }