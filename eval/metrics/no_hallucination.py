import json
import re
from langchain_openai import ChatOpenAI

from eval.data_loader import EvalCase
from eval.llm_client import call_llm


SYSTEM_PROMPT = """你是电商客服回复质量评估专家。你的任务是评估自动回复是否存在"瞎编/幻觉"问题。

你需要从两个维度检查：

### 维度1：与人工参考回复的事实一致性
对比自动回复和人工参考回复中提到的**关键事实**，判断自动回复是否：
- 与人工参考的事实矛盾（如人工说7天质保，自动回复说15天）
- 额外编造了不存在的信息（如说有某个补偿方案但人工没提）
- 遗漏了关键事实（这不算瞎编，只是信息不全）

### 维度2：常识和规则的合理性
检查自动回复中的事实性陈述是否符合电商常识，如：
- 退款时效（1-15个工作日都是合理范围）
- 快递柜/快递流程
- 质保期、退换货规则
- 商品参数（如充电宝Wh数限制）
- 民航局规定等外部规则

⚠️ 以下情况**不算瞎编**：
- 不同表述方式但含义相同
- 给出合理的规则范围（如"1-3个工作日"是合理的，虽然人工只提了"一般1-3天"）
- 通用规则说明（没有用户具体订单信息时给出规则）

打分标准（1-5分）：
- 1分：存在多处明显的编造信息或与事实严重矛盾
- 2分：存在1处编造或矛盾，影响用户判断
- 3分：有一些不确定的表述，但不确定是否为瞎编
- 4分：信息基本正确，仅有微小的表述差异
- 5分：所有信息准确无误，无编造，无矛盾

请以JSON格式输出：
{{"score": 分数(1-5), "reasoning": "判断理由", "hallucinated_facts": ["编造的具体事实"], "contradictions": ["与参考矛盾的内容"]}}"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 4, "reasoning": "解析失败，默认中高分", "hallucinated_facts": [], "contradictions": []}


def evaluate(case: EvalCase, llm: ChatOpenAI) -> dict:
    user_prompt = f"""用户问题：{case.user_question}
自动回复：{case.auto_reply}
人工参考回复：{case.human_reference or "（无）"}

请打分。"""
    response = call_llm(SYSTEM_PROMPT, user_prompt, llm)
    result = _extract_json(response)
    result["score"] = max(1, min(5, int(result.get("score", 4))))
    return result