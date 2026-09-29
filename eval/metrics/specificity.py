import json
import re
from langchain_openai import ChatOpenAI

from eval.data_loader import EvalCase
from eval.llm_client import call_llm


SYSTEM_PROMPT = """你是电商客服回复质量评估专家。你的任务是评估自动回复的"针对性"——即是否识别了用户的问题是针对**具体实体**（某个商品、某个订单、"我的"退款）还是**通用咨询**，并给出了对应回应。

### 判断标准：

1. **识别具体问题**（高针对性）：
   - 用户问的是"这个"充电宝、"我的"退款、"我那个退货"——指向具体商品/订单
   - 回复应该主动追问信息（订单号、商品名）或直接查询给出具体答案

2. **识别通用问题**（中等针对性）：
   - 用户问的是"退款规则是什么"、"退货邮费谁出"——通用规则
   - 回复给出规则说明即可

3. **识别模糊/多槽位问题**（需要追问）：
   - 用户问的是模糊指代（"那两款手机"）、一条消息多个问题、多件商品
   - 回复应该主动追问具体信息

打分标准（1-5分）：
- 1分：未识别具体问题，给了通用套话（用户问"这个XX"，回复说"请看详情页"）
- 2分：识别不清晰，部分对部分错
- 3分：基本识别了问题类型，但回应不够精准（该追问的没追问，该查的没查）
- 4分：正确识别问题类型，给出了合适的回应
- 5分：精准识别问题的所有槽位（包括多问题、模糊指代），主动追问/查询给出最精准回应

请以JSON格式输出：
{{"score": 分数(1-5), "reasoning": "判断理由", "question_type": "具体问题/通用问题/模糊多槽位", "missed_slots": ["未识别的信息槽位"], "suggestion": "改进建议"}}"""


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
        return {"score": 3, "reasoning": "解析失败", "question_type": "未知", "missed_slots": [], "suggestion": ""}


def evaluate(case: EvalCase, llm: ChatOpenAI) -> dict:
    user_prompt = f"""用户问题：{case.user_question}
自动回复：{case.auto_reply}
人工参考回复：{case.human_reference or "（无）"}

请打分。"""
    response = call_llm(SYSTEM_PROMPT, user_prompt, llm)
    result = _extract_json(response)
    result["score"] = max(1, min(5, int(result.get("score", 3))))
    return result