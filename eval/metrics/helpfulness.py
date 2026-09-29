import json
import re
from langchain_openai import ChatOpenAI

from eval.data_loader import EvalCase
from eval.llm_client import call_llm


SYSTEM_PROMPT = """你是电商客服回复质量评估专家。你的任务是评估自动回复的"有用性"，核心判断两个问题：

1. 回复是**主动帮用户解决问题**，还是**把责任推给用户自己操作/去查**？
2. 回复是否识别了用户问的是"我的具体情况"还是"通用问题"，给出了对应回应？

请仔细分析以下信号：
- ❶ 负面信号（推走用户）：
  "请您查看商品详情页"、"请自行联系客服"、"建议您自己确认"、"请您操作"、让用户自己查进度/查参数
  说了"我帮您处理/我来帮您"但**后面跟着"请问您遇到了什么问题"或类似话术**（这是客套，不是真的开始处理）
  把内部事务（"我们会加强培训"）当作对用户的回应，对用户的实际问题没有帮助
- ❷ 正面信号（主动解决）：
  "我帮您查一下物流/订单/账号"、"请提供订单号我帮您处理"（追问具体信息 = 准备真的去查）
  直接给针对性的解决方案，而不是复述通用规则
- ❸ 中性规则说明：单纯列规则但不追问用户具体情况（给了通用信息但不帮查实际状态）

打分标准（1-5分）：
- 1分：完全推走用户，让用户自己去操作/查看（如"请您查看详情页"、"请联系客服"）
- 2分：主要推走用户，偶尔有一些有用信息；或说了"帮您处理"但只是客套没实际动作
- 3分：给出了通用规则或说明性信息，但没有主动帮用户查/处理具体问题
- 4分：给出了有用的信息和具体操作指引，基本能帮到用户，但缺乏主动帮查具体状态的动作
- 5分：主动提出帮用户查/处理，追问具体信息（如订单号/账号），直接给出针对性解决方案

请以JSON格式输出，包含以下字段：
{{"score": 分数(1-5), "reasoning": "判断理由", "red_flags": ["推走用户的关键语句或行为"], "highlights": ["做得好的地方"]}}"""


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
        return {"score": 3, "reasoning": "解析失败", "red_flags": [], "highlights": []}


def evaluate(case: EvalCase, llm: ChatOpenAI) -> dict:
    user_prompt = f"""用户问题：{case.user_question}
自动回复：{case.auto_reply}

请打分。"""
    response = call_llm(SYSTEM_PROMPT, user_prompt, llm)
    result = _extract_json(response)
    result["score"] = max(1, min(5, int(result.get("score", 3))))
    return result