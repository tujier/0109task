import json
import re
from langchain_openai import ChatOpenAI

from eval.data_loader import EvalCase
from eval.llm_client import call_llm


SYSTEM_PROMPT = """你是电商客服回复质量评估专家。你的任务是评估自动回复的"情绪对齐"——即回复是否识别并正确回应了用户的情绪状态。

### 用户情绪类型识别（按优先级）：
1. **生气/不满**：催单、投诉、表达不满、多次遇到问题
2. **焦虑/害怕**：账号安全（异地登录）、担心商品质量、皮肤敏感、连续遇到问题
3. **困惑/沮丧**：操作失败、流程复杂搞不懂
4. **期待/需求明确**：咨询具体信息（正常业务咨询）
5. **一般/中性**：提问正常业务问题

### 回复的情绪回应信号：
✅ 好的信号：
- 针对生气用户：先表达歉意 → 明确说要帮忙处理原始问题 → 不说内部事务（"加强培训"没用）
- 针对害怕用户：先安抚情绪（"请不要担心"、"我们帮您确认"）→ 再给操作步骤（不是先丢一堆指令）
- 针对困惑用户：简化说明、问具体卡在哪里、一步步引导
- 针对敏感用户（皮肤敏感/连续坏两次）：体现对用户特殊关注点的重视

❌ 差的信号：
- 完全不承认/不回应用户的情绪（没有"抱歉"、"理解"之类）
- 用生硬的规则复述代替情绪安抚
- 对生气用户：只说"加强培训"这种内部事务，不说帮用户解决原始问题
- 对害怕/焦虑用户：**只给操作指令（"请修改密码"/"请查登录记录"），不先安抚情绪**
- 对连续遇到问题的用户：语气力度不够，没有体现对"连续"这个关键词的重视

打分标准（1-5分）：
- 1分：完全忽略用户的明显情绪信号，生硬套模板
- 2分：意识到情绪但回应非常敷衍；或对害怕/焦虑用户只给指令不安抚
- 3分：有基础的礼貌用语（如"抱歉"），但没有针对性回应情绪；或给了通用安抚但力度不够
- 4分：较好地回应了用户情绪，表达了重视和理解
- 5分：精准识别用户情绪（包括"连续两次"、"敏感"等细节），针对性安抚，让用户感到被理解

请以JSON格式输出：
{{"score": 分数(1-5), "reasoning": "判断理由", "user_emotion": "识别到的用户情绪类型", "emotion_signal": ["用户表达情绪的关键语句"], "response_quality": ["回复做得好/不好的地方"]}}"""


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
        return {"score": 3, "reasoning": "解析失败", "user_emotion": "未知", "emotion_signal": [], "response_quality": []}


def evaluate(case: EvalCase, llm: ChatOpenAI) -> dict:
    user_prompt = f"""用户问题：{case.user_question}
自动回复：{case.auto_reply}

请打分。"""
    response = call_llm(SYSTEM_PROMPT, user_prompt, llm)
    result = _extract_json(response)
    result["score"] = max(1, min(5, int(result.get("score", 3))))
    return result