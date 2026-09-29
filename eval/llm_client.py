import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def _get_provider_config():
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if "deepseek" in provider:
        return {
            "provider": "deepseek",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "model": os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "base_url": "https://api.deepseek.com/v1",
        }
    elif "qwen" in provider:
        return {
            "provider": "qwen",
            "api_key": os.getenv("QWEN_API_KEY"),
            "model": os.getenv("LLM_MODEL") or os.getenv("QWEN_MODEL", "qwen-turbo"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {provider}，请选择 deepseek 或 qwen")


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    cfg = _get_provider_config()
    if not cfg["api_key"]:
        raise ValueError(f"{cfg['provider'].upper()}_API_KEY 未配置，请在 .env 中设置")
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=temperature,
    )


def get_ragas_llm():
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    cfg = _get_provider_config()
    if not cfg["api_key"]:
        raise ValueError(f"{cfg['provider'].upper()}_API_KEY 未配置，请在 .env 中设置")

    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    # max_tokens 需覆盖评分输出的全部内容；原来 4096 会被 deepseek 推理模型思考占满导致截断
    return llm_factory(model=cfg["model"], client=client, max_tokens=8192)


def call_llm(system_prompt: str, user_prompt: str, llm: ChatOpenAI) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content