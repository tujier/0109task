# 0109task - 客服自动回复质量评估

基于 RAGAS 框架 + 自定义 LLM-as-Judge 指标，对自动回复进行多维度质量评估。

## 指标定义

| 指标 | 权重 | 量化方式 | 对应业务方要求 |
|------|------|---------|---------------|
| answer_correctness | 0.30 | RAGAS AnswerCorrectness（事实一致性 F1 分数） | 准确 |
| helpfulness | 0.30 | LLM-as-Judge：主动帮用户解决 vs 推走用户 | 有用 |
| no_hallucination | 0.20 | LLM-as-Judge：事实冲突与常识矛盾检测 | 不瞎编 |
| emotion_alignment | 0.15 | LLM-as-Judge：情绪识别与针对性回应质量 | 语气好 |
| specificity | 0.05 | LLM-as-Judge：具体问题 vs 通用规则识别 | 有用的子维度 |

**综合得分 = Σ(指标分/5 × 权重)**，满分 1.0。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置 API Key（编辑 .env，复制 .env.example 后填写）
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=sk-your-key

# 运行评估
uv run python run_eval.py
```

## 切换大模型

编辑 `.env`，修改 `LLM_PROVIDER`：

```ini
# 使用 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx

# 使用 Qwen
LLM_PROVIDER=qwen
QWEN_API_KEY=sk-xxx
```

## 调整权重

编辑 `eval/config.py`：

```python
weights = {
    "answer_correctness": 0.30,
    "helpfulness": 0.30,
    "no_hallucination": 0.20,
    "emotion_alignment": 0.15,
    "specificity": 0.05,
}
```

## 评估结果

20 条自动回复评估已完成，结果保存于 `output/eval_results.json`。

- 综合得分（加权平均）：**0.588 / 1.0**
- 最高分：0.795（case_08 — 手机壳材质咨询）
- 最低分：0.390（case_13 — 面膜成分 + 皮肤敏感）
- 标准差：0.113

## 文件结构

```
0109task/
├── data/
│   ├── task3_auto_replies.json      # 20 条自动回复
│   ├── task3_human_ref.json         # 人工标注参考回复
│   └── task3_eval_criteria.md       # 业务方评估要求
├── eval/
│   ├── config.py                    # 配置（权重、路径、参数）
│   ├── data_loader.py               # 数据加载与对齐
│   ├── llm_client.py                # LLM 封装（deepseek/qwen 可切换）
│   ├── evaluator.py                 # 主流水线
│   ├── report.py                    # 报告生成（终端 + JSON）
│   └── metrics/
│       ├── ragas_metrics.py         # RAGAS AnswerCorrectness
│       ├── helpfulness.py           # 自定义：行动导向
│       ├── no_hallucination.py      # 自定义：不瞎编
│       ├── emotion_alignment.py     # 自定义：情绪对齐
│       └── specificity.py           # 自定义：针对性
├── output/
│   └── eval_results.json            # 完整评估结果
├── .env.example                     # 环境变量模板
├── pyproject.toml                   # 依赖声明
└── run_eval.py                      # 入口脚本
```

## 局限性

1. **RAGAS AnswerCorrectness 低估操作指引类回复**：6/20 条因自动回复类型（操作指引）与人工参考（事实陈述）不匹配导致 accuracy 为 0。改进：替换为自定义 LLM-as-Judge 指标，或加入 embedding 做语义相似度。
2. **依赖 LLM 判断**：4 个自定义指标均使用 LLM-as-Judge，存在随机性。改进：多次采样取平均。
3. **no_hallucination 零区分度**：当前数据无瞎编 case，该指标占权重但无区分作用。改进：缩减权重至 0.10，加给 helpfulness。
4. **领域知识限制**：涉及特定平台规则时，LLM 常识判断可能不准。改进：引入平台内部知识库作为 contexts。

## AI 工具使用情况

- **评估框架**：RAGAS 0.4.x（AnswerCorrectness）
- **LLM 评估**：deepseek-chat / qwen-turbo（通过 langchain + OpenAI 兼容接口接入）
- **代码实现**：基于项目需求完成代码编写、数据结构设计、评估 Prompt 设计
- **报告生成**：pandas 统计计算 + rich 库终端输出

## 图片