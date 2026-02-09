# 多Agent多轮对话合成系统

## 概述

这是一个基于多Agent框架的对话轨迹合成系统，用于生成真实的用户-助手-工具交互数据。系统模拟三个专用角色：

- **User Agent**: 模拟真实用户，生成自然语言查询
- **Assistant Agent**: 核心助手，负责推理、规划工具调用、生成chain-of-thought
- **Tool Agent**: 模拟工具执行，返回合理的模拟结果

## 系统架构

```
用户查询 → Assistant推理 → 工具调用 → 工具返回 → Assistant处理 → 用户响应 → ...
```

### 核心特性

1. **真实的多轮交互**: 支持助手主动澄清、多轮对话、自纠正
2. **Chain-of-Thought推理**: 每个Assistant响应都包含明确的推理步骤
3. **动态参数生成**: 从上下文提取参数或智能生成模拟值
4. **LLM驱动**: 使用DashScope API生成自然、真实的对话内容

## 文件结构

```
code/
├── MultiAgentDialogueSynthesis.py  # 主要合成系统
├── test_synthesis.py                # 测试脚本
├── requirements.txt                 # Python依赖
├── .env                            # API密钥配置
└── README_MultiAgent.md            # 本文档

data/
├── refined_tools.json              # 工具描述数据
├── sampled_tool_chains.json        # 采样的工具链
├── synthesized_dialogues.json      # 输出：合成的对话
└── test_dialogues.json             # 测试输出
```

## 安装依赖

```bash
cd code
pip install -r requirements.txt
```

## 配置

确保 `.env` 文件包含以下配置：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

## 使用方法

### 1. 测试单个对话合成

```bash
python test_synthesis.py
```

这会：
- 选择一个工具链
- 生成完整的多轮对话
- 打印详细的对话过程
- 显示统计信息

### 2. 批量合成对话

修改 `MultiAgentDialogueSynthesis.py` 中的 `main()` 函数：

```python
# 调整合成数量
num_samples = 50  # 合成50个对话
```

然后运行：

```bash
python MultiAgentDialogueSynthesis.py
```

### 3. 在代码中使用

```python
from MultiAgentDialogueSynthesis import MultiAgentDialogueSynthesis

# 初始化
synthesizer = MultiAgentDialogueSynthesis(
    tools_file='data/refined_tools.json',
    chains_file='data/sampled_tool_chains.json'
)

# 合成单个对话
chain = ['Tool1', 'Tool2', 'Tool3']
dialogue = synthesizer.synthesize_dialogue(chain)

# 批量合成
dialogues = synthesizer.batch_synthesize(
    num_samples=100,
    output_file='data/output.json'
)
```

## 输出格式

每个合成的对话包含以下结构：

```json
{
  "chain_length": 4,
  "chain": ["Tool1", "Tool2", "Tool3", "Tool4"],
  "conversation": [
    {
      "role": "user",
      "content": "用户查询内容",
      "turn": 1
    },
    {
      "role": "assistant",
      "content": "助手响应",
      "reasoning": ["推理步骤1", "推理步骤2"],
      "tool_call": {
        "tool_name": "Tool1",
        "parameters": {"param1": "value1"}
      },
      "turn": 2
    },
    {
      "role": "tool",
      "tool_name": "Tool1",
      "tool_result": {"status": "success", "data": {}},
      "turn": 3
    }
  ],
  "total_turns": 10,
  "assistant_turns": 5
}
```

## 关键功能说明

### 1. User Agent - 查询生成

使用LLM根据工具链生成自然的用户查询：
- 口语化表达
- 不直接提及工具名称
- 体现真实用户意图

### 2. Assistant Agent - 推理与决策

生成包含以下内容的响应：
- **推理步骤**: 明确的思考过程
- **工具选择**: 基于上下文选择合适工具
- **参数生成**: 从上下文提取或智能生成
- **澄清机制**: 信息不足时主动询问

### 3. Tool Agent - 结果模拟

使用LLM生成合理的工具返回结果：
- 符合工具功能描述
- 包含合理的模拟数据
- JSON格式输出

## 参数调整

### 控制对话长度

```python
dialogue = synthesizer.synthesize_dialogue(
    chain=tool_chain,
    max_turns=15  # 最大对话轮数
)
```

### 调整澄清概率

在 `_check_clarification_needed()` 方法中：

```python
if random.random() < 0.15:  # 调整这个值（0-1）
    # 生成澄清问题
```

### 调整LLM温度

在各个LLM调用中修改 `temperature` 参数：

```python
response = self.client.chat.completions.create(
    model="qwen-plus",
    messages=[...],
    temperature=0.7  # 0.0-1.0，越高越随机
)
```

## 性能优化

1. **批量保存**: 每10个对话自动保存一次，防止数据丢失
2. **错误处理**: LLM调用失败时自动回退到规则方法
3. **上下文管理**: 只保留最近3轮对话历史，减少token消耗

## 预期输出

根据ToolACE论文，目标是合成约160k个assistant响应轮次。

示例统计：
- 44个工具链
- 每个链平均生成5-8轮assistant响应
- 总计约220-350个assistant轮次

要达到160k轮次，需要处理约20k-30k个工具链。

## 故障排除

### 1. API密钥错误

```
ValueError: 未找到DASHSCOPE_API_KEY环境变量
```

解决：检查 `.env` 文件是否正确配置

### 2. JSON解析错误

LLM返回的内容可能包含markdown格式，系统会自动提取JSON部分

### 3. 工具未找到

确保 `refined_tools.json` 包含工具链中引用的所有工具

## 扩展建议

1. **添加更多Agent角色**: 如Evaluator Agent评估对话质量
2. **支持并行处理**: 使用多线程/多进程加速批量合成
3. **质量过滤**: 添加对话质量评分和过滤机制
4. **多样性增强**: 为同一工具链生成多个不同的对话变体

## 参考

基于ToolACE论文的多Agent多轮轨迹合成方法实现。
