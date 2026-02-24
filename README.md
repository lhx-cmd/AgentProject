# ToolMind - 工具链规划智能体系统

基于函数图和随机游走的多工具链规划与偏好数据生成系统。

## 项目简介

ToolMind 是一个智能工具链规划系统，通过构建工具函数依赖图，自动生成合理的多步骤任务执行计划。系统支持：

- 基于语义相似度的工具函数依赖图构建
- 随机游走采样生成多样化工具链
- 自动生成工具链使用场景
- 智能任务规划与多维度评价
- 偏好数据生成（用于强化学习/DPO训练）


### 环境要求

```bash
pip install torch sentence-transformers dashscope openai python-dotenv tqdm
```

### 配置API密钥

创建 `code/.env` 文件：

```env
DASHSCOPE_API_KEY=your_dashscope_key
OPENAI_API_KEY=your_openai_key
```


### 核心功能

#### 增强流程 ⭐推荐

1. 数据准备

从原始工具数据中提取和精炼函数定义，包括函数名、描述和参数信息。
-输出：data\refined_tools.json

```bash
python code/PrepareData.py
```

2. 构建增强版函数图

在基础版基础上增加三重验证机制：
- **类型匹配验证**：推断输出类型，验证类型兼容性（解决 JSON vs String 问题）
- **逻辑兼容性检查**：黑白名单机制，过滤逻辑不通的连接（解决"时间→天气"问题）
- **工具价值评分**：识别高价值目标工具，为采样提供指导
- 显著提升边的质量和可执行性

-输出：data\function_graph_enhanced.json

```bash
python code/FunctionGraphEnhanced.py
```

3. 选择采样策略

**选项A：目标导向采样（适合顺序任务）**
目标导向的智能采样策略：
- **反向采样**：从高价值工具反向构建链，确保任务目标感（70%）
- **正向采样**：探索性采样，增加多样性（30%）
- **循环避免**：严格的visited集合，防止无效长链
- **质量保证**：确保每条链至少包含一个目标工具
- 工具链实用性从30%提升到85%+

```bash
python code/RandomWalkSamplingEnhanced.py
```

**选项B：聚类采样（适合并行+顺序任务）⭐推荐**

支持4种任务模式：
- 并行任务（30%）：查询天气+预订车票
- 顺序任务（30%）：查询→分析→报告
- 混合任务（20%）：并行+顺序组合
- 跨簇任务（20%）：多领域协作

-输出：data/sampled_tool_chains_cluster_zh.json

```bash
python code/ClusterBasedSampling.py
```

4. 生成场景

为每条工具链自动生成实际应用场景描述：
- 基于工具功能智能推断使用场景
- 支持通义千问/OpenAI API
- 生成简洁的场景标签（10-20字）

- 输出："data\tool_chains_with_scenarios_cluster.json"
**自动识别输入文件**

```bash
# 默认使用聚类采样结果
python code/GenerateScenarios.py
```

5. 去重和质量过滤 ⭐新增



**基于大模型的质量评估**：
- 使用 **qwen-turbo** 评估（不同于生成场景的qwen-plus）
- 避免自恋问题：评估模型与生成模型分离
- 多维度评分（0-40分）：
  - 场景质量（0-10）：清晰度、实用性、长度
  - 工具相关性（0-10）：工具间协同性
  - 工具完整性（0-10）：工具数量、完整度
  - 场景匹配度（0-10）：场景与工具的匹配
- 质量阈值：24分（平均6分）以上通过

**智能去重**：
- 基于工具集Jaccard相似度
- 基于场景描述语义相似度
- 综合判断（工具集85%相似 或 工具集70%+场景85%相似）

-输出：data\tool_chains_filtered_cluster.json

```bash
# 默认处理聚类场景数据
python code/FilterAndDeduplicate.py
```

6. 生成规划数据

- 多源奖励机制：LLM评分 + 结构奖励 + 效率奖励
- 自我修正：低分触发重新规划
- 偏好数据生成：用于DPO训练
- 规划模型与评价模型分离，避免自恋陷阱

-输出：data\plan_results\plan_preference_data_rl.json

```bash
# 优化版（快速）
python code/PlanAgentSystemOptimized.py

# 强化学习版（高质量）
python code/PlanAgentRL.py
```

### 数据流

```
原始工具数据 (JSONL)
    ↓ [PrepareData]
标准化工具定义 (500+ tools)
    → data/refined_tools.json
    ↓ [FunctionGraphEnhanced]
函数依赖图 (nodes + edges + metadata)
    → data/function_graph_enhanced.json
    ↓ [ClusterBasedSampling]
工具链集合 (1000 chains, 4 types)
    → data/sampled_tool_chains_cluster_zh.json
    ↓ [GenerateScenarios]
带场景工具链 (80 chains with scenarios)
    → data/tool_chains_with_scenarios_cluster.json
    ↓ [FilterAndDeduplicate]
高质量数据 (50-60 chains, score ≥ 24)
    → data/tool_chains_filtered_cluster.json
    ↓ [PlanAgentRL]
偏好训练数据 (preference pairs for DPO)
    → data/plan_results/plan_preference_data_rl.json
```



## 数据格式

### 工具链格式
```json
{
  "scenario": "社交媒体数据分析",
  "tools": {
    "get_instagram_posts": "获取Instagram帖子数据",
    "analyze_sentiment": "分析文本情感倾向"
  },
  "tools_length": 2
}
```

### 规划输出格式
```json
{
  "fixed_question": "标准化的用户问题",
  "thought": "规划思路说明",
  "steps": [
    {
      "thought": "步骤思考过程",
      "title": "步骤标题",
      "content": "步骤详细描述",
      "tools": ["工具名称"],
      "dependencies": ["前序步骤标题"]
    }
  ]
}
```

### 偏好数据格式
```json
{
  "user_question": "用户问题",
  "samples": [...],
  "best_plan": {...},
  "best_score": 42.5,
  "preference_pairs": [
    {
      "chosen": {...},
      "chosen_score": 42.5,
      "rejected": {...},
      "rejected_score": 35.2,
      "score_diff": 7.3
    }
  ]
}
```

## 评价维度

系统从5个维度评价规划质量（每项0-10分）：

1. **completeness（完整性）**：是否完整覆盖用户需求
2. **rationality（合理性）**：步骤分解是否合理、逻辑是否清晰
3. **tool_usage（工具使用）**：工具选择是否恰当
4. **dependencies（依赖关系）**：步骤间依赖是否正确
5. **executability（可执行性）**：规划是否可实际执行

强化学习版额外包含：
- **structure_reward（结构奖励）**：步骤数量、依赖正确性、标题清晰度
- **efficiency_reward（效率奖励）**：基于Token消耗的效率评估

## 项目结构

```
.
├── code/
│   ├── PrepareData.py                    # 数据准备
│   ├── FunctionGraph.py                  # 函数图构建（基础版）
│   ├── FunctionGraphEnhanced.py          # 函数图构建（增强版）⭐
│   ├── RandomWalkSampling.py             # 随机游走采样（基础版）
│   ├── RandomWalkSamplingEnhanced.py     # 随机游走采样（增强版）
│   ├── ClusterBasedSampling.py           # 聚类采样（支持并行任务）⭐
│   ├── GenerateScenarios.py              # 场景生成
│   ├── FilterAndDeduplicate.py           # 去重和质量过滤⭐
│   ├── PlanAgentSystemOptimized.py       # 规划系统（优化版）
│   ├── PlanAgentRL.py                    # 规划系统（强化学习版）
│   └── .env                              # API密钥配置
├── data/
│   ├── ToolACE-query.jsonl               # 原始工具数据
│   ├── refined_tools.json                # 精炼后的工具
│   ├── function_graph.json               # 函数依赖图（基础版）
│   ├── function_graph_enhanced.json      # 函数依赖图（增强版）⭐
│   ├── sampled_tool_chains_en.json       # 工具链-英文（基础版）
│   ├── sampled_tool_chains_zh.json       # 工具链-中文（基础版）
│   ├── sampled_tool_chains_enhanced_en.json  # 工具链-英文（增强版）
│   ├── sampled_tool_chains_enhanced_zh.json  # 工具链-中文（增强版）
│   ├── sampled_tool_chains_cluster_en.json   # 工具链-英文（聚类版）⭐
│   ├── sampled_tool_chains_cluster_zh.json   # 工具链-中文（聚类版）⭐
│   ├── tool_chains_with_scenarios_cluster.json  # 带场景的工具链
│   ├── tool_chains_filtered_cluster.json      # 过滤后的高质量数据⭐
│   └── plan_results/                     # 规划结果
│       └── plan_preference_data*.json
├── docs/
│   ├── QUICKSTART.md                     # 快速开始指南
│   ├── OPTIMIZATION.md                   # 优化方案详细说明
│   └── SUMMARY.md                        # 优化总结
└── README.md
```

## 技术特点

- **语义理解**：使用Sentence-BERT进行工具间语义相似度计算
- **图结构**：基于有向图建模工具依赖关系
- **类型系统**：自动推断和验证工具输入输出类型兼容性 ⭐
- **逻辑验证**：黑白名单机制过滤逻辑不合理的工具连接 ⭐
- **目标导向**：反向采样确保工具链具有明确的任务目标 ⭐
- **随机游走**：生成多样化的工具链组合
- **多模型支持**：兼容通义千问和OpenAI API
- **并发优化**：使用线程池提升处理速度
- **自我修正**：低分规划自动重新生成
- **偏好学习**：生成用于DPO训练的偏好数据

## 核心优化

### 问题1：逻辑断层
**原问题**：语义相似≠逻辑可用（如"时间"和"天气"语义相关，但不能直接连接）

**解决方案**：LogicValidator 逻辑验证器
- 维护逻辑兼容/不兼容的关键词对（黑白名单）
- 自动检查输出-输入的逻辑兼容性
- 调整边的置信度（+0.2 或 -0.3）

### 问题2：搜索空间冗余
**原问题**：随机游走容易产生循环或无效长链，缺乏任务目标感

**解决方案**：GoalOrientedSampler 目标导向采样器
- 识别高价值工具（analyze, report, generate等）
- 70%反向采样：从目标工具反向构建链
- 30%正向采样：探索性采样增加多样性
- 严格避免循环，确保链的实用性

### 问题3：类型不匹配
**原问题**：Sentence-BERT无法处理数据格式要求（JSON vs String）

**解决方案**：TypeMatcher 类型匹配器
- 基于关键词推断工具输出类型
- 验证输出类型与输入类型的兼容性
- 支持类型转换规则（object→string等）

详细优化方案请查看：[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)

## 注意事项

1. 首次运行会下载Sentence-BERT模型（约80MB）
2. 建议使用GPU加速函数图构建
3. API调用有频率限制，建议设置合理的延迟
4. 强化学习版处理速度较慢但质量更高
5. 偏好数据可用于后续的模型微调（DPO/RLHF）
6. **推荐使用增强版**（FunctionGraphEnhanced + RandomWalkSamplingEnhanced）获得更高质量的工具链

## 版本对比

| 特性 | 基础版 | 增强版 |
|------|--------|--------|
| 语义相似度计算 | ✓ | ✓ |
| 类型兼容性验证 | ✗ | ✓ |
| 逻辑合理性检查 | ✗ | ✓ |
| 工具价值评分 | ✗ | ✓ |
| 目标导向采样 | ✗ | ✓ |
| 循环避免 | 简单 | 严格 |
| 任务目标感 | 弱（~30%） | 强（~85%） |
| 推荐使用 | 快速原型 | 生产环境 ⭐ |

## 文档

- 📖 [快速开始指南](docs/QUICKSTART.md) - 从零开始的完整教程
- 🔧 [优化方案详解](docs/OPTIMIZATION.md) - 三大问题的详细解决方案
- 📊 [优化总结](docs/SUMMARY.md) - 效果对比和使用建议

## 许可证

MIT License
