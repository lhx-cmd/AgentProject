"""
场景去重和质量过滤（基于大模型）

功能：
1. 去重：基于工具集相似度去重
2. 质量过滤：使用大模型评估场景和工具集质量
   - 使用不同的模型（qwen-turbo）避免自恋问题
   - 场景描述的合理性和实用性
   - 工具集的相关性和完整性
   - 工具与场景的匹配度
"""

import json
import os
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import time

load_dotenv('code/.env')

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

class LLMQualityEvaluator:
    """
    基于大模型的质量评估器
    
    使用不同的模型（qwen-turbo）避免自恋问题
    """
    
    def __init__(self, use_dashscope=True):
        self.use_dashscope = use_dashscope
        self.client = None
        
        if use_dashscope:
            try:
                import dashscope
                dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
                self.dashscope = dashscope
                # 使用不同的模型：qwen-turbo（快速且不同于生成场景的qwen-plus）
                self.model_name = 'deepseek-v3.2'
                print(f"✓ 使用阿里云通义千问评估质量 (模型: {self.model_name})")
            except ImportError:
                print("警告：未安装 dashscope 库")
                self.use_dashscope = False
        
        if not self.use_dashscope:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                # 使用不同的模型：gpt-4o-mini（不同于生成场景的模型）
                self.model_name = 'gpt-4o-mini'
                print(f"✓ 使用OpenAI评估质量 (模型: {self.model_name})")
            except Exception as e:
                print(f"初始化失败: {e}")
    
    def call_llm(self, messages, temperature=0.3):
        """调用LLM"""
        try:
            if self.use_dashscope:
                from dashscope import Generation
                
                response = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    result_format='message',
                    temperature=temperature
                )
                
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
                else:
                    return None
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None
    
    def evaluate_quality(self, scenario, tools_dict):
        """
        评估场景和工具集的质量
        
        Returns:
            {
                'is_valid': bool,
                'score': float (0-10),
                'reason': str,
                'details': {
                    'scenario_quality': float,
                    'tools_relevance': float,
                    'tools_completeness': float,
                    'scenario_tools_match': float
                }
            }
        """
        # 构建工具列表
        tools_list = []
        for tool_name, description in list(tools_dict.items())[:10]:  # 最多显示10个
            tools_list.append(f"- {tool_name}: {description[:60]}...")
        
        tools_text = "\n".join(tools_list)
        if len(tools_dict) > 10:
            tools_text += f"\n... (共{len(tools_dict)}个工具)"
        
        # 构建评估提示词
        prompt = f"""请评估以下场景和工具集的质量。

场景描述：{scenario}

工具集（共{len(tools_dict)}个）：
{tools_text}

请从以下4个维度评分（每项0-10分）：

1. scenario_quality（场景质量）：
   - 场景描述是否清晰、具体、有实际应用价值？
   - 是否避免了技术术语（如"并行"、"顺序"、"混合"等）？
   - 长度是否合适（8-15字为佳）？

2. tools_relevance（工具相关性）：
   - 工具之间是否相关，能否协同完成某个任务？
   - 是否存在明显不相关的工具？

3. tools_completeness（工具完整性）：
   - 工具集是否完整，能否支撑场景的实现？
   - 工具数量是否合理（2-10个为佳）？
   - 是否缺少关键工具？

4. scenario_tools_match（场景与工具匹配度）：
   - 场景描述与工具集是否匹配？
   - 工具集能否实现场景描述的功能？

评分标准：
- 9-10分：优秀，几乎完美
- 7-8分：良好，有小瑕疵
- 5-6分：及格，有明显问题
- 3-4分：较差，问题较多
- 0-2分：很差，严重问题

请严格按照以下JSON格式返回（不要有其他内容）：
{{
  "scenario_quality": 8,
  "tools_relevance": 9,
  "tools_completeness": 7,
  "scenario_tools_match": 8,
  "total_score": 32,
  "is_valid": true,
  "reason": "简短的评价理由（一句话）"
}}

注意：
1. total_score是4个维度的总和（0-40分）
2. is_valid: 如果total_score >= 24（平均6分），则为true，否则为false
3. 请客观评分，不要所有场景都给相同分数"""
        
        messages = [
            {"role": "system", "content": "你是一个严格、客观的数据质量评审专家。请根据实际质量给出差异化的评分。"},
            {"role": "user", "content": prompt}
        ]
        
        result = self.call_llm(messages, temperature=0.3)
        
        if not result:
            return None
        
        try:
            # 提取JSON
            if '```json' in result:
                result = result.split('```json')[1].split('```')[0].strip()
            elif '```' in result:
                result = result.split('```')[1].split('```')[0].strip()
            
            evaluation = json.loads(result)
            
            # 验证必要字段
            required_fields = ['scenario_quality', 'tools_relevance', 'tools_completeness', 
                             'scenario_tools_match', 'total_score', 'is_valid', 'reason']
            
            for field in required_fields:
                if field not in evaluation:
                    print(f"警告：评估结果缺少字段 {field}")
                    return None
            
            # 构建返回结果
            return {
                'is_valid': evaluation['is_valid'],
                'score': evaluation['total_score'],
                'reason': evaluation['reason'],
                'details': {
                    'scenario_quality': evaluation['scenario_quality'],
                    'tools_relevance': evaluation['tools_relevance'],
                    'tools_completeness': evaluation['tools_completeness'],
                    'scenario_tools_match': evaluation['scenario_tools_match']
                }
            }
            
        except Exception as e:
            print(f"解析评估结果失败: {e}")
            print(f"原始结果: {result[:200]}...")
            return None

class ScenarioDeduplicator:
    """场景去重器"""
    
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
    
    def calculate_tools_similarity(self, tools1, tools2):
        """
        计算两个工具集的相似度（基于Jaccard相似度）
        
        Returns:
            similarity (0-1)
        """
        set1 = set(tools1.keys())
        set2 = set(tools2.keys())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_scenario_similarity(self, scenario1, scenario2):
        """
        计算两个场景描述的语义相似度
        
        Returns:
            similarity (0-1)
        """
        # 使用Sentence-BERT计算语义相似度
        embeddings = model.encode([scenario1, scenario2], convert_to_tensor=True)
        similarity = cosine_similarity(
            embeddings[0].cpu().numpy().reshape(1, -1),
            embeddings[1].cpu().numpy().reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def is_duplicate(self, chain1, chain2):
        """
        判断两个场景是否重复
        
        考虑：
        1. 工具集相似度
        2. 场景描述相似度
        """
        # 1. 工具集相似度
        tools_sim = self.calculate_tools_similarity(chain1['tools'], chain2['tools'])
        
        # 如果工具集完全相同，直接判定为重复
        if tools_sim >= 0.95:
            return True, f"工具集几乎相同 ({tools_sim:.2f})"
        
        # 2. 场景描述相似度
        scenario_sim = self.calculate_scenario_similarity(
            chain1['scenario'], 
            chain2['scenario']
        )
        
        # 如果工具集和场景都很相似，判定为重复
        if tools_sim >= 0.7 and scenario_sim >= 0.85:
            return True, f"工具集和场景都相似 (tools={tools_sim:.2f}, scenario={scenario_sim:.2f})"
        
        # 如果工具集相似度很高
        if tools_sim >= self.similarity_threshold:
            return True, f"工具集相似度过高 ({tools_sim:.2f})"
        
        return False, "不重复"
    
    def deduplicate(self, chains):
        """
        去重工具链
        
        策略：保留第一个出现的，删除后续重复的
        """
        unique_chains = []
        duplicate_count = 0
        
        print("\n开始去重...")
        for i, chain in enumerate(tqdm(chains, desc="去重进度")):
            is_dup = False
            
            for existing_chain in unique_chains:
                is_duplicate, reason = self.is_duplicate(chain, existing_chain)
                if is_duplicate:
                    is_dup = True
                    duplicate_count += 1
                    break
            
            if not is_dup:
                unique_chains.append(chain)
        
        print(f"✓ 去重完成：保留 {len(unique_chains)} 条，删除 {duplicate_count} 条重复")
        return unique_chains

def filter_and_deduplicate(input_file, output_file, similarity_threshold=0.85, 
                           quality_threshold=24, batch_size=5):
    """
    过滤和去重场景数据
    
    Args:
        input_file: 输入文件
        output_file: 输出文件
        similarity_threshold: 相似度阈值（0-1）
        quality_threshold: 质量阈值（0-40）
        batch_size: 批量评估大小（避免API限流）
    """
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    
    print(f"已加载 {len(chains)} 条场景数据")
    
    # 1. 质量过滤（使用大模型）
    print("\n" + "="*60)
    print("步骤1：质量过滤（基于大模型评估）")
    print("="*60)
    
    evaluator = LLMQualityEvaluator(use_dashscope=True)
    
    filtered_chains = []
    filter_stats = {
        'total': len(chains),
        'passed': 0,
        'failed': 0,
        'scores': [],
        'reasons': Counter()
    }
    
    print(f"\n开始评估（批量大小: {batch_size}）...")
    
    for i, chain in enumerate(tqdm(chains, desc="质量评估")):
        # 评估质量
        evaluation = evaluator.evaluate_quality(chain['scenario'], chain['tools'])
        
        if not evaluation:
            # 评估失败，跳过
            filter_stats['failed'] += 1
            filter_stats['reasons']['评估失败'] += 1
            continue
        
        # 记录评分
        filter_stats['scores'].append(evaluation['score'])
        
        # 判断是否通过
        if evaluation['is_valid'] and evaluation['score'] >= quality_threshold:
            # 保存评估结果
            chain['quality_evaluation'] = evaluation
            filtered_chains.append(chain)
            filter_stats['passed'] += 1
        else:
            filter_stats['failed'] += 1
            filter_stats['reasons'][evaluation['reason']] += 1
        
        # 避免API限流
        if (i + 1) % batch_size == 0:
            time.sleep(1)
    
    print(f"\n质量过滤结果:")
    print(f"  总数: {filter_stats['total']}")
    print(f"  通过: {filter_stats['passed']} ({filter_stats['passed']/filter_stats['total']*100:.1f}%)")
    print(f"  未通过: {filter_stats['failed']}")
    
    if filter_stats['scores']:
        print(f"\n评分统计:")
        print(f"  平均分: {np.mean(filter_stats['scores']):.2f}")
        print(f"  最高分: {max(filter_stats['scores']):.2f}")
        print(f"  最低分: {min(filter_stats['scores']):.2f}")
        print(f"  标准差: {np.std(filter_stats['scores']):.2f}")
    
    if filter_stats['reasons']:
        print(f"\n未通过原因统计 (Top 10):")
        for reason, count in filter_stats['reasons'].most_common(10):
            print(f"  {reason}: {count} 条")
    
    # 2. 去重
    print("\n" + "="*60)
    print("步骤2：去重")
    print("="*60)
    
    deduplicator = ScenarioDeduplicator(similarity_threshold=similarity_threshold)
    unique_chains = deduplicator.deduplicate(filtered_chains)
    
    # 3. 统计分析
    print("\n" + "="*60)
    print("最终统计")
    print("="*60)
    
    # 工具数量分布
    tools_count_dist = Counter([chain['tools_count'] for chain in unique_chains])
    print(f"\n工具数量分布:")
    for count in sorted(tools_count_dist.keys()):
        num = tools_count_dist[count]
        print(f"  {count} 个工具: {num} 条 ({num/len(unique_chains)*100:.1f}%)")
    
    # 评分分布
    if unique_chains:
        scores = [chain['quality_evaluation']['score'] for chain in unique_chains]
        print(f"\n质量评分分布:")
        print(f"  平均分: {np.mean(scores):.2f}")
        print(f"  中位数: {np.median(scores):.2f}")
        print(f"  最高分: {max(scores):.2f}")
        print(f"  最低分: {min(scores):.2f}")
    
    # 4. 保存结果
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_chains, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 完成！")
    print(f"  原始数据: {len(chains)} 条")
    print(f"  质量过滤后: {len(filtered_chains)} 条")
    print(f"  去重后: {len(unique_chains)} 条")
    print(f"  保留率: {len(unique_chains)/len(chains)*100:.1f}%")
    print(f"  结果已保存至: {output_file}")
    
    # 5. 显示示例
    print("\n" + "="*60)
    print("高质量场景示例:")
    print("="*60)
    
    # 按评分排序，显示前5个
    sorted_chains = sorted(unique_chains, 
                          key=lambda x: x['quality_evaluation']['score'], 
                          reverse=True)
    
    for i, example in enumerate(sorted_chains[:5], 1):
        eval_data = example['quality_evaluation']
        print(f"\n示例 {i}:")
        print(f"场景: {example['scenario']}")
        print(f"工具数量: {example['tools_count']}")
        print(f"质量评分: {eval_data['score']}/40")
        print(f"  - 场景质量: {eval_data['details']['scenario_quality']}/10")
        print(f"  - 工具相关性: {eval_data['details']['tools_relevance']}/10")
        print(f"  - 工具完整性: {eval_data['details']['tools_completeness']}/10")
        print(f"  - 场景匹配度: {eval_data['details']['scenario_tools_match']}/10")
        print(f"评价: {eval_data['reason']}")
        print(f"工具列表: {', '.join(list(example['tools'].keys())[:3])}...")

def main():
    """主函数"""
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'data/tool_chains_with_scenarios_cluster.json'
    
    # 生成输出文件名
    if 'cluster' in input_file:
        output_file = 'data/tool_chains_filtered_cluster.json'
    elif 'enhanced' in input_file:
        output_file = 'data/tool_chains_filtered_enhanced.json'
    else:
        output_file = 'data/tool_chains_filtered.json'
    
    print("="*60)
    print("场景去重和质量过滤（基于大模型）")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"评估模型: qwen-turbo (避免自恋)")
    print(f"相似度阈值: 0.85")
    print(f"质量阈值: 24/40 (平均6分)")
    
    filter_and_deduplicate(
        input_file, 
        output_file, 
        similarity_threshold=0.85,
        quality_threshold=24,
        batch_size=5
    )

if __name__ == '__main__':
    main()
