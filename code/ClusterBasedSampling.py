"""
基于聚类的子图采样

解决问题：
- 随机游走只能生成线性依赖链（A→B→C）
- 无法生成并行任务（A + B，无依赖关系）
- 缺少场景相关性（如"旅行"场景下的多个独立工具）

解决方案：
1. 基于语义相似度对工具进行聚类
2. 从同一簇中采样多个工具（场景相关但无依赖）
3. 混合策略：聚类采样 + 随机游走
"""

import json
import os
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

load_dotenv('code/.env')

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

class ClusterBasedSampler:
    """
    基于聚类的工具采样器
    
    策略：
    1. 对工具进行语义聚类（场景相关性）
    2. 从同一簇采样多个工具（并行任务）
    3. 跨簇采样（依赖任务）
    """
    
    def __init__(self, graph_data, num_clusters=50):
        self.nodes = {n['name']: n for n in graph_data['nodes']}
        self.edges = graph_data['edges']
        self.num_clusters = num_clusters
        
        # 构建邻接表
        self.forward_adj = defaultdict(list)
        self.backward_adj = defaultdict(list)
        
        for edge in self.edges:
            from_node = edge['from']
            to_node = edge['to']
            
            self.forward_adj[from_node].append({
                'to': to_node,
                'confidence': edge['confidence']
            })
            
            self.backward_adj[to_node].append({
                'from': from_node,
                'confidence': edge['confidence']
            })
        
        print(f"✓ 加载了 {len(self.nodes)} 个工具")
        print(f"✓ 加载了 {len(self.edges)} 条边")
        
        # 执行聚类
        self.clusters = None
        self.tool_to_cluster = {}
        self.cluster_to_tools = defaultdict(list)
        self._perform_clustering()
    
    def _perform_clustering(self):
        """对工具进行语义聚类"""
        print(f"\n开始对工具进行语义聚类...")
        
        # 1. 准备工具描述
        tool_names = list(self.nodes.keys())
        tool_descriptions = [
            f"{name} {self.nodes[name]['description']}"
            for name in tool_names
        ]
        
        # 2. 计算嵌入向量
        print("  - 计算工具嵌入向量...")
        embeddings = model.encode(tool_descriptions, convert_to_tensor=True, show_progress_bar=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # 3. 使用KMeans聚类
        print(f"  - 执行KMeans聚类 (k={self.num_clusters})...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # 4. 构建聚类映射
        for tool_name, cluster_id in zip(tool_names, cluster_labels):
            self.tool_to_cluster[tool_name] = int(cluster_id)
            self.cluster_to_tools[int(cluster_id)].append(tool_name)
        
        # 5. 分析聚类质量
        cluster_sizes = [len(tools) for tools in self.cluster_to_tools.values()]
        
        print(f"\n✓ 聚类完成！")
        print(f"  - 簇数量: {len(self.cluster_to_tools)}")
        print(f"  - 平均簇大小: {np.mean(cluster_sizes):.1f}")
        print(f"  - 最大簇大小: {max(cluster_sizes)}")
        print(f"  - 最小簇大小: {min(cluster_sizes)}")
        
        # 6. 显示示例簇
        print(f"\n示例簇（前3个）:")
        for cluster_id in sorted(self.cluster_to_tools.keys())[:3]:
            tools = self.cluster_to_tools[cluster_id][:5]
            print(f"  簇 {cluster_id}: {', '.join(tools[:3])}...")
    
    def sample_parallel_chain(self, min_tools=2, max_tools=4):
        """
        采样并行任务链（无依赖关系）
        
        策略：从同一簇中采样多个工具
        """
        # 选择一个簇
        cluster_id = random.choice(list(self.cluster_to_tools.keys()))
        tools_in_cluster = self.cluster_to_tools[cluster_id]
        
        # 过滤掉没有参数的工具（更有意义）
        valid_tools = [
            t for t in tools_in_cluster
            if self.nodes[t].get('value_score', 0.5) >= 0.4  # 至少中等价值
        ]
        
        if len(valid_tools) < min_tools:
            return None
        
        # 随机采样
        num_tools = random.randint(min_tools, min(max_tools, len(valid_tools)))
        selected_tools = random.sample(valid_tools, num_tools)
        
        return {
            'tools': selected_tools,
            'type': 'parallel',
            'cluster_id': cluster_id,
            'dependencies': None  # 并行任务无依赖
        }
    
    def sample_sequential_chain(self, min_len=2, max_len=5):
        """
        采样顺序任务链（有依赖关系）
        
        策略：随机游走，可能跨簇
        """
        # 选择起始工具
        start_tool = random.choice(list(self.nodes.keys()))
        chain = [start_tool]
        visited = {start_tool}
        
        target_length = random.randint(min_len, max_len)
        
        while len(chain) < target_length:
            current = chain[-1]
            successors = self.forward_adj.get(current, [])
            
            if not successors:
                break
            
            # 过滤已访问
            valid_successors = [
                s for s in successors
                if s['to'] not in visited
            ]
            
            if not valid_successors:
                break
            
            # 加权选择
            weights = [s['confidence'] for s in valid_successors]
            chosen = random.choices(valid_successors, weights=weights, k=1)[0]
            
            next_tool = chosen['to']
            chain.append(next_tool)
            visited.add(next_tool)
        
        if len(chain) < min_len:
            return None
        
        return {
            'tools': chain,
            'type': 'sequential',
            'dependencies': 'linear'  # 线性依赖
        }
    
    def sample_hybrid_chain(self, num_parallel=2, num_sequential=1):
        """
        采样混合任务链（并行 + 顺序）
        
        策略：
        1. 先采样一些并行任务（同簇）
        2. 再采样一个顺序链（可能跨簇）
        """
        components = []
        
        # 1. 采样并行任务
        for _ in range(num_parallel):
            parallel = self.sample_parallel_chain(min_tools=2, max_tools=3)
            if parallel:
                components.append(parallel)
        
        # 2. 采样顺序链
        for _ in range(num_sequential):
            sequential = self.sample_sequential_chain(min_len=2, max_len=4)
            if sequential:
                components.append(sequential)
        
        if not components:
            return None
        
        # 合并所有工具
        all_tools = []
        for comp in components:
            all_tools.extend(comp['tools'])
        
        # 去重
        all_tools = list(dict.fromkeys(all_tools))
        
        return {
            'tools': all_tools,
            'type': 'hybrid',
            'components': components
        }
    
    def sample_cross_cluster_chain(self, num_clusters=2, tools_per_cluster=2):
        """
        采样跨簇任务链
        
        策略：从多个不同的簇中各采样一些工具
        适用于：需要多个领域协作的复杂任务
        """
        # 选择多个簇
        available_clusters = list(self.cluster_to_tools.keys())
        if len(available_clusters) < num_clusters:
            return None
        
        selected_clusters = random.sample(available_clusters, num_clusters)
        
        all_tools = []
        for cluster_id in selected_clusters:
            tools_in_cluster = self.cluster_to_tools[cluster_id]
            
            # 从每个簇采样工具
            num_to_sample = min(tools_per_cluster, len(tools_in_cluster))
            sampled = random.sample(tools_in_cluster, num_to_sample)
            all_tools.extend(sampled)
        
        if len(all_tools) < 2:
            return None
        
        return {
            'tools': all_tools,
            'type': 'cross_cluster',
            'clusters': selected_clusters,
            'dependencies': 'mixed'  # 混合依赖
        }
    
    def sample_diverse_chains(self, num_chains=1000, strategy_weights=None):
        """
        采样多样化的工具链
        
        Args:
            strategy_weights: 各策略的权重
                {
                    'parallel': 0.3,      # 并行任务
                    'sequential': 0.3,    # 顺序任务
                    'hybrid': 0.2,        # 混合任务
                    'cross_cluster': 0.2  # 跨簇任务
                }
        """
        if strategy_weights is None:
            strategy_weights = {
                'parallel': 0.3,
                'sequential': 0.3,
                'hybrid': 0.2,
                'cross_cluster': 0.2
            }
        
        strategies = list(strategy_weights.keys())
        weights = list(strategy_weights.values())
        
        chains = []
        
        print(f"\n开始采样 {num_chains} 条工具链...")
        print(f"策略分布: {strategy_weights}")
        
        for _ in tqdm(range(num_chains), desc="采样进度"):
            # 随机选择策略
            strategy = random.choices(strategies, weights=weights, k=1)[0]
            
            chain = None
            if strategy == 'parallel':
                chain = self.sample_parallel_chain(min_tools=2, max_tools=4)
            elif strategy == 'sequential':
                chain = self.sample_sequential_chain(min_len=2, max_len=5)
            elif strategy == 'hybrid':
                chain = self.sample_hybrid_chain(num_parallel=1, num_sequential=1)
            elif strategy == 'cross_cluster':
                chain = self.sample_cross_cluster_chain(num_clusters=2, tools_per_cluster=2)
            
            if chain:
                chains.append(chain)
        
        print(f"✓ 成功采样 {len(chains)} 条工具链")
        return chains
    
    def analyze_chains(self, chains):
        """分析工具链质量"""
        stats = {
            'total': len(chains),
            'type_dist': defaultdict(int),
            'length_dist': defaultdict(int),
            'avg_length': 0,
            'parallel_ratio': 0,
            'sequential_ratio': 0,
            'hybrid_ratio': 0,
            'cross_cluster_ratio': 0
        }
        
        total_length = 0
        for chain in chains:
            chain_type = chain.get('type', 'unknown')
            stats['type_dist'][chain_type] += 1
            
            length = len(chain['tools'])
            stats['length_dist'][length] += 1
            total_length += length
        
        if chains:
            stats['avg_length'] = total_length / len(chains)
            stats['parallel_ratio'] = stats['type_dist']['parallel'] / len(chains)
            stats['sequential_ratio'] = stats['type_dist']['sequential'] / len(chains)
            stats['hybrid_ratio'] = stats['type_dist']['hybrid'] / len(chains)
            stats['cross_cluster_ratio'] = stats['type_dist']['cross_cluster'] / len(chains)
        
        return stats

def translate_batch_to_chinese_dashscope(descriptions_dict):
    """使用阿里云DashScope API批量翻译"""
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        print("警告：未安装 dashscope 库")
        return descriptions_dict
    
    dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
    
    def is_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    to_translate = {}
    already_chinese = {}
    
    for tool_name, desc in descriptions_dict.items():
        if is_chinese(desc):
            already_chinese[tool_name] = desc
        else:
            to_translate[tool_name] = desc
    
    if not to_translate:
        return already_chinese
    
    translated = {}
    batch_size = 30
    tool_names = list(to_translate.keys())
    
    print(f"使用通义千问翻译 {len(tool_names)} 个描述...")
    
    for i in tqdm(range(0, len(tool_names), batch_size), desc="翻译进度"):
        batch_tools = tool_names[i:i+batch_size]
        batch_descriptions = [to_translate[name] for name in batch_tools]
        
        prompt = "请将以下API工具描述翻译成简洁的中文，保持专业性和准确性。每行一个描述，按顺序翻译：\n\n"
        for idx, desc in enumerate(batch_descriptions, 1):
            prompt += f"{idx}. {desc}\n"
        
        prompt += "\n请按相同顺序返回翻译结果，每行一个翻译，格式为：序号. 翻译内容"
        
        try:
            response = Generation.call(
                model='qwen-plus',
                messages=[
                    {"role": "system", "content": "你是专业的技术翻译助手。"},
                    {"role": "user", "content": prompt}
                ],
                result_format='message',
                temperature=0.3
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                translations = content.strip().split('\n')
                translations = [t.strip() for t in translations if t.strip()]
                
                for j, tool_name in enumerate(batch_tools):
                    if j < len(translations):
                        translation = translations[j]
                        translation = translation.lstrip('0123456789.、 ')
                        translated[tool_name] = translation
                    else:
                        translated[tool_name] = to_translate[tool_name]
            else:
                for tool_name in batch_tools:
                    translated[tool_name] = to_translate[tool_name]
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"翻译批次出错: {e}")
            for tool_name in batch_tools:
                translated[tool_name] = to_translate[tool_name]
    
    result = {**already_chinese, **translated}
    return result

def cluster_based_sampling(graph_path, output_path_en, output_path_zh, 
                           num_chains=1000, num_clusters=50):
    """
    基于聚类的工具链采样
    """
    if not os.path.exists(graph_path):
        print(f"错误：找不到图文件 {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 初始化采样器
    sampler = ClusterBasedSampler(graph_data, num_clusters=num_clusters)
    
    # 采样工具链
    chains = sampler.sample_diverse_chains(
        num_chains=num_chains,
        strategy_weights={
            'parallel': 0.3,      # 30% 并行任务（无依赖）
            'sequential': 0.3,    # 30% 顺序任务（线性依赖）
            'hybrid': 0.2,        # 20% 混合任务
            'cross_cluster': 0.2  # 20% 跨簇任务
        }
    )
    
    # 分析质量
    stats = sampler.analyze_chains(chains)
    
    print(f"\n{'='*60}")
    print("工具链质量分析:")
    print(f"{'='*60}")
    print(f"总数: {stats['total']}")
    print(f"平均长度: {stats['avg_length']:.2f}")
    print(f"\n类型分布:")
    print(f"  并行任务: {stats['type_dist']['parallel']} ({stats['parallel_ratio']*100:.1f}%)")
    print(f"  顺序任务: {stats['type_dist']['sequential']} ({stats['sequential_ratio']*100:.1f}%)")
    print(f"  混合任务: {stats['type_dist']['hybrid']} ({stats['hybrid_ratio']*100:.1f}%)")
    print(f"  跨簇任务: {stats['type_dist']['cross_cluster']} ({stats['cross_cluster_ratio']*100:.1f}%)")
    
    print(f"\n长度分布:")
    for length in sorted(stats['length_dist'].keys()):
        count = stats['length_dist'][length]
        print(f"  长度 {length}: {count} 条 ({count/stats['total']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # 构建输出数据（英文版）
    sampled_chains_en = []
    for chain in chains:
        tools_dict_en = {}
        for tool_name in chain['tools']:
            node = sampler.nodes[tool_name]
            tools_dict_en[tool_name] = node['description']
        
        sampled_chains_en.append({
            "length": len(chain['tools']),
            "tools": tools_dict_en,
            "type": chain['type'],
            "dependencies": chain.get('dependencies'),
            "metadata": {
                k: v for k, v in chain.items() 
                if k not in ['tools', 'type', 'dependencies']
            }
        })
    
    # 保存英文版
    with open(output_path_en, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains_en, f, indent=2, ensure_ascii=False)
    print(f"✓ 英文版已保存: {output_path_en}")
    
    # 翻译成中文
    print("\n开始翻译到中文...")
    all_descriptions = {}
    for chain in sampled_chains_en:
        all_descriptions.update(chain['tools'])
    
    unique_descriptions = {k: v for k, v in all_descriptions.items()}
    
    tool_descriptions_zh = None
    if os.getenv('DASHSCOPE_API_KEY'):
        tool_descriptions_zh = translate_batch_to_chinese_dashscope(unique_descriptions)
    
    if tool_descriptions_zh is None:
        print("警告：翻译失败，使用英文原文")
        tool_descriptions_zh = unique_descriptions
    
    # 构建中文版
    sampled_chains_zh = []
    for chain_en in sampled_chains_en:
        tools_dict_zh = {}
        for tool_name in chain_en['tools'].keys():
            tools_dict_zh[tool_name] = tool_descriptions_zh.get(
                tool_name, 
                chain_en['tools'][tool_name]
            )
        
        sampled_chains_zh.append({
            "length": chain_en['length'],
            "tools": tools_dict_zh,
            "type": chain_en['type'],
            "dependencies": chain_en['dependencies'],
            "metadata": chain_en.get('metadata', {})
        })
    
    # 保存中文版
    with open(output_path_zh, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains_zh, f, indent=2, ensure_ascii=False)
    print(f"✓ 中文版已保存: {output_path_zh}")
    
    print(f"\n{'='*60}")
    print(f"✓ 完成！生成了 {len(chains)} 条多样化工具链")
    print(f"  - 支持并行任务（无依赖）")
    print(f"  - 支持顺序任务（线性依赖）")
    print(f"  - 支持混合任务（并行+顺序）")
    print(f"  - 支持跨簇任务（多领域协作）")
    print(f"{'='*60}")

# --- 运行 ---
if __name__ == '__main__':
    graph_file = r'data/function_graph_enhanced.json'
    chains_file_en = r'data/sampled_tool_chains_cluster_en.json'
    chains_file_zh = r'data/sampled_tool_chains_cluster_zh.json'
    
    cluster_based_sampling(
        graph_file, 
        chains_file_en, 
        chains_file_zh,
        num_chains=1000,
        num_clusters=50
    )
