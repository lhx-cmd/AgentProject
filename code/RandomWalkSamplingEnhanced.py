import json
import random
import os
from collections import defaultdict, deque
from tqdm import tqdm
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv('code/.env')

class GoalOrientedSampler:
    """
    目标导向的工具链采样器
    
    改进：
    1. 从高价值工具反向采样，确保任务目标感
    2. 避免循环和无效长链
    3. 基于类型和逻辑的路径验证
    """
    
    def __init__(self, graph_data):
        self.nodes = {n['name']: n for n in graph_data['nodes']}
        self.edges = graph_data['edges']
        
        # 构建邻接表（正向和反向）
        self.forward_adj = defaultdict(list)  # A -> B
        self.backward_adj = defaultdict(list)  # B -> A
        
        for edge in self.edges:
            from_node = edge['from']
            to_node = edge['to']
            
            self.forward_adj[from_node].append({
                'to': to_node,
                'confidence': edge['confidence'],
                'via_parameter': edge['via_parameter']
            })
            
            self.backward_adj[to_node].append({
                'from': from_node,
                'confidence': edge['confidence'],
                'via_parameter': edge['via_parameter']
            })
        
        # 识别高价值工具（目标工具）
        self.goal_tools = [
            name for name, node in self.nodes.items()
            if node.get('is_goal_tool', False)
        ]
        
        print(f"✓ 加载了 {len(self.nodes)} 个工具")
        print(f"✓ 加载了 {len(self.edges)} 条边")
        print(f"✓ 识别了 {len(self.goal_tools)} 个高价值目标工具")
    
    def backward_sample_chain(self, goal_tool, min_len=3, max_len=8):
        """
        从目标工具反向采样工具链
        
        策略：
        1. 从高价值工具开始
        2. 反向查找依赖
        3. 避免循环
        4. 确保逻辑连贯
        """
        chain = [goal_tool]
        visited = {goal_tool}
        
        current = goal_tool
        target_length = random.randint(min_len, max_len)
        
        while len(chain) < target_length:
            # 获取可以提供输入的前驱工具
            predecessors = self.backward_adj.get(current, [])
            
            if not predecessors:
                break
            
            # 过滤已访问的工具（避免循环）
            valid_predecessors = [
                p for p in predecessors
                if p['from'] not in visited
            ]
            
            if not valid_predecessors:
                break
            
            # 基于置信度加权选择
            weights = [p['confidence'] for p in valid_predecessors]
            chosen = random.choices(valid_predecessors, weights=weights, k=1)[0]
            
            prev_tool = chosen['from']
            chain.insert(0, prev_tool)  # 插入到链头
            visited.add(prev_tool)
            current = prev_tool
        
        return chain if len(chain) >= min_len else None
    
    def forward_sample_chain(self, start_tool, min_len=3, max_len=8):
        """
        从起始工具正向采样（传统方式，作为补充）
        
        改进：
        1. 优先选择通向高价值工具的路径
        2. 避免循环
        3. 限制低价值工具的连续出现
        """
        chain = [start_tool]
        visited = {start_tool}
        low_value_streak = 0  # 连续低价值工具计数
        
        current = start_tool
        target_length = random.randint(min_len, max_len)
        
        while len(chain) < target_length:
            successors = self.forward_adj.get(current, [])
            
            if not successors:
                break
            
            # 过滤已访问的工具
            valid_successors = [
                s for s in successors
                if s['to'] not in visited
            ]
            
            if not valid_successors:
                break
            
            # 计算选择权重（考虑置信度和工具价值）
            weights = []
            for s in valid_successors:
                next_tool = s['to']
                base_weight = s['confidence']
                
                # 提升高价值工具的权重
                if self.nodes[next_tool].get('is_goal_tool', False):
                    base_weight *= 1.5
                
                # 如果已经连续多个低价值工具，降低低价值工具权重
                if low_value_streak >= 2:
                    if not self.nodes[next_tool].get('is_goal_tool', False):
                        base_weight *= 0.5
                
                weights.append(base_weight)
            
            # 选择下一个工具
            chosen = random.choices(valid_successors, weights=weights, k=1)[0]
            next_tool = chosen['to']
            
            chain.append(next_tool)
            visited.add(next_tool)
            current = next_tool
            
            # 更新低价值工具计数
            if self.nodes[next_tool].get('is_goal_tool', False):
                low_value_streak = 0
            else:
                low_value_streak += 1
        
        # 验证链的质量
        if len(chain) >= min_len:
            # 检查是否至少有一个高价值工具
            has_goal = any(self.nodes[t].get('is_goal_tool', False) for t in chain)
            if has_goal:
                return chain
        
        return None
    
    def sample_diverse_chains(self, num_chains=1000, min_len=3, max_len=8, 
                             backward_ratio=0.7):
        """
        采样多样化的工具链
        
        Args:
            backward_ratio: 反向采样的比例（0-1）
        """
        chains = []
        
        num_backward = int(num_chains * backward_ratio)
        num_forward = num_chains - num_backward
        
        print(f"开始采样 {num_chains} 条工具链...")
        print(f"  - 反向采样（目标导向）: {num_backward} 条")
        print(f"  - 正向采样（探索性）: {num_forward} 条")
        
        # 1. 反向采样（从目标工具开始）
        for _ in tqdm(range(num_backward), desc="反向采样"):
            if not self.goal_tools:
                break
            
            goal_tool = random.choice(self.goal_tools)
            chain = self.backward_sample_chain(goal_tool, min_len, max_len)
            
            if chain:
                chains.append(chain)
        
        # 2. 正向采样（从任意工具开始）
        all_tools = list(self.nodes.keys())
        for _ in tqdm(range(num_forward), desc="正向采样"):
            start_tool = random.choice(all_tools)
            chain = self.forward_sample_chain(start_tool, min_len, max_len)
            
            if chain:
                chains.append(chain)
        
        print(f"✓ 成功采样 {len(chains)} 条有效工具链")
        return chains
    
    def deduplicate_chains(self, chains):
        """去重工具链"""
        unique_chains = []
        seen = set()
        
        for chain in chains:
            chain_key = tuple(chain)
            if chain_key not in seen:
                seen.add(chain_key)
                unique_chains.append(chain)
        
        print(f"✓ 去重后保留 {len(unique_chains)} 条唯一工具链")
        return unique_chains
    
    def analyze_chains(self, chains):
        """分析工具链质量"""
        stats = {
            'total': len(chains),
            'length_dist': defaultdict(int),
            'has_goal_tool': 0,
            'avg_length': 0,
            'goal_tool_usage': defaultdict(int)
        }
        
        total_length = 0
        for chain in chains:
            length = len(chain)
            stats['length_dist'][length] += 1
            total_length += length
            
            # 检查是否包含高价值工具
            has_goal = False
            for tool in chain:
                if self.nodes[tool].get('is_goal_tool', False):
                    has_goal = True
                    stats['goal_tool_usage'][tool] += 1
            
            if has_goal:
                stats['has_goal_tool'] += 1
        
        stats['avg_length'] = total_length / len(chains) if chains else 0
        
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

def enhanced_random_walk_sampling(graph_path, output_path_en, output_path_zh, 
                                  num_chains=1000, min_len=3, max_len=8):
    """
    增强版随机游走采样
    """
    if not os.path.exists(graph_path):
        print(f"错误：找不到图文件 {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 初始化采样器
    sampler = GoalOrientedSampler(graph_data)
    
    # 采样工具链
    chains = sampler.sample_diverse_chains(
        num_chains=num_chains,
        min_len=min_len,
        max_len=max_len,
        backward_ratio=0.7  # 70%反向采样，30%正向采样
    )
    
    # 去重
    chains = sampler.deduplicate_chains(chains)
    
    # 分析质量
    stats = sampler.analyze_chains(chains)
    
    print(f"\n{'='*60}")
    print("工具链质量分析:")
    print(f"{'='*60}")
    print(f"总数: {stats['total']}")
    print(f"平均长度: {stats['avg_length']:.2f}")
    print(f"包含目标工具: {stats['has_goal_tool']} ({stats['has_goal_tool']/stats['total']*100:.1f}%)")
    print(f"\n长度分布:")
    for length in sorted(stats['length_dist'].keys()):
        count = stats['length_dist'][length]
        print(f"  长度 {length}: {count} 条 ({count/stats['total']*100:.1f}%)")
    
    print(f"\n最常用的目标工具 (Top 10):")
    top_goals = sorted(stats['goal_tool_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
    for tool, count in top_goals:
        print(f"  {tool}: {count} 次")
    print(f"{'='*60}\n")
    
    # 构建输出数据（英文版）
    sampled_chains_en = []
    for chain in chains:
        tools_dict_en = {}
        for tool_name in chain:
            node = sampler.nodes[tool_name]
            tools_dict_en[tool_name] = node['description']
        
        sampled_chains_en.append({
            "length": len(chain),
            "tools": tools_dict_en,
            "has_goal_tool": any(sampler.nodes[t].get('is_goal_tool', False) for t in chain)
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
            "has_goal_tool": chain_en['has_goal_tool']
        })
    
    # 保存中文版
    with open(output_path_zh, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains_zh, f, indent=2, ensure_ascii=False)
    print(f"✓ 中文版已保存: {output_path_zh}")
    
    print(f"\n{'='*60}")
    print(f"✓ 完成！生成了 {len(chains)} 条高质量工具链")
    print(f"{'='*60}")

# --- 运行 ---
if __name__ == '__main__':
    graph_file = r'data/function_graph_enhanced.json'
    chains_file_en = r'data/sampled_tool_chains_enhanced_en.json'
    chains_file_zh = r'data/sampled_tool_chains_enhanced_zh.json'
    
    enhanced_random_walk_sampling(
        graph_file, 
        chains_file_en, 
        chains_file_zh,
        num_chains=1000,
        min_len=3,
        max_len=8
    )
