import json
import random
import os
from collections import defaultdict
from tqdm import tqdm

def random_walk_sampling(graph_path, output_path, num_chains=1000, min_len=4, max_len=10):
    if not os.path.exists(graph_path):
        print(f"错误：找不到图文件 {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    nodes = graph_data['nodes']
    edges = graph_data['edges']

    # 1. 构建邻接表以便快速查询下一步
    adj_list = defaultdict(list)
    for edge in edges:
        # 存储目标节点和置信度，置信度可作为转移概率的权重
        adj_list[edge['from']].append((edge['to'], edge['confidence']))

    sampled_chains = []
    
    print(f"开始随机游走采样，目标生成 {num_chains} 条工具链...")

    for _ in tqdm(range(num_chains), desc="采样进度"):
        # 随机选择一个起始节点
        current_node = random.choice(nodes)
        chain = [current_node]
        
        # 随机决定当前链的目标长度
        target_length = random.randint(min_len, max_len)
        
        while len(chain) < target_length:
            neighbors = adj_list.get(current_node, [])
            if not neighbors:
                break # 遇到死胡同，停止当前链的采样
            
            # 根据 ToolMind 逻辑：可以选择均匀分布，也可以根据 confidence 加权
            # 这里采用加权随机选择，让逻辑更相关的工具更大概率连在一起
            next_nodes = [n[0] for n in neighbors]
            weights = [n[1] for n in neighbors]
            
            next_node = random.choices(next_nodes, weights=weights, k=1)[0]
            
            # 避免简单的回环（可选：如果不需要重复调用同一个工具）
            if next_node in chain[-2:]: 
                # 尝试重新选一次或直接停止
                break
                
            chain.append(next_node)
            current_node = next_node
        
        # 只有达到最小长度的链才会被保留
        if len(chain) >= min_len:
            sampled_chains.append({
                "length": len(chain),
                "chain": chain
            })

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains, f, indent=4, ensure_ascii=False)
    
    print(f"\n采样完成！成功生成 {len(sampled_chains)} 条有效工具链。")
    print(f"结果已存至: {output_path}")

# --- 运行 ---
graph_file = r'data\function_graph.json'
chains_file = r'data\sampled_tool_chains.json'

random_walk_sampling(graph_file, chains_file)