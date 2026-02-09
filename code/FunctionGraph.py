import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm  # 导入进度条库

# 自动检测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"当前使用设备: {device}")

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def build_function_graph(input_path, output_path, similarity_threshold=0.7):
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)

    num_tools = len(tools)
    graph_edges = []
    
    print(f"正在准备向量数据 (共 {num_tools} 个函数)...")
    
    # --- 优化策略：预计算所有向量，避免在循环中重复调用模型 ---
    # 1. 模拟所有 A 的输出描述并预计算向量
    a_outputs_texts = []
    for t in tools:
        # 模拟输出：函数名 + 描述前5个词
        text = f"{t['function_name']} {' '.join(t['description'].split()[:5])}"
        a_outputs_texts.append(text)
    
    print("正在计算输出向量...")
    output_embeddings = model.encode(a_outputs_texts, convert_to_tensor=True, show_progress_bar=True)

    # 2. 收集所有 B 的参数描述并预计算向量
    # 注意：一个函数可能有多个参数，我们记录每个参数属于哪个函数
    param_info_list = [] # 存储 (tool_index, param_name, embedding)
    all_param_texts = []
    
    for idx, t in enumerate(tools):
        for p in t['parameters']:
            all_param_texts.append(p['description'])
            param_info_list.append({
                "tool_idx": idx,
                "param_name": p['name']
            })
    
    print(f"正在计算参数向量 (共 {len(all_param_texts)} 个参数)...")
    param_embeddings = model.encode(all_param_texts, convert_to_tensor=True, show_progress_bar=True)

    # --- 开始相关性匹配 ---
    print(f"开始构建函数图 (预计进行 {num_tools} 轮匹配)...")
    
    # 外层循环使用 tqdm 显示进度
    for i in tqdm(range(num_tools), desc="构建进度"):
        emb_a = output_embeddings[i]
        
        # 遍历所有参数向量进行匹配
        # 计算当前工具 A 的输出与所有参数描述的相似度
        # 这里的 util.cos_sim 支持批量计算，速度极快
        cosine_scores = util.cos_sim(emb_a, param_embeddings)[0]
        
        # 找出超过阈值的索引
        relevant_indices = (cosine_scores > similarity_threshold).nonzero(as_tuple=True)[0]
        
        for rel_idx in relevant_indices:
            rel_idx = int(rel_idx)
            target_param = param_info_list[rel_idx]
            j = target_param['tool_idx']
            
            if i == j: continue  # 排除自环
            
            graph_edges.append({
                "from": tools[i]['function_name'],
                "to": tools[j]['function_name'],
                "via_parameter": target_param['param_name'],
                "confidence": float(cosine_scores[rel_idx])
            })

    # 保存图结构
    graph_data = {
        "nodes": [t['function_name'] for t in tools],
        "edges": graph_edges
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n图构建完成！共生成 {len(graph_edges)} 条有向边，已存至: {output_path}")

# --- 运行 ---
input_file = r'data\refined_tools.json'
graph_file = r'data\function_graph.json'

build_function_graph(input_file, graph_file)