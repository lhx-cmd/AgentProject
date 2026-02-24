import json
import random
import os
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv('code/.env')

def translate_batch_to_chinese_dashscope(descriptions_dict):
    """
    使用阿里云DashScope API（通义千问）批量翻译描述到中文
    
    Args:
        descriptions_dict: {tool_name: english_description}
    
    Returns:
        {tool_name: chinese_description}
    """
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        print("警告：未安装 dashscope 库，请运行: pip install dashscope")
        return descriptions_dict
    
    # 设置API key
    dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
    
    # 检查是否已经是中文
    def is_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # 分离需要翻译和不需要翻译的
    to_translate = {}
    already_chinese = {}
    
    for tool_name, desc in descriptions_dict.items():
        if is_chinese(desc):
            already_chinese[tool_name] = desc
        else:
            to_translate[tool_name] = desc
    
    if not to_translate:
        return already_chinese
    
    # 批量翻译（每次最多30个）
    translated = {}
    batch_size = 30
    tool_names = list(to_translate.keys())
    
    print(f"使用通义千问翻译 {len(tool_names)} 个描述...")
    
    for i in tqdm(range(0, len(tool_names), batch_size), desc="翻译进度"):
        batch_tools = tool_names[i:i+batch_size]
        batch_descriptions = [to_translate[name] for name in batch_tools]
        
        # 构建翻译提示
        prompt = "请将以下API工具描述翻译成简洁的中文，保持专业性和准确性。每行一个描述，按顺序翻译：\n\n"
        for idx, desc in enumerate(batch_descriptions, 1):
            prompt += f"{idx}. {desc}\n"
        
        prompt += "\n请按相同顺序返回翻译结果，每行一个翻译，格式为：序号. 翻译内容"
        
        try:
            response = Generation.call(
                model='qwen-plus',
                messages=[
                    {"role": "system", "content": "你是一个专业的技术翻译助手，擅长将API和工具描述翻译成简洁准确的中文。"},
                    {"role": "user", "content": prompt}
                ],
                result_format='message',
                temperature=0.3
            )
            
            if response.status_code == 200:
                # 解析翻译结果
                content = response.output.choices[0].message.content
                translations = content.strip().split('\n')
                translations = [t.strip() for t in translations if t.strip()]
                
                # 匹配工具名和翻译
                for j, tool_name in enumerate(batch_tools):
                    if j < len(translations):
                        # 移除可能的序号前缀
                        translation = translations[j]
                        translation = translation.lstrip('0123456789.、 ')
                        translated[tool_name] = translation
                    else:
                        # 如果翻译结果不够，使用原文
                        translated[tool_name] = to_translate[tool_name]
            else:
                print(f"API调用失败: {response.code} - {response.message}")
                # 出错时使用原文
                for tool_name in batch_tools:
                    translated[tool_name] = to_translate[tool_name]
            
            # 避免API限流
            time.sleep(0.5)
            
        except Exception as e:
            print(f"翻译批次 {i//batch_size + 1} 时出错: {e}")
            # 出错时使用原文
            for tool_name in batch_tools:
                translated[tool_name] = to_translate[tool_name]
    
    # 合并结果
    result = {**already_chinese, **translated}
    return result

def translate_batch_to_chinese_openai(descriptions_dict, client):
    """
    使用OpenAI API批量翻译描述到中文
    
    Args:
        descriptions_dict: {tool_name: english_description}
        client: OpenAI client
    
    Returns:
        {tool_name: chinese_description}
    """
    # 检查是否已经是中文
    def is_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # 分离需要翻译和不需要翻译的
    to_translate = {}
    already_chinese = {}
    
    for tool_name, desc in descriptions_dict.items():
        if is_chinese(desc):
            already_chinese[tool_name] = desc
        else:
            to_translate[tool_name] = desc
    
    if not to_translate:
        return already_chinese
    
    # 批量翻译（每次最多50个）
    translated = {}
    batch_size = 50
    tool_names = list(to_translate.keys())
    
    for i in range(0, len(tool_names), batch_size):
        batch_tools = tool_names[i:i+batch_size]
        batch_descriptions = [to_translate[name] for name in batch_tools]
        
        # 构建翻译提示
        prompt = "请将以下API工具描述翻译成简洁的中文，保持专业性和准确性。每行一个描述，按顺序翻译：\n\n"
        for idx, desc in enumerate(batch_descriptions, 1):
            prompt += f"{idx}. {desc}\n"
        
        prompt += "\n请按相同顺序返回翻译结果，每行一个翻译，不要添加序号或其他内容。"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的技术翻译助手，擅长将API和工具描述翻译成简洁准确的中文。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # 解析翻译结果
            translations = response.choices[0].message.content.strip().split('\n')
            translations = [t.strip() for t in translations if t.strip()]
            
            # 匹配工具名和翻译
            for j, tool_name in enumerate(batch_tools):
                if j < len(translations):
                    # 移除可能的序号前缀
                    translation = translations[j]
                    translation = translation.lstrip('0123456789.、 ')
                    translated[tool_name] = translation
                else:
                    # 如果翻译结果不够，使用原文
                    translated[tool_name] = to_translate[tool_name]
            
            # 避免API限流
            time.sleep(0.5)
            
        except Exception as e:
            print(f"翻译批次 {i//batch_size + 1} 时出错: {e}")
            # 出错时使用原文
            for tool_name in batch_tools:
                translated[tool_name] = to_translate[tool_name]
    
    # 合并结果
    result = {**already_chinese, **translated}
    return result

def random_walk_sampling(graph_path, tools_path, output_path_en, output_path_zh, num_chains=1000, min_len=4, max_len=10):
    if not os.path.exists(graph_path):
        print(f"错误：找不到图文件 {graph_path}")
        return
    
    if not os.path.exists(tools_path):
        print(f"错误：找不到工具文件 {tools_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 加载工具描述信息
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    
    # 构建工具名称到描述的映射（英文原版）
    tool_descriptions_en = {}
    for tool in tools_data:
        tool_name = tool.get('function_name')
        description = tool.get('description', '')
        if tool_name:
            tool_descriptions_en[tool_name] = description
    
    print(f"已加载 {len(tool_descriptions_en)} 个工具描述")

    nodes = graph_data['nodes']
    edges = graph_data['edges']

    # 1. 构建邻接表以便快速查询下一步
    adj_list = defaultdict(list)
    for edge in edges:
        # 存储目标节点和置信度，置信度可作为转移概率的权重
        adj_list[edge['from']].append((edge['to'], edge['confidence']))

    sampled_chains_en = []  # 英文版
    sampled_chains_zh = []  # 中文版
    
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
            # 构建英文版 tools 字典
            tools_dict_en = {}
            for tool_name in chain:
                tools_dict_en[tool_name] = tool_descriptions_en.get(tool_name, "No description available")
            
            sampled_chains_en.append({
                "length": len(chain),
                "tools": tools_dict_en
            })
    
    print(f"\n采样完成！成功生成 {len(sampled_chains_en)} 条有效工具链。")
    
    # 保存英文版
    with open(output_path_en, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains_en, f, indent=2, ensure_ascii=False)
    print(f"英文版已存至: {output_path_en}")
    
    # 翻译成中文版
    print("\n开始翻译描述到中文...")
    
    # 收集所有需要翻译的描述
    all_descriptions = {}
    for chain in sampled_chains_en:
        all_descriptions.update(chain['tools'])
    
    # 去重
    unique_descriptions = {k: v for k, v in all_descriptions.items()}
    print(f"需要翻译 {len(unique_descriptions)} 个唯一描述...")
    
    # 尝试使用DashScope API（通义千问）
    tool_descriptions_zh = None
    
    if os.getenv('DASHSCOPE_API_KEY'):
        print("使用阿里云通义千问进行翻译...")
        try:
            tool_descriptions_zh = translate_batch_to_chinese_dashscope(unique_descriptions)
        except Exception as e:
            print(f"DashScope翻译失败: {e}")
    
    # 如果DashScope失败，尝试OpenAI
    if tool_descriptions_zh is None and os.getenv('OPENAI_API_KEY'):
        print("使用OpenAI进行翻译...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            tool_descriptions_zh = translate_batch_to_chinese_openai(unique_descriptions, client)
        except Exception as e:
            print(f"OpenAI翻译失败: {e}")
    
    # 如果都失败，使用原文
    if tool_descriptions_zh is None:
        print("警告：翻译失败，将使用英文原文")
        tool_descriptions_zh = unique_descriptions
    
    # 构建中文版工具链
    for chain_en in sampled_chains_en:
        tools_dict_zh = {}
        for tool_name in chain_en['tools'].keys():
            tools_dict_zh[tool_name] = tool_descriptions_zh.get(tool_name, chain_en['tools'][tool_name])
        
        sampled_chains_zh.append({
            "length": chain_en['length'],
            "tools": tools_dict_zh
        })
    
    # 保存中文版
    with open(output_path_zh, 'w', encoding='utf-8') as f:
        json.dump(sampled_chains_zh, f, indent=2, ensure_ascii=False)
    print(f"中文版已存至: {output_path_zh}")
    
    print(f"\n✓ 完成！生成了 {len(sampled_chains_en)} 条工具链（英文版和中文版）")

# --- 运行 ---
graph_file = r'data\function_graph.json'
tools_file = r'data\refined_tools.json'
chains_file_en = r'data\sampled_tool_chains_en.json'
chains_file_zh = r'data\sampled_tool_chains_zh.json'

random_walk_sampling(graph_file, tools_file, chains_file_en, chains_file_zh)