import json
import os

def refine_tools(file_path):
    refined_functions = {}
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                # 适配不同的数据结构，优先找 tools，找不到则看本身是否是工具定义
                raw_tools = data.get("tools", []) if isinstance(data, dict) else []
                
                for item in raw_tools:
                    func = item.get("function", {})
                    name = func.get("name")
                    description = func.get("description", "")
                    parameters = func.get("parameters", {})
                    
                    if not name or not description or len(description) < 10:
                        continue
                        
                    props = parameters.get("properties", {})
                    required = parameters.get("required", []) or []
                    
                    refined_params = []
                    is_valid_params = True
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type")
                        if not param_type:
                            is_valid_params = False
                            break
                        refined_params.append({
                            "name": param_name,
                            "type": param_type,
                            "description": param_info.get("description", ""),
                            "required": param_name in required
                        })
                        
                    if not is_valid_params:
                        continue

                    # 去重并保存
                    if name not in refined_functions:
                        refined_functions[name] = {
                            "function_name": name,
                            "description": description,
                            "parameters": refined_params,
                            "raw_schema": parameters  # 保留原始 Schema 方便后续对比
                        }
            except json.JSONDecodeError:
                continue

    return list(refined_functions.values())

def save_refined_data(data, output_path):
    # 自动创建目标文件夹
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 indent=4 让保存的 JSON 具有可读性
        # ensure_ascii=False 确保中文描述不会变成 \uXXXX 编码
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"保存成功！共计 {len(data)} 个函数，已存至: {output_path}")

# --- 主程序 ---
input_file = r'data\ToolACE-query.jsonl'
output_file = r'data\refined_tools.json'

# 1. 提取与精炼
result = refine_tools(input_file)

# 2. 保存结果
if result:
    save_refined_data(result, output_file)