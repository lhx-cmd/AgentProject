import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Set

# 自动检测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"当前使用设备: {device}")

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

class TypeMatcher:
    """类型匹配器：解决类型不匹配问题"""
    
    # 基础类型映射
    TYPE_GROUPS = {
        'string': {'string', 'str', 'text'},
        'number': {'number', 'integer', 'float', 'int'},
        'boolean': {'boolean', 'bool'},
        'array': {'array', 'list'},
        'object': {'object', 'dict', 'json'},
        'any': {'any'}
    }
    
    # 反向映射
    TYPE_TO_GROUP = {}
    for group, types in TYPE_GROUPS.items():
        for t in types:
            TYPE_TO_GROUP[t.lower()] = group
    
    @classmethod
    def normalize_type(cls, type_str) -> str:
        """标准化类型名称"""
        if not type_str:
            return 'any'
        
        # 处理列表类型（取第一个元素）
        if isinstance(type_str, list):
            if not type_str:
                return 'any'
            type_str = type_str[0]
        
        # 确保是字符串
        if not isinstance(type_str, str):
            return 'any'
        
        type_lower = type_str.lower().strip()
        return cls.TYPE_TO_GROUP.get(type_lower, 'any')
    
    @classmethod
    def is_compatible(cls, output_type: str, input_type: str) -> bool:
        """
        判断输出类型是否可以作为输入类型
        
        兼容规则：
        1. 相同类型组 -> 兼容
        2. any 类型 -> 兼容任何类型
        3. object/json 可以转换为 string
        4. array 可以转换为 string
        """
        out_norm = cls.normalize_type(output_type)
        in_norm = cls.normalize_type(input_type)
        
        # 相同类型
        if out_norm == in_norm:
            return True
        
        # any 类型
        if out_norm == 'any' or in_norm == 'any':
            return True
        
        # 特殊兼容规则
        if out_norm == 'object' and in_norm == 'string':
            return True
        if out_norm == 'array' and in_norm == 'string':
            return True
        
        return False

class LogicValidator:
    """逻辑验证器：解决逻辑断层问题"""
    
    # 逻辑不兼容的关键词对（黑名单）
    INCOMPATIBLE_PAIRS = [
        # 时间相关
        ({'time', 'date', 'timestamp', 'clock'}, {'weather', 'temperature', 'forecast'}),
        ({'time', 'date'}, {'price', 'stock', 'market'}),
        
        # 地理位置相关
        ({'location', 'address', 'place'}, {'user', 'account', 'profile'}),
        ({'coordinate', 'latitude', 'longitude'}, {'text', 'content', 'message'}),
        
        # 身份认证相关
        ({'token', 'auth', 'credential'}, {'weather', 'news', 'article'}),
        ({'password', 'secret'}, {'public', 'display', 'show'}),
        
        # 数据格式相关
        ({'image', 'photo', 'picture'}, {'text', 'string', 'word'}),
        ({'video', 'media'}, {'number', 'count', 'amount'}),
        
        # 业务逻辑相关
        ({'create', 'add', 'insert'}, {'delete', 'remove'}),
        ({'login', 'signin'}, {'logout', 'signout'}),
    ]
    
    # 逻辑兼容的关键词对（白名单，优先级更高）
    COMPATIBLE_PAIRS = [
        # 查询-详情链
        ({'search', 'query', 'find', 'list'}, {'detail', 'info', 'get'}),
        
        # 数据-分析链
        ({'data', 'statistics', 'metrics'}, {'analyze', 'analysis', 'report'}),
        
        # 内容-处理链
        ({'content', 'text', 'article'}, {'translate', 'summarize', 'extract'}),
        
        # 用户-操作链
        ({'user', 'account', 'profile'}, {'update', 'modify', 'edit'}),
        
        # 位置-服务链
        ({'location', 'address', 'place'}, {'nearby', 'around', 'distance'}),
        
        # 文件-处理链
        ({'file', 'document', 'upload'}, {'parse', 'process', 'convert'}),
    ]
    
    @classmethod
    def extract_keywords(cls, text: str) -> Set[str]:
        """从文本中提取关键词"""
        # 转小写并分词
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
    
    @classmethod
    def check_logic_compatibility(cls, output_desc: str, input_desc: str) -> Tuple[bool, float]:
        """
        检查逻辑兼容性
        
        Returns:
            (is_compatible, confidence_boost)
            - is_compatible: 是否逻辑兼容
            - confidence_boost: 置信度调整（-0.3 到 +0.2）
        """
        out_keywords = cls.extract_keywords(output_desc)
        in_keywords = cls.extract_keywords(input_desc)
        
        # 1. 检查白名单（逻辑兼容）
        for out_set, in_set in cls.COMPATIBLE_PAIRS:
            if out_keywords & out_set and in_keywords & in_set:
                return True, 0.2  # 提升置信度
        
        # 2. 检查黑名单（逻辑不兼容）
        for out_set, in_set in cls.INCOMPATIBLE_PAIRS:
            if out_keywords & out_set and in_keywords & in_set:
                return False, -0.3  # 降低置信度
        
        # 3. 默认：中性
        return True, 0.0

class SemanticFilter:
    """语义过滤器：解决搜索空间冗余问题"""
    
    # 任务目标关键词（高价值工具）
    GOAL_KEYWORDS = {
        'analyze', 'analysis', 'report', 'summary', 'summarize',
        'generate', 'create', 'build', 'make',
        'send', 'notify', 'alert', 'email',
        'save', 'store', 'export', 'download',
        'visualize', 'chart', 'graph', 'plot',
        'recommend', 'suggest', 'predict',
        'translate', 'convert', 'transform'
    }
    
    # 中间步骤关键词（中等价值）
    INTERMEDIATE_KEYWORDS = {
        'get', 'fetch', 'retrieve', 'query', 'search', 'find',
        'filter', 'sort', 'group', 'aggregate',
        'parse', 'extract', 'process', 'calculate',
        'validate', 'check', 'verify'
    }
    
    # 低价值关键词（应避免作为终点）
    LOW_VALUE_KEYWORDS = {
        'list', 'show', 'display', 'view',
        'count', 'length', 'size'
    }
    
    @classmethod
    def calculate_tool_value(cls, tool_name: str, description: str) -> float:
        """
        计算工具的价值分数（0-1）
        
        高价值工具更适合作为工具链的终点
        """
        text = (tool_name + ' ' + description).lower()
        keywords = re.findall(r'\b\w+\b', text)
        
        score = 0.5  # 基础分
        
        for word in keywords:
            if word in cls.GOAL_KEYWORDS:
                score += 0.3
            elif word in cls.INTERMEDIATE_KEYWORDS:
                score += 0.1
            elif word in cls.LOW_VALUE_KEYWORDS:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    @classmethod
    def is_valid_chain_end(cls, tool_name: str, description: str) -> bool:
        """判断工具是否适合作为链的终点"""
        value = cls.calculate_tool_value(tool_name, description)
        return value >= 0.6  # 高价值工具才能作为终点

def infer_output_type(tool_name: str, description: str) -> str:
    """
    推断工具的输出类型
    
    基于工具名称和描述推断
    """
    text = (tool_name + ' ' + description).lower()
    
    # 规则匹配
    if any(word in text for word in ['list', 'array', 'multiple', 'all']):
        return 'array'
    elif any(word in text for word in ['count', 'number', 'amount', 'price', 'score']):
        return 'number'
    elif any(word in text for word in ['is', 'has', 'exists', 'valid', 'check']):
        return 'boolean'
    elif any(word in text for word in ['data', 'info', 'detail', 'profile', 'object']):
        return 'object'
    else:
        return 'string'  # 默认

def build_enhanced_function_graph(input_path, output_path, similarity_threshold=0.65):
    """
    构建增强版函数图
    
    改进：
    1. 类型匹配验证
    2. 逻辑兼容性检查
    3. 工具价值评分
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)

    num_tools = len(tools)
    graph_edges = []
    
    print(f"正在准备向量数据 (共 {num_tools} 个函数)...")
    
    # 数据质量检查
    tools_with_params = sum(1 for t in tools if t.get('parameters'))
    total_params = sum(len(t.get('parameters', [])) for t in tools)
    print(f"  - 有参数的工具: {tools_with_params}/{num_tools}")
    print(f"  - 总参数数: {total_params}")
    
    # 1. 预计算输出向量和推断输出类型
    a_outputs_texts = []
    output_types = []
    tool_values = []
    
    for t in tools:
        # 模拟输出描述
        text = f"{t['function_name']} {' '.join(t['description'].split()[:5])}"
        a_outputs_texts.append(text)
        
        # 推断输出类型
        output_type = infer_output_type(t['function_name'], t['description'])
        output_types.append(output_type)
        
        # 计算工具价值
        value = SemanticFilter.calculate_tool_value(t['function_name'], t['description'])
        tool_values.append(value)
    
    print("正在计算输出向量...")
    output_embeddings = model.encode(a_outputs_texts, convert_to_tensor=True, show_progress_bar=True)

    # 2. 收集参数信息
    param_info_list = []
    all_param_texts = []
    
    for idx, t in enumerate(tools):
        for p in t['parameters']:
            all_param_texts.append(p['description'])
            
            # 处理参数类型（可能是字符串、列表或其他格式）
            param_type = p.get('type', 'any')
            if isinstance(param_type, list):
                param_type = param_type[0] if param_type else 'any'
            elif not isinstance(param_type, str):
                param_type = 'any'
            
            param_info_list.append({
                "tool_idx": idx,
                "param_name": p['name'],
                "param_type": param_type,
                "param_desc": p['description']
            })
    
    print(f"正在计算参数向量 (共 {len(all_param_texts)} 个参数)...")
    param_embeddings = model.encode(all_param_texts, convert_to_tensor=True, show_progress_bar=True)

    # 3. 构建图（带增强验证）
    print(f"开始构建增强版函数图...")
    
    stats = {
        'total_candidates': 0,
        'passed_semantic': 0,
        'passed_type': 0,
        'passed_logic': 0,
        'final_edges': 0
    }
    
    for i in tqdm(range(num_tools), desc="构建进度"):
        emb_a = output_embeddings[i]
        output_type_a = output_types[i]
        output_desc_a = a_outputs_texts[i]
        
        # 计算语义相似度
        cosine_scores = util.cos_sim(emb_a, param_embeddings)[0]
        relevant_indices = (cosine_scores > similarity_threshold).nonzero(as_tuple=True)[0]
        
        stats['total_candidates'] += len(relevant_indices)
        
        for rel_idx in relevant_indices:
            rel_idx = int(rel_idx)
            target_param = param_info_list[rel_idx]
            j = target_param['tool_idx']
            
            if i == j:
                continue  # 排除自环
            
            stats['passed_semantic'] += 1
            
            # 验证1：类型兼容性
            param_type = target_param['param_type']
            if not TypeMatcher.is_compatible(output_type_a, param_type):
                continue
            
            stats['passed_type'] += 1
            
            # 验证2：逻辑兼容性
            param_desc = target_param['param_desc']
            is_logic_ok, confidence_boost = LogicValidator.check_logic_compatibility(
                output_desc_a, param_desc
            )
            
            if not is_logic_ok:
                continue
            
            stats['passed_logic'] += 1
            
            # 调整置信度
            final_confidence = float(cosine_scores[rel_idx]) + confidence_boost
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            # 只保留调整后仍然高于阈值的边
            if final_confidence < similarity_threshold:
                continue
            
            graph_edges.append({
                "from": tools[i]['function_name'],
                "to": tools[j]['function_name'],
                "via_parameter": target_param['param_name'],
                "confidence": round(final_confidence, 4),
                "type_match": f"{output_type_a}->{param_type}",
                "semantic_score": round(float(cosine_scores[rel_idx]), 4)
            })
            
            stats['final_edges'] += 1

    # 4. 保存图结构（包含工具元数据）
    graph_data = {
        "metadata": {
            "num_tools": num_tools,
            "num_edges": len(graph_edges),
            "similarity_threshold": similarity_threshold,
            "enhancements": ["type_matching", "logic_validation", "value_scoring"]
        },
        "nodes": [
            {
                "name": t['function_name'],
                "description": t['description'],
                "output_type": output_types[idx],
                "value_score": round(tool_values[idx], 3),
                "is_goal_tool": tool_values[idx] >= 0.6
            }
            for idx, t in enumerate(tools)
        ],
        "edges": graph_edges
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    # 5. 输出统计信息
    print(f"\n{'='*60}")
    print("图构建完成！")
    print(f"{'='*60}")
    print(f"总工具数: {num_tools}")
    print(f"总边数: {len(graph_edges)}")
    print(f"\n过滤统计:")
    print(f"  语义候选: {stats['total_candidates']}")
    print(f"  通过语义: {stats['passed_semantic']} ({stats['passed_semantic']/max(1,stats['total_candidates'])*100:.1f}%)")
    print(f"  通过类型: {stats['passed_type']} ({stats['passed_type']/max(1,stats['passed_semantic'])*100:.1f}%)")
    print(f"  通过逻辑: {stats['passed_logic']} ({stats['passed_logic']/max(1,stats['passed_type'])*100:.1f}%)")
    print(f"  最终保留: {stats['final_edges']}")
    print(f"\n高价值工具数: {sum(1 for v in tool_values if v >= 0.6)}")
    print(f"已保存至: {output_path}")
    print(f"{'='*60}")

# --- 运行 ---
if __name__ == '__main__':
    input_file = r'data/refined_tools.json'
    graph_file = r'data/function_graph_enhanced.json'
    
    build_enhanced_function_graph(input_file, graph_file, similarity_threshold=0.65)
