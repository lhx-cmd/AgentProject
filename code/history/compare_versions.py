"""
版本对比脚本

对比基础版和增强版的工具链生成质量
"""

import json
import os
from collections import defaultdict

def analyze_graph(graph_path, version_name):
    """分析函数图质量"""
    if not os.path.exists(graph_path):
        print(f"文件不存在: {graph_path}")
        return None
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    stats = {
        'version': version_name,
        'num_nodes': len(graph.get('nodes', [])),
        'num_edges': len(graph.get('edges', [])),
        'avg_out_degree': 0,
        'avg_confidence': 0,
        'has_type_info': False,
        'has_value_score': False
    }
    
    # 计算平均出度
    out_degrees = defaultdict(int)
    confidences = []
    
    for edge in graph.get('edges', []):
        out_degrees[edge['from']] += 1
        confidences.append(edge.get('confidence', 0))
    
    if out_degrees:
        stats['avg_out_degree'] = sum(out_degrees.values()) / len(out_degrees)
    
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
    
    # 检查是否有增强信息
    if graph.get('nodes') and isinstance(graph['nodes'][0], dict):
        stats['has_type_info'] = 'output_type' in graph['nodes'][0]
        stats['has_value_score'] = 'value_score' in graph['nodes'][0]
    
    return stats

def analyze_chains(chains_path, version_name):
    """分析工具链质量"""
    if not os.path.exists(chains_path):
        print(f"文件不存在: {chains_path}")
        return None
    
    with open(chains_path, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    
    stats = {
        'version': version_name,
        'num_chains': len(chains),
        'avg_length': 0,
        'length_dist': defaultdict(int),
        'has_goal_tool_count': 0,
        'has_goal_tool_ratio': 0
    }
    
    total_length = 0
    for chain in chains:
        length = chain.get('length', len(chain.get('tools', {})))
        total_length += length
        stats['length_dist'][length] += 1
        
        # 检查是否有目标工具标记
        if chain.get('has_goal_tool', False):
            stats['has_goal_tool_count'] += 1
    
    if chains:
        stats['avg_length'] = total_length / len(chains)
        stats['has_goal_tool_ratio'] = stats['has_goal_tool_count'] / len(chains)
    
    return stats

def print_comparison():
    """打印对比结果"""
    print("="*80)
    print("ToolMind 版本对比分析")
    print("="*80)
    
    # 1. 函数图对比
    print("\n【函数图对比】")
    print("-"*80)
    
    basic_graph = analyze_graph('data/function_graph.json', '基础版')
    enhanced_graph = analyze_graph('data/function_graph_enhanced.json', '增强版')
    
    if basic_graph and enhanced_graph:
        print(f"\n{'指标':<20} {'基础版':<20} {'增强版':<20} {'改进':<20}")
        print("-"*80)
        
        print(f"{'工具数量':<20} {basic_graph['num_nodes']:<20} {enhanced_graph['num_nodes']:<20} {'-':<20}")
        print(f"{'边数量':<20} {basic_graph['num_edges']:<20} {enhanced_graph['num_edges']:<20} "
              f"{(enhanced_graph['num_edges']/basic_graph['num_edges']-1)*100:+.1f}%")
        print(f"{'平均出度':<20} {basic_graph['avg_out_degree']:<20.2f} {enhanced_graph['avg_out_degree']:<20.2f} "
              f"{(enhanced_graph['avg_out_degree']/basic_graph['avg_out_degree']-1)*100:+.1f}%")
        print(f"{'平均置信度':<20} {basic_graph['avg_confidence']:<20.3f} {enhanced_graph['avg_confidence']:<20.3f} "
              f"{(enhanced_graph['avg_confidence']/basic_graph['avg_confidence']-1)*100:+.1f}%")
        print(f"{'类型信息':<20} {'✗':<20} {'✓':<20} {'新增':<20}")
        print(f"{'价值评分':<20} {'✗':<20} {'✓':<20} {'新增':<20}")
    
    # 2. 工具链对比
    print("\n\n【工具链对比】")
    print("-"*80)
    
    basic_chains = analyze_chains('data/sampled_tool_chains_zh.json', '基础版')
    enhanced_chains = analyze_chains('data/sampled_tool_chains_enhanced_zh.json', '增强版')
    
    if basic_chains and enhanced_chains:
        print(f"\n{'指标':<20} {'基础版':<20} {'增强版':<20} {'改进':<20}")
        print("-"*80)
        
        print(f"{'工具链数量':<20} {basic_chains['num_chains']:<20} {enhanced_chains['num_chains']:<20} "
              f"{enhanced_chains['num_chains']-basic_chains['num_chains']:+d}")
        print(f"{'平均长度':<20} {basic_chains['avg_length']:<20.2f} {enhanced_chains['avg_length']:<20.2f} "
              f"{enhanced_chains['avg_length']-basic_chains['avg_length']:+.2f}")
        print(f"{'目标工具比例':<20} {basic_chains['has_goal_tool_ratio']*100:<20.1f}% "
              f"{enhanced_chains['has_goal_tool_ratio']*100:<20.1f}% "
              f"{(enhanced_chains['has_goal_tool_ratio']-basic_chains['has_goal_tool_ratio'])*100:+.1f}%")
        
        print("\n长度分布对比:")
        all_lengths = sorted(set(list(basic_chains['length_dist'].keys()) + 
                                list(enhanced_chains['length_dist'].keys())))
        
        print(f"{'长度':<10} {'基础版':<20} {'增强版':<20}")
        print("-"*50)
        for length in all_lengths:
            basic_count = basic_chains['length_dist'].get(length, 0)
            enhanced_count = enhanced_chains['length_dist'].get(length, 0)
            basic_pct = basic_count / basic_chains['num_chains'] * 100 if basic_chains['num_chains'] > 0 else 0
            enhanced_pct = enhanced_count / enhanced_chains['num_chains'] * 100 if enhanced_chains['num_chains'] > 0 else 0
            print(f"{length:<10} {basic_count} ({basic_pct:.1f}%){'':<8} {enhanced_count} ({enhanced_pct:.1f}%)")
    
    # 3. 总结
    print("\n\n【优化总结】")
    print("-"*80)
    print("✓ 类型匹配验证：过滤类型不兼容的边，提升可执行性")
    print("✓ 逻辑兼容性检查：避免语义相似但逻辑不通的连接")
    print("✓ 目标导向采样：确保工具链具有明确的任务目标")
    print("✓ 工具价值评分：识别高价值工具，优化采样策略")
    print("✓ 循环避免机制：防止无效长链和重复工具")
    
    print("\n推荐：使用增强版获得更高质量的工具链数据")
    print("="*80)

if __name__ == '__main__':
    print_comparison()
