import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import random

# 加载环境变量
load_dotenv('code/.env')

def generate_scenario_dashscope(tools_dict):
    """
    使用阿里云DashScope API（通义千问）生成工具链使用场景
    
    Args:
        tools_dict: {"tool_name": "description"}
    
    Returns:
        scenario: 场景描述字符串
    """
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        print("警告：未安装 dashscope 库，请运行: pip install dashscope")
        return None
    
    # 设置API key
    dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
    
    # 构建工具列表描述
    tools_list = []
    for idx, (tool_name, description) in enumerate(tools_dict.items(), 1):
        tools_list.append(f"{idx}. {tool_name}: {description}")
    
    tools_text = "\n".join(tools_list)
    
    # 构建提示词（不提及任务类型）
    prompt = f"""请根据以下工具链，生成一个简洁、具体的使用场景描述（8-15字）。

工具链包含的工具：
{tools_text}

要求：
1. 场景描述要简洁明了，8-15个字
2. 场景要能体现这些工具的实际应用场景
3. 场景要贴近真实业务需求
4. 只返回场景描述，不要其他内容
5. 不要使用"并行"、"顺序"、"混合"、"跨领域"等技术术语

示例格式：
- 非洲旅游规划
- 股票投资分析
- 社交媒体营销
- 在线教育平台管理
- 电商数据分析

请直接返回场景描述："""
    
    try:
        response = Generation.call(
            model='qwen-max-latest',
            messages=[
                {"role": "system", "content": "你是一个专业的产品经理，擅长根据工具功能设计实际应用场景。"},
                {"role": "user", "content": prompt}
            ],
            result_format='message',
            temperature=0.7
        )
        
        if response.status_code == 200:
            scenario = response.output.choices[0].message.content.strip()
            # 清理可能的多余内容
            scenario = scenario.split('\n')[0].strip()
            # 移除可能的引号和标点
            scenario = scenario.strip('"\'""''。，、')
            # 移除可能的序号
            scenario = scenario.lstrip('0123456789.-、 ')
            return scenario
        else:
            print(f"API调用失败: {response.code} - {response.message}")
            return None
            
    except Exception as e:
        print(f"生成场景时出错: {e}")
        return None

def generate_scenario_openai(tools_dict, client):
    """
    使用OpenAI API生成工具链使用场景
    
    Args:
        tools_dict: {"tool_name": "description"}
        client: OpenAI client
    
    Returns:
        scenario: 场景描述字符串
    """
    # 构建工具列表描述
    tools_list = []
    for idx, (tool_name, description) in enumerate(tools_dict.items(), 1):
        tools_list.append(f"{idx}. {tool_name}: {description}")
    
    tools_text = "\n".join(tools_list)
    
    # 构建提示词
    prompt = f"""请根据以下工具链，生成一个简洁、具体的使用场景描述（8-15字）。

工具链包含的工具：
{tools_text}

要求：
1. 场景描述要简洁明了，8-15个字
2. 场景要能体现这些工具的实际应用场景
3. 场景要贴近真实业务需求
4. 只返回场景描述，不要其他内容
5. 不要使用"并行"、"顺序"、"混合"、"跨领域"等技术术语

示例格式：
- 非洲旅游规划
- 股票投资分析
- 社交媒体营销
- 在线教育平台管理
- 电商数据分析

请直接返回场景描述："""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的产品经理，擅长根据工具功能设计实际应用场景。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        scenario = response.choices[0].message.content.strip()
        # 清理可能的多余内容
        scenario = scenario.split('\n')[0].strip()
        # 移除可能的引号和标点
        scenario = scenario.strip('"\'""''。，、')
        # 移除可能的序号
        scenario = scenario.lstrip('0123456789.-、 ')
        return scenario
        
    except Exception as e:
        print(f"生成场景时出错: {e}")
        return None

def sample_chains_by_type(chains, samples_per_type=20):
    """
    从每个类型中采样指定数量的工具链
    
    Args:
        chains: 所有工具链
        samples_per_type: 每个类型采样的数量
    
    Returns:
        采样后的工具链列表
    """
    # 按类型分组
    chains_by_type = {}
    for chain in chains:
        chain_type = chain.get('type', 'unknown')
        if chain_type not in chains_by_type:
            chains_by_type[chain_type] = []
        chains_by_type[chain_type].append(chain)
    
    # 从每个类型中采样
    sampled_chains = []
    for chain_type, type_chains in chains_by_type.items():
        if len(type_chains) <= samples_per_type:
            sampled_chains.extend(type_chains)
        else:
            sampled_chains.extend(random.sample(type_chains, samples_per_type))
    
    print(f"\n采样统计:")
    for chain_type, type_chains in sorted(chains_by_type.items()):
        sampled_count = min(len(type_chains), samples_per_type)
        print(f"  {chain_type}: {sampled_count}/{len(type_chains)} 条")
    
    return sampled_chains

def generate_scenarios_for_chains(input_file, output_file, samples_per_type=20):
    """
    为工具链生成使用场景
    
    Args:
        input_file: 输入的工具链文件（中文版）
        output_file: 输出的带场景的工具链文件
        samples_per_type: 每个类型采样的数量
    """
    # 加载工具链数据
    with open(input_file, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    
    print(f"已加载 {len(chains)} 条工具链")
    
    # 从每个类型中采样
    sampled_chains = sample_chains_by_type(chains, samples_per_type)
    print(f"\n采样后共 {len(sampled_chains)} 条工具链")
    
    # 初始化API客户端
    client = None
    use_dashscope = False
    
    if os.getenv('DASHSCOPE_API_KEY'):
        print("\n使用阿里云通义千问生成场景...")
        use_dashscope = True
    elif os.getenv('OPENAI_API_KEY'):
        print("\n使用OpenAI生成场景...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception as e:
            print(f"初始化OpenAI客户端失败: {e}")
            return
    else:
        print("错误：未找到API密钥，请在.env文件中配置DASHSCOPE_API_KEY或OPENAI_API_KEY")
        return
    
    # 为每条工具链生成场景
    chains_with_scenarios = []
    
    for chain in tqdm(sampled_chains, desc="生成场景"):
        tools_dict = chain['tools']
        
        # 生成场景
        scenario = None
        if use_dashscope:
            scenario = generate_scenario_dashscope(tools_dict)
        else:
            scenario = generate_scenario_openai(tools_dict, client)
        
        # 如果生成失败，使用默认场景
        if not scenario:
            # 根据工具生成简单场景
            first_tool = list(tools_dict.keys())[0]
            if 'Instagram' in first_tool or 'TikTok' in first_tool or 'Twitter' in first_tool:
                scenario = "社交媒体数据分析"
            elif 'Player' in first_tool or 'Statistics' in first_tool:
                scenario = "体育数据统计查询"
            elif 'QR' in first_tool or 'qr' in first_tool.lower():
                scenario = "二维码生成与管理"
            elif 'Video' in first_tool or 'Download' in first_tool:
                scenario = "视频下载与处理"
            elif 'News' in first_tool or 'Article' in first_tool:
                scenario = "新闻资讯获取"
            elif 'Movie' in first_tool or 'Film' in first_tool:
                scenario = "影视信息查询"
            elif 'Weather' in first_tool or 'weather' in first_tool.lower():
                scenario = "天气查询服务"
            elif 'Hotel' in first_tool or 'hotel' in first_tool.lower():
                scenario = "酒店预订服务"
            else:
                scenario = "数据查询与处理"
        
        # 构建输出格式（只保留必要字段）
        chain_with_scenario = {
            "scenario": scenario,
            "tools": tools_dict,
            "tools_count": len(tools_dict)
        }
        
        chains_with_scenarios.append(chain_with_scenario)
        
        # 避免API限流
        time.sleep(0.5)
    
    # 保存结果
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chains_with_scenarios, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 完成！已为 {len(chains_with_scenarios)} 条工具链生成场景")
    print(f"结果已保存至: {output_file}")
    
    # 显示示例
    if chains_with_scenarios:
        print("\n" + "="*60)
        print("场景生成示例:")
        print("="*60)
        for i, example in enumerate(chains_with_scenarios[:5], 1):
            print(f"\n示例 {i}:")
            print(f"场景: {example['scenario']}")
            print(f"工具数量: {example['tools_count']}")
            print(f"工具列表: {', '.join(list(example['tools'].keys())[:3])}...")

def main():
    """
    主函数
    """
    import sys
    
    # 支持命令行参数选择输入文件
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 默认使用聚类采样的结果
        input_file = 'data/sampled_tool_chains_cluster_zh.json'
    
    # 根据输入文件名生成输出文件名
    if 'cluster' in input_file:
        output_file = 'data/tool_chains_with_scenarios_cluster.json'
    elif 'enhanced' in input_file:
        output_file = 'data/tool_chains_with_scenarios_enhanced.json'
    else:
        output_file = 'data/tool_chains_with_scenarios.json'
    
    print("开始为工具链生成使用场景...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 每个类型采样20条
    generate_scenarios_for_chains(input_file, output_file, samples_per_type=20)

if __name__ == '__main__':
    main()
