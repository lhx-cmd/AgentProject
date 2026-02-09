"""
生成格式化的对话数据
输出格式: {"UserContent": "...", "ToolChain": [...], ...}
"""
import json
from MultiAgentDialogueSynthesis import MultiAgentDialogueSynthesis

def generate_formatted_dialogues(num_samples=5, output_file='data/formatted_dialogues.json'):
    """
    生成格式化的对话数据
    
    Args:
        num_samples: 要生成的对话数量
        output_file: 输出文件路径
    """
    print("="*60)
    print("开始生成格式化对话数据")
    print("="*60)
    
    # 初始化合成器
    synthesizer = MultiAgentDialogueSynthesis(
        tools_file='data/refined_tools.json',
        chains_file='data/sampled_tool_chains.json'
    )
    
    all_dialogues = []
    
    print(f"\n将生成 {num_samples} 个对话...\n")
    
    for i, chain_data in enumerate(synthesizer.chains[:num_samples]):
        print(f"进度: {i+1}/{num_samples}")
        
        chain = chain_data['chain']
        
        # 生成对话
        dialogue = synthesizer.synthesize_dialogue(chain)
        
        # 格式化输出（只保留关键字段）
        formatted_dialogue = {
            "UserContent": dialogue["UserContent"],
            "ToolChain": dialogue["ToolChain"],
            "conversation": dialogue["conversation"],
            "total_turns": dialogue["total_turns"],
            "assistant_turns": dialogue["assistant_turns"]
        }
        
        all_dialogues.append(formatted_dialogue)
        
        # 每5个保存一次
        if (i + 1) % 5 == 0:
            save_dialogues(all_dialogues, output_file)
            print(f"✓ 已保存 {i+1} 个对话\n")
    
    # 最终保存
    save_dialogues(all_dialogues, output_file)
    
    print("\n" + "="*60)
    print("生成完成！")
    print(f"总对话数: {len(all_dialogues)}")
    print(f"输出文件: {output_file}")
    print("="*60)
    
    # 显示示例
    if all_dialogues:
        print("\n示例对话:")
        print("-"*60)
        example = all_dialogues[0]
        print(f"UserContent: {example['UserContent']}")
        print(f"ToolChain: {example['ToolChain']}")
        print(f"Total Turns: {example['total_turns']}")
        print("-"*60)
    
    return all_dialogues

def save_dialogues(dialogues, output_file):
    """保存对话到JSON文件"""
    import os
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)

def preview_format():
    """预览输出格式"""
    print("\n输出格式示例:")
    print("="*60)
    
    example = {
        "UserContent": "我想查询一些体育赛事的排名信息",
        "ToolChain": [
            "Get Tournament Standings",
            "LeagueTotalStandings",
            "Baseball League Total Standings"
        ],
        "conversation": [
            {
                "role": "user",
                "content": "我想查询一些体育赛事的排名信息",
                "turn": 1
            },
            {
                "role": "assistant",
                "content": "我将调用 Get Tournament Standings 来帮助您",
                "reasoning": ["理解用户需求：查询体育赛事排名", "选择工具：Get Tournament Standings"],
                "tool_call": {
                    "tool_name": "Get Tournament Standings",
                    "parameters": {"tournament_id": "mock_1234"}
                },
                "turn": 2
            }
        ],
        "total_turns": 8,
        "assistant_turns": 4
    }
    
    print(json.dumps(example, ensure_ascii=False, indent=2))
    print("="*60)

if __name__ == "__main__":
    # 预览格式
    preview_format()
    
    # 生成对话（可以调整数量）
    print("\n开始生成...\n")
    
    # 先生成少量测试
    dialogues = generate_formatted_dialogues(
        num_samples=5,  # 可以改为更大的数字，如 50, 100
        output_file='data/formatted_dialogues.json'
    )
    
    print("\n✅ 完成！")
