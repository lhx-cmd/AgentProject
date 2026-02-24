"""
测试多Agent对话合成系统
"""
import json
from MultiAgentDialogueSynthesis import MultiAgentDialogueSynthesis

def test_single_dialogue():
    """测试单个对话合成"""
    print("="*60)
    print("测试单个对话合成")
    print("="*60)
    
    # 初始化
    synthesizer = MultiAgentDialogueSynthesis(
        tools_file='data/refined_tools.json',
        chains_file='data/sampled_tool_chains.json'
    )
    
    # 选择一个短链进行测试
    test_chain = synthesizer.chains[0]['chain']  # 第一个链
    print(f"\n测试工具链: {test_chain}")
    print(f"链长度: {len(test_chain)}\n")
    
    # 合成对话
    dialogue = synthesizer.synthesize_dialogue(test_chain)
    
    # 打印结果
    print("\n生成的对话:")
    print("-"*60)
    for turn in dialogue['conversation']:
        role = turn['role']
        content = turn.get('content', '')
        
        if role == 'user':
            print(f"\n👤 User: {content}")
        elif role == 'assistant':
            print(f"\n🤖 Assistant: {content}")
            if 'reasoning' in turn:
                print(f"   💭 推理: {turn['reasoning'][:2]}")
            if 'tool_call' in turn:
                print(f"   🔧 工具调用: {turn['tool_call']['tool_name']}")
        elif role == 'tool':
            print(f"\n⚙️  Tool ({turn['tool_name']}): 返回结果")
    
    print("\n" + "="*60)
    print(f"对话统计:")
    print(f"- 总轮数: {dialogue['total_turns']}")
    print(f"- Assistant轮数: {dialogue['assistant_turns']}")
    print(f"- 工具调用数: {len([t for t in dialogue['conversation'] if t['role'] == 'tool'])}")
    print("="*60)
    
    return dialogue

def test_batch_synthesis():
    """测试批量合成（少量样本）"""
    print("\n" + "="*60)
    print("测试批量对话合成（3个样本）")
    print("="*60)
    
    synthesizer = MultiAgentDialogueSynthesis(
        tools_file='data/refined_tools.json',
        chains_file='data/sampled_tool_chains.json'
    )
    
    # 只合成3个对话进行测试
    dialogues = synthesizer.batch_synthesize(
        num_samples=3,
        output_file='data/test_dialogues.json'
    )
    
    print(f"\n测试完成！生成了 {len(dialogues)} 个对话")
    print(f"结果已保存到: data/test_dialogues.json")
    
    return dialogues

if __name__ == "__main__":
    # 测试单个对话
    print("\n🚀 开始测试...\n")
    
    try:
        # 1. 测试单个对话合成
        dialogue = test_single_dialogue()
        
        # 2. 测试批量合成
        # dialogues = test_batch_synthesis()
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
