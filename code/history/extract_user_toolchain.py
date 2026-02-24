"""
从synthesized_dialogues.json中提取UserContent和ToolChain
"""
import json
import os

def extract_user_toolchain(input_file, output_file):
    """
    从对话文件中提取UserContent和ToolChain
    
    Args:
        input_file: 输入的对话文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    print(f"共找到 {len(dialogues)} 个对话")
    
    # 提取UserContent和ToolChain
    extracted_data = []
    for i, dialogue in enumerate(dialogues):
        extracted_item = {
            "UserContent": dialogue.get("UserContent", ""),
            "ToolChain": dialogue.get("ToolChain", [])
        }
        extracted_data.append(extracted_item)
        
        # 打印前3个示例
        if i < 3:
            print(f"\n示例 {i+1}:")
            print(f"  UserContent: {extracted_item['UserContent'][:50]}...")
            print(f"  ToolChain: {extracted_item['ToolChain'][:3]}...")
    
    # 保存提取的数据
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 提取完成！")
    print(f"已保存到: {output_file}")
    print(f"共提取 {len(extracted_data)} 条记录")
    
    return extracted_data

def main():
    """主函数"""
    input_file = 'data/filtered_dialogues.json'
    output_file = 'data/user_toolchain_extracted_filtered.json'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到文件 {input_file}")
        return
    
    # 提取数据
    extracted_data = extract_user_toolchain(input_file, output_file)
    
    # 显示统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print(f"- 总记录数: {len(extracted_data)}")
    
    # 统计ToolChain长度分布
    chain_lengths = [len(item['ToolChain']) for item in extracted_data]
    if chain_lengths:
        print(f"- ToolChain长度范围: {min(chain_lengths)} - {max(chain_lengths)}")
        print(f"- 平均ToolChain长度: {sum(chain_lengths)/len(chain_lengths):.2f}")
    
    # 统计UserContent长度
    content_lengths = [len(item['UserContent']) for item in extracted_data]
    if content_lengths:
        print(f"- UserContent平均长度: {sum(content_lengths)/len(content_lengths):.1f} 字符")
    
    print("="*60)

if __name__ == "__main__":
    main()
