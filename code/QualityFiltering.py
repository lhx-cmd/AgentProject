import json
import os
from typing import List, Dict, Any, Tuple

class QualityFiltering:
    """
    两阶段质量过滤系统
    1. 轨迹级别过滤（Trajectory-Level）：检查整体对话完整性和任务成功性
    2. 轮次级别过滤（Turn-Level）：精细检查每个assistant响应，移除错误或次优步骤
    """
    
    def __init__(self, dialogues_file: str):
        """
        初始化质量过滤系统
        
        Args:
            dialogues_file: 合成对话文件路径
        """
        self.dialogues = self._load_dialogues(dialogues_file)
        self.filtered_dialogues = []
        self.filter_stats = {
            'total_dialogues': 0,
            'trajectory_filtered': 0,
            'turn_filtered': 0,
            'passed_dialogues': 0,
            'total_turns_before': 0,
            'total_turns_after': 0,
            'removed_turns': 0
        }
    
    def _load_dialogues(self, file_path: str) -> List[Dict]:
        """加载对话数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def filter_all(self) -> List[Dict]:
        """
        执行完整的两阶段过滤流程
        
        Returns:
            过滤后的高质量对话列表
        """
        self.filter_stats['total_dialogues'] = len(self.dialogues)
        
        for dialogue in self.dialogues:
            # 阶段1: 轨迹级别过滤
            if not self._trajectory_level_filter(dialogue):
                self.filter_stats['trajectory_filtered'] += 1
                continue
            
            # 阶段2: 轮次级别过滤
            filtered_dialogue = self._turn_level_filter(dialogue)
            
            # 检查过滤后是否仍然有效
            if self._is_valid_after_turn_filter(filtered_dialogue):
                self.filtered_dialogues.append(filtered_dialogue)
                self.filter_stats['passed_dialogues'] += 1
            else:
                self.filter_stats['turn_filtered'] += 1
        
        return self.filtered_dialogues
    
    def _trajectory_level_filter(self, dialogue: Dict) -> bool:
        """
        轨迹级别过滤：检查整体对话质量
        
        过滤标准：
        1. 对话完整性：是否有用户查询和assistant响应
        2. 任务成功性：是否成功调用了工具链中的工具
        3. 对话合理性：轮次数是否在合理范围内
        4. 推理质量：assistant是否提供了推理过程
        """
        conversation = dialogue.get('conversation', [])
        tool_chain = dialogue.get('ToolChain', [])
        
        # 检查1: 对话不能为空
        if not conversation or len(conversation) < 3:
            return False
        
        # 检查2: 必须有用户查询
        user_turns = [turn for turn in conversation if turn.get('role') == 'user']
        if not user_turns:
            return False
        
        # 检查3: 必须有assistant响应
        assistant_turns = [turn for turn in conversation if turn.get('role') == 'assistant']
        if not assistant_turns:
            return False
        
        # 检查4: 检查是否有工具调用
        tool_calls = [turn for turn in conversation if turn.get('role') == 'tool']
        if not tool_calls:
            return False
        
        # 检查5: 工具调用数量应该合理（至少调用了工具链中的部分工具）
        if len(tool_calls) < min(2, len(tool_chain)):
            return False
        
        # 检查6: 对话轮次不应过长（避免无效循环）
        if len(conversation) > 50:  # 设置最大轮次限制
            return False
        
        # 检查7: 检查是否有过多的重复澄清（可能表示对话陷入循环）
        if self._has_excessive_clarifications(conversation):
            return False
        
        # 检查8: 检查assistant是否提供了推理过程
        reasoning_turns = [turn for turn in assistant_turns 
                          if turn.get('reasoning') and len(turn.get('reasoning', [])) > 0]
        if len(reasoning_turns) < len(tool_calls):  # 每次工具调用应该有推理
            return False
        
        # 检查9: 检查工具调用是否成功
        failed_tools = [turn for turn in tool_calls 
                       if turn.get('tool_result', {}).get('status') != 'success']
        if len(failed_tools) > len(tool_calls) * 0.3:  # 失败率不应超过30%
            return False
        
        return True
    
    def _has_excessive_clarifications(self, conversation: List[Dict]) -> bool:
        """
        检查是否有过多的澄清请求（可能表示对话陷入循环）
        """
        clarification_keywords = ['请问', '请提供', '请确认', '请告知', '请说明']
        
        assistant_turns = [turn for turn in conversation if turn.get('role') == 'assistant']
        clarification_count = 0
        
        for turn in assistant_turns:
            content = turn.get('content', '')
            if any(keyword in content for keyword in clarification_keywords):
                clarification_count += 1
        
        # 如果超过50%的assistant回复都是澄清，认为有问题
        return clarification_count > len(assistant_turns) * 0.5
    
    def _turn_level_filter(self, dialogue: Dict) -> Dict:
        """
        轮次级别过滤：精细检查每个assistant响应
        
        过滤标准：
        1. 移除无效的工具调用（参数错误、工具不存在等）
        2. 移除推理断裂的轮次（推理与行动不一致）
        3. 移除重复或冗余的轮次
        4. 保留自纠正信号（assistant发现错误并修正）
        """
        conversation = dialogue.get('conversation', [])
        filtered_conversation = []
        
        self.filter_stats['total_turns_before'] += len(conversation)
        
        i = 0
        while i < len(conversation):
            turn = conversation[i]
            role = turn.get('role')
            
            # 用户轮次和工具轮次直接保留
            if role in ['user', 'tool']:
                filtered_conversation.append(turn)
                i += 1
                continue
            
            # Assistant轮次需要检查
            if role == 'assistant':
                if self._is_valid_assistant_turn(turn, conversation, i):
                    filtered_conversation.append(turn)
                else:
                    self.filter_stats['removed_turns'] += 1
                    # 如果移除了assistant轮次，可能需要移除对应的tool轮次
                    if i + 1 < len(conversation) and conversation[i + 1].get('role') == 'tool':
                        i += 1  # 跳过对应的tool轮次
                        self.filter_stats['removed_turns'] += 1
            
            i += 1
        
        self.filter_stats['total_turns_after'] += len(filtered_conversation)
        
        # 更新对话
        filtered_dialogue = dialogue.copy()
        filtered_dialogue['conversation'] = filtered_conversation
        filtered_dialogue['total_turns'] = len(filtered_conversation)
        filtered_dialogue['assistant_turns'] = len([t for t in filtered_conversation 
                                                    if t.get('role') == 'assistant'])
        
        return filtered_dialogue

    
    def _is_valid_assistant_turn(self, turn: Dict, conversation: List[Dict], 
                                 turn_index: int) -> bool:
        """
        检查assistant轮次是否有效
        
        检查项：
        1. 推理质量：是否有合理的推理过程
        2. 工具调用有效性：如果有工具调用，参数是否合理
        3. 内容质量：回复是否有实质内容
        4. 一致性：推理与行动是否一致
        """
        content = turn.get('content', '').strip()
        reasoning = turn.get('reasoning', [])
        tool_call = turn.get('tool_call')
        
        # 检查1: 内容不能为空
        if not content:
            return False
        
        # 检查2: 如果有工具调用，必须有推理过程
        if tool_call and len(reasoning) < 2:
            return False
        
        # 检查3: 检查是否是无意义的重复
        if self._is_meaningless_repetition(turn, conversation, turn_index):
            return False
        
        # 检查4: 检查工具调用的有效性
        if tool_call:
            if not self._is_valid_tool_call(tool_call, turn):
                return False
        
        # 检查5: 检查推理的连贯性
        if reasoning and not self._is_coherent_reasoning(reasoning):
            return False
        
        # 检查6: 检查是否是无效的澄清循环
        if self._is_invalid_clarification_loop(turn, conversation, turn_index):
            return False
        
        return True
    
    def _is_meaningless_repetition(self, turn: Dict, conversation: List[Dict], 
                                   turn_index: int) -> bool:
        """
        检查是否是无意义的重复
        """
        content = turn.get('content', '')
        
        # 检查前面的assistant轮次
        for i in range(max(0, turn_index - 5), turn_index):
            if conversation[i].get('role') == 'assistant':
                prev_content = conversation[i].get('content', '')
                # 如果内容高度相似（简单的字符串匹配）
                if self._similarity_ratio(content, prev_content) > 0.8:
                    return True
        
        return False
    
    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（简化版）
        """
        if not str1 or not str2:
            return 0.0
        
        # 简单的基于长度和公共子串的相似度
        longer = max(len(str1), len(str2))
        shorter = min(len(str1), len(str2))
        
        if longer == 0:
            return 1.0
        
        # 计算公共字符数
        common = sum(1 for c in str1 if c in str2)
        return common / longer
    
    def _is_valid_tool_call(self, tool_call: Dict, turn: Dict) -> bool:
        """
        检查工具调用是否有效
        """
        tool_name = tool_call.get('tool_name')
        parameters = tool_call.get('parameters', {})
        
        # 检查1: 工具名称不能为空
        if not tool_name:
            return False
        
        # 检查2: 参数不能全是mock值（表示没有真实参数）
        if parameters:
            mock_count = sum(1 for v in parameters.values() 
                           if isinstance(v, str) and 'mock' in v.lower())
            if mock_count == len(parameters):
                return False
        
        # 检查3: 推理中应该提到工具名称
        reasoning = turn.get('reasoning', [])
        reasoning_text = ' '.join(reasoning)
        if tool_name not in reasoning_text:
            return False
        
        return True
    
    def _is_coherent_reasoning(self, reasoning: List[str]) -> bool:
        """
        检查推理是否连贯
        """
        if not reasoning or len(reasoning) < 2:
            return False
        
        # 检查每条推理是否有实质内容
        for step in reasoning:
            if len(step.strip()) < 10:  # 太短的推理可能无意义
                return False
        
        # 检查是否有明显的逻辑断裂（简化检查）
        # 这里可以添加更复杂的NLP检查
        
        return True
    
    def _is_invalid_clarification_loop(self, turn: Dict, conversation: List[Dict], 
                                       turn_index: int) -> bool:
        """
        检查是否陷入无效的澄清循环
        """
        content = turn.get('content', '')
        
        # 如果这是一个澄清请求
        clarification_keywords = ['请问', '请提供', '请确认']
        if not any(keyword in content for keyword in clarification_keywords):
            return False
        
        # 检查用户是否已经提供了相同的信息
        if turn_index < 2:
            return False
        
        # 查看前面的用户回复
        for i in range(turn_index - 1, max(0, turn_index - 10), -1):
            if conversation[i].get('role') == 'user':
                user_content = conversation[i].get('content', '')
                # 如果用户已经提供了信息，但assistant还在要求相同信息
                if self._similarity_ratio(content, user_content) > 0.5:
                    return True
        
        return False
    
    def _is_valid_after_turn_filter(self, dialogue: Dict) -> bool:
        """
        检查轮次过滤后对话是否仍然有效
        """
        conversation = dialogue.get('conversation', [])
        
        # 至少要有3轮对话
        if len(conversation) < 3:
            return False
        
        # 必须有用户、assistant和tool轮次
        roles = set(turn.get('role') for turn in conversation)
        if not {'user', 'assistant', 'tool'}.issubset(roles):
            return False
        
        # 至少要有1次成功的工具调用
        tool_turns = [turn for turn in conversation if turn.get('role') == 'tool']
        if not tool_turns:
            return False
        
        return True
    
    def save_filtered_dialogues(self, output_file: str):
        """
        保存过滤后的对话
        """
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.filtered_dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已保存 {len(self.filtered_dialogues)} 个高质量对话到: {output_file}")
    
    def print_statistics(self):
        """
        打印过滤统计信息
        """
        print("\n" + "="*60)
        print("质量过滤统计报告")
        print("="*60)
        print(f"总对话数: {self.filter_stats['total_dialogues']}")
        print(f"轨迹级别过滤: {self.filter_stats['trajectory_filtered']} "
              f"({self.filter_stats['trajectory_filtered']/self.filter_stats['total_dialogues']*100:.1f}%)")
        print(f"轮次级别过滤: {self.filter_stats['turn_filtered']} "
              f"({self.filter_stats['turn_filtered']/self.filter_stats['total_dialogues']*100:.1f}%)")
        print(f"通过过滤的对话: {self.filter_stats['passed_dialogues']} "
              f"({self.filter_stats['passed_dialogues']/self.filter_stats['total_dialogues']*100:.1f}%)")
        print(f"\n总轮次数（过滤前）: {self.filter_stats['total_turns_before']}")
        print(f"总轮次数（过滤后）: {self.filter_stats['total_turns_after']}")
        print(f"移除的轮次: {self.filter_stats['removed_turns']} "
              f"({self.filter_stats['removed_turns']/self.filter_stats['total_turns_before']*100:.1f}%)")
        print("="*60)


def main():
    """
    主函数：执行质量过滤流程
    """
    # 输入输出文件路径
    input_file = 'data/synthesized_dialogues.json'
    output_file = 'data/filtered_dialogues.json'
    
    print("开始质量过滤流程...")
    print(f"输入文件: {input_file}")
    
    # 创建过滤器
    filter_system = QualityFiltering(input_file)
    
    # 执行过滤
    print("\n执行两阶段过滤...")
    filtered_dialogues = filter_system.filter_all()
    
    # 保存结果
    filter_system.save_filtered_dialogues(output_file)
    
    # 打印统计信息
    filter_system.print_statistics()
    
    # 显示示例
    if filtered_dialogues:
        print("\n" + "="*60)
        print("过滤后的对话示例:")
        print("="*60)
        example = filtered_dialogues[0]
        print(f"用户查询: {example['UserContent']}")
        print(f"工具链长度: {example['chain_length']}")
        print(f"总轮次: {example['total_turns']}")
        print(f"Assistant轮次: {example['assistant_turns']}")
        print(f"工具链: {' -> '.join(example['ToolChain'][:3])}...")


if __name__ == '__main__':
    main()
