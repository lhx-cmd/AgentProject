import json
import os
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
from http import HTTPStatus

# 加载环境变量
load_dotenv()

class MultiAgentDialogueSynthesis:
    """
    多Agent多轮对话合成系统
    模拟User Agent、Assistant Agent和Tool Agent之间的真实交互
    """
    
    def __init__(self, tools_file: str, chains_file: str):
        """
        初始化多Agent对话合成系统
        
        Args:
            tools_file: 工具描述文件路径
            chains_file: 采样的工具链文件路径
        """
        self.tools = self._load_tools(tools_file)
        self.chains = self._load_chains(chains_file)
        self.tool_dict = {tool['function_name']: tool for tool in self.tools}
        
        # 配置DashScope API
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY环境变量")
        
        # 使用DashScope的兼容接口
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
    def _load_tools(self, file_path: str) -> List[Dict]:
        """加载工具描述数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_chains(self, file_path: str) -> List[Dict]:
        """加载采样的工具链"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_user_query(self, chain: List[str]) -> str:
        """
        User Agent: 根据工具链生成用户查询
        模拟真实用户意图
        """
        # 获取链中的工具信息
        tool_descriptions = []
        for tool_name in chain[:2]:  # 使用前2个工具推断用户意图
            if tool_name in self.tool_dict:
                tool_descriptions.append(self.tool_dict[tool_name]['description'])
        
        # 根据工具类型生成不同风格的查询
        query_templates = [
            f"我需要{self._extract_action(tool_descriptions)}，你能帮我吗？",
            f"帮我{self._extract_action(tool_descriptions)}",
            f"请协助我完成{self._extract_action(tool_descriptions)}的任务",
            f"我想{self._extract_action(tool_descriptions)}，应该怎么做？"
        ]
        
        return random.choice(query_templates)
    
    def _extract_action(self, descriptions: List[str]) -> str:
        """从工具描述中提取动作关键词"""
        if not descriptions:
            return "完成一些操作"
        
        # 简化：提取描述中的关键动作
        desc = descriptions[0].lower()
        if 'search' in desc or '搜索' in desc:
            return "搜索相关信息"
        elif 'get' in desc or '获取' in desc or 'retrieve' in desc:
            return "获取数据"
        elif 'create' in desc or '创建' in desc:
            return "创建内容"
        elif 'update' in desc or '更新' in desc:
            return "更新信息"
        else:
            return "处理相关任务"

    
    def generate_tool_result(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """
        Tool Agent: 模拟工具执行并返回结果
        使用LLM生成合理的模拟结果
        """
        tool_info = self.tool_dict.get(tool_name)
        if not tool_info:
            return {"error": "Tool not found"}
        
        # 使用LLM生成合理的工具返回结果
        prompt = f"""你是一个工具执行模拟器。根据以下工具信息和参数，生成一个合理的JSON格式返回结果。

工具名称: {tool_name}
工具描述: {tool_info['description']}
调用参数: {json.dumps(parameters, ensure_ascii=False)}

要求：
1. 返回结果必须是有效的JSON格式
2. 结果应该符合工具的功能描述
3. 包含合理的模拟数据
4. 只返回JSON，不要有其他解释文字

示例格式：
{{"status": "success", "data": {{"id": "123", "name": "example"}}, "message": "操作成功"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result_text = response.choices[0].message.content.strip()
            # 提取JSON（可能包含markdown代码块）
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(result_text)
        except Exception as e:
            print(f"生成工具结果时出错: {e}")
            return {"status": "success", "data": {}, "message": "模拟执行成功"}
    
    def generate_user_query_with_llm(self, chain: List[str]) -> str:
        """
        使用LLM生成更真实的用户查询
        """
        # 获取工具链信息
        tool_infos = []
        for tool_name in chain[:3]:
            if tool_name in self.tool_dict:
                tool_infos.append({
                    "name": tool_name,
                    "description": self.tool_dict[tool_name]['description']
                })
        
        prompt = f"""你是一个用户行为模拟器。根据以下工具链，生成一个真实用户可能提出的自然语言查询。

工具链信息：
{json.dumps(tool_infos, ensure_ascii=False, indent=2)}

要求：
1. 查询应该自然、口语化
2. 体现用户的真实需求和意图
3. 不要直接提及工具名称
4. 长度控制在1-2句话
5. 只返回查询文本，不要有其他解释

示例：
- "我想查一下最近的天气情况，然后帮我订个机票"
- "帮我搜索一些相关的新闻资讯"
- "能帮我获取用户的详细信息吗？"
"""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"生成用户查询时出错: {e}")
            return self.generate_user_query(chain)  # 回退到原方法
    
    def generate_assistant_response_with_llm(self, conversation_history: List[Dict], 
                                             current_tool: str) -> Dict[str, Any]:
        """
        使用LLM生成Assistant的推理和响应
        """
        tool_info = self.tool_dict.get(current_tool)
        if not tool_info:
            return None
        
        # 构建对话历史
        history_text = "\n".join([
            f"{turn['role']}: {turn.get('content', turn.get('reasoning', ''))}"
            for turn in conversation_history[-3:]  # 只保留最近3轮
        ])
        
        prompt = f"""你是一个智能助手Agent。根据对话历史和当前需要调用的工具，生成你的推理过程和工具调用决策。

对话历史：
{history_text}

当前工具信息：
名称: {current_tool}
描述: {tool_info['description']}
参数: {json.dumps(tool_info['parameters'], ensure_ascii=False, indent=2)}

要求：
1. 生成清晰的推理步骤（chain-of-thought）
2. 决定是否需要调用工具
3. 如果调用工具，生成合理的参数值
4. 如果信息不足，生成澄清问题
5. 返回JSON格式

返回格式：
{{
    "reasoning": ["推理步骤1", "推理步骤2", "推理步骤3"],
    "action": "call_tool" 或 "ask_clarification",
    "tool_call": {{"tool_name": "{current_tool}", "parameters": {{}}}},
    "clarification": "需要澄清的问题（如果有）"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(result_text)
        except Exception as e:
            print(f"生成助手响应时出错: {e}")
            return self.generate_assistant_reasoning("", current_tool, conversation_history)

    
    def synthesize_dialogue(self, chain: List[str], max_turns: int = 10) -> Dict[str, Any]:
        """
        合成完整的多轮对话
        
        Args:
            chain: 工具调用链
            max_turns: 最大对话轮数
            
        Returns:
            完整的对话轨迹
        """
        conversation = []
        tool_index = 0
        
        # 1. User Agent: 生成初始查询
        user_query = self.generate_user_query_with_llm(chain)
        conversation.append({
            "role": "user",
            "content": user_query,
            "turn": len(conversation) + 1
        })
        
        print(f"\n{'='*60}")
        print(f"开始合成对话 - 工具链长度: {len(chain)}")
        print(f"User: {user_query}")
        
        # 2. 多轮交互
        while tool_index < len(chain) and len(conversation) < max_turns * 2:
            current_tool = chain[tool_index]
            
            # Assistant Agent: 推理和决策
            assistant_response = self.generate_assistant_response_with_llm(
                conversation, current_tool
            )
            
            if not assistant_response:
                break
            
            # 检查是否需要澄清
            if assistant_response.get('action') == 'ask_clarification':
                clarification = assistant_response.get('clarification', '')
                conversation.append({
                    "role": "assistant",
                    "content": clarification,
                    "reasoning": assistant_response.get('reasoning', []),
                    "turn": len(conversation) + 1
                })
                print(f"Assistant (澄清): {clarification}")
                
                # User Agent: 提供澄清信息
                user_clarification = self._generate_user_clarification(current_tool)
                conversation.append({
                    "role": "user",
                    "content": user_clarification,
                    "turn": len(conversation) + 1
                })
                print(f"User (回复): {user_clarification}")
                continue
            
            # Assistant调用工具
            tool_call = assistant_response.get('tool_call', {})
            reasoning = assistant_response.get('reasoning', [])
            
            conversation.append({
                "role": "assistant",
                "content": f"我将调用 {current_tool} 来帮助您",
                "reasoning": reasoning,
                "tool_call": tool_call,
                "turn": len(conversation) + 1
            })
            print(f"Assistant: 调用工具 {current_tool}")
            print(f"  推理: {reasoning[:2]}")  # 只打印前2步
            
            # Tool Agent: 执行工具并返回结果
            tool_result = self.generate_tool_result(
                current_tool, 
                tool_call.get('parameters', {})
            )
            
            conversation.append({
                "role": "tool",
                "tool_name": current_tool,
                "tool_result": tool_result,
                "turn": len(conversation) + 1
            })
            print(f"Tool: 返回结果 {list(tool_result.keys())[:3]}")
            
            # Assistant Agent: 处理工具结果并响应用户
            summary = self._generate_result_summary(current_tool, tool_result)
            conversation.append({
                "role": "assistant",
                "content": summary,
                "turn": len(conversation) + 1
            })
            print(f"Assistant: {summary[:50]}...")
            
            tool_index += 1
            
            # 如果还有更多工具，可能需要继续
            if tool_index < len(chain):
                # 随机决定是否需要用户确认继续
                if random.random() < 0.3:  # 30%概率需要确认
                    conversation.append({
                        "role": "user",
                        "content": "继续",
                        "turn": len(conversation) + 1
                    })
                    print(f"User: 继续")
        
        # 3. 最终总结
        final_summary = self._generate_final_summary(conversation)
        conversation.append({
            "role": "assistant",
            "content": final_summary,
            "turn": len(conversation) + 1
        })
        print(f"Assistant (总结): {final_summary[:50]}...")
        print(f"{'='*60}\n")
        
        # 提取用户内容（第一个用户查询）
        user_content = conversation[0]['content'] if conversation else ""
        
        return {
            "UserContent": user_content,
            "ToolChain": chain,
            "conversation": conversation,
            "chain_length": len(chain),
            "total_turns": len(conversation),
            "assistant_turns": len([t for t in conversation if t['role'] == 'assistant'])
        }
    
    def _generate_mock_value(self, param_name: str, param_type: str) -> Any:
        """生成模拟参数值"""
        param_lower = param_name.lower()
        
        # 根据参数名称和类型生成合理的模拟值
        if param_type == 'string':
            if 'id' in param_lower:
                return f"mock_{random.randint(1000, 9999)}"
            elif 'name' in param_lower:
                return random.choice(['John Doe', 'Jane Smith', 'Test User'])
            elif 'email' in param_lower:
                return 'user@example.com'
            elif 'url' in param_lower or 'link' in param_lower:
                return 'https://example.com'
            elif 'code' in param_lower:
                return f"CODE{random.randint(100, 999)}"
            else:
                return 'mock_value'
        
        elif param_type == 'int' or param_type == 'float':
            if 'count' in param_lower or 'limit' in param_lower:
                return random.randint(10, 50)
            elif 'page' in param_lower:
                return 1
            else:
                return random.randint(1, 100)
        
        elif param_type == 'bool':
            return random.choice([True, False])
        
        return None
    
    def _generate_user_clarification(self, tool_name: str) -> str:
        """生成用户的澄清回复"""
        tool_info = self.tool_dict.get(tool_name, {})
        params = tool_info.get('parameters', [])
        
        if params:
            required_params = [p for p in params if p.get('required')]
            if required_params:
                param = random.choice(required_params)
                param_name = param['name']
                mock_value = self._generate_mock_value(param_name, param['type'])
                return f"好的，{param_name} 是 {mock_value}"
        
        return "好的，请继续"
    
    def _generate_result_summary(self, tool_name: str, result: Dict) -> str:
        """生成工具结果的摘要"""
        summaries = [
            f"已成功调用 {tool_name}，获取到相关数据。",
            f"好的，我已经通过 {tool_name} 获取了信息。",
            f"完成了 {tool_name} 的调用，结果如下。"
        ]
        return random.choice(summaries)
    
    def _generate_final_summary(self, conversation: List[Dict]) -> str:
        """生成最终总结"""
        tool_calls = [t for t in conversation if t.get('role') == 'tool']
        return f"任务完成！我总共调用了 {len(tool_calls)} 个工具来帮助您完成请求。"

    
    def batch_synthesize(self, num_samples: int = None, output_file: str = None) -> List[Dict]:
        """
        批量合成对话
        
        Args:
            num_samples: 要合成的样本数量，None表示处理所有链
            output_file: 输出文件路径
            
        Returns:
            所有合成的对话列表
        """
        if num_samples is None:
            num_samples = len(self.chains)
        else:
            num_samples = min(num_samples, len(self.chains))
        
        all_dialogues = []
        total_assistant_turns = 0
        
        print(f"\n开始批量合成 {num_samples} 个对话...")
        print(f"{'='*60}\n")
        
        for i, chain_data in enumerate(self.chains[:num_samples]):
            print(f"进度: {i+1}/{num_samples}")
            
            chain = chain_data['chain']
            dialogue = self.synthesize_dialogue(chain)
            all_dialogues.append(dialogue)
            
            total_assistant_turns += dialogue['assistant_turns']
            
            # 每10个保存一次（防止数据丢失）
            if output_file and (i + 1) % 10 == 0:
                self._save_dialogues(all_dialogues, output_file)
                print(f"已保存 {i+1} 个对话到 {output_file}")
        
        # 最终保存
        if output_file:
            self._save_dialogues(all_dialogues, output_file)
        
        print(f"\n{'='*60}")
        print(f"批量合成完成！")
        print(f"总对话数: {len(all_dialogues)}")
        print(f"总Assistant响应轮次: {total_assistant_turns}")
        print(f"平均每个对话的Assistant轮次: {total_assistant_turns/len(all_dialogues):.2f}")
        print(f"{'='*60}\n")
        
        return all_dialogues
    
    def _save_dialogues(self, dialogues: List[Dict], output_file: str):
        """保存对话到文件"""
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    # 配置文件路径
    tools_file = 'data/refined_tools.json'
    chains_file = 'data/sampled_tool_chains.json'
    output_file = 'data/synthesized_dialogues.json'
    
    # 初始化合成器
    print("初始化多Agent对话合成系统...")
    synthesizer = MultiAgentDialogueSynthesis(tools_file, chains_file)
    
    # 批量合成（先测试少量）
    num_samples = 5  # 可以调整数量
    dialogues = synthesizer.batch_synthesize(
        num_samples=num_samples,
        output_file=output_file
    )
    
    print(f"\n对话已保存到: {output_file}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"- 总对话数: {len(dialogues)}")
    print(f"- 平均对话轮数: {sum(d['total_turns'] for d in dialogues) / len(dialogues):.2f}")
    print(f"- 平均工具链长度: {sum(d['chain_length'] for d in dialogues) / len(dialogues):.2f}")


if __name__ == "__main__":
    main()
