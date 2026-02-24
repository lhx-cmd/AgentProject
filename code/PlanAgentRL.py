import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 加载环境变量
load_dotenv('code/.env')

class PlanAgentRL:
    """
    Plan智能体系统（强化学习版）
    
    优化方向：
    1. 多源Reward：LLM评分 + 结构奖励 + 效率奖励
    2. 自我修正：低分触发重新规划
    3. 偏好数据生成：用于DPO训练
    """
    
    def __init__(self, plan_model='qwen-max', eval_model='doubao-seed', score_threshold=35):
        """
        初始化Plan智能体
        
        Args:
            plan_model: 规划模型 ('qwen-max', 'gpt-4o', 'deepseek')
            eval_model: 评价模型 ('doubao-seed', 'qwen-turbo', 'gpt-4o-mini')
            score_threshold: 重新规划阈值
        """
        self.plan_model_name = plan_model
        self.eval_model_name = eval_model
        self.score_threshold = score_threshold
        self.lock = threading.Lock()
        
        # 模型配置映射
        self.model_configs = {
            # 规划模型
            'qwen-max': {'provider': 'dashscope', 'model': 'qwen-max-latest'},
            'deepseek': {'provider': 'dashscope', 'model': 'deepseek-v3.2'},
            'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o'},
            
            # 评价模型
            'doubao-seed': {'provider': 'doubao', 'model': 'doubao-seed-2-0-pro-260215'},
            'qwen-turbo': {'provider': 'dashscope', 'model': 'qwen-turbo'},
            'gpt-4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
        }
        
        # 初始化客户端
        self.dashscope = None
        self.openai_client = None
        self.doubao_client = None
        
        # 根据需要初始化对应的客户端
        plan_provider = self.model_configs[plan_model]['provider']
        eval_provider = self.model_configs[eval_model]['provider']
        
        if plan_provider == 'dashscope' or eval_provider == 'dashscope':
            try:
                import dashscope
                dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
                self.dashscope = dashscope
            except ImportError:
                print("警告：未安装 dashscope 库")
        
        if plan_provider == 'openai' or eval_provider == 'openai':
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                print(f"初始化OpenAI失败: {e}")
        
        if eval_provider == 'doubao':
            try:
                from openai import OpenAI
                ark_api_key = os.getenv('ARK_API_KEY')
                if ark_api_key:
                    self.doubao_client = OpenAI(
                        api_key=ark_api_key,
                        base_url="https://ark.cn-beijing.volces.com/api/v3"
                    )
                else:
                    print("警告：未找到 ARK_API_KEY")
            except Exception as e:
                print(f"初始化豆包失败: {e}")
        
        print(f"✓ 规划模型: {self.model_configs[plan_model]['model']}")
        print(f"✓ 评价模型: {self.model_configs[eval_model]['model']}")
    
    def call_llm(self, messages, temperature=0.7, system_prompt=None, use_eval_model=False):
        """
        统一的LLM调用接口
        
        Args:
            use_eval_model: True使用评价模型，False使用规划模型
        """
        model_name = self.eval_model_name if use_eval_model else self.plan_model_name
        config = self.model_configs[model_name]
        provider = config['provider']
        model = config['model']
        
        try:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            if provider == 'dashscope':
                from dashscope import Generation
                response = Generation.call(
                    model=model,
                    messages=messages,
                    result_format='message',
                    temperature=temperature
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
                else:
                    print(f"API调用失败: {response.code} - {response.message}")
                    return None
            
            elif provider == 'doubao':
                response = self.doubao_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            elif provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLM调用失败 (model={model}): {e}")
            return None
    
    def generate_user_question(self, scenario, tools_dict):
        """生成用户问题"""
        tools_items = list(tools_dict.items())[:3]
        tools_list = "\n".join([f"- {name}" for name, _ in tools_items])
        
        prompt = f"""场景：{scenario}
工具：{tools_list}

生成一个自然的用户问题（20-60字）："""
        
        result = self.call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        return result.strip('"\'') if result else f"关于{scenario}的问题"
    
    def generate_plan(self, user_question, tools_dict, temperature=0.7, previous_feedback=None):
        """
        生成规划（支持自我修正）
        
        Args:
            previous_feedback: 上一次规划的反馈，用于改进
        """
        tools_list = "\n".join([f"- {name}: {desc[:60]}..." for name, desc in list(tools_dict.items())[:5]])
        
        feedback_prompt = ""
        if previous_feedback:
            feedback_prompt = f"\n\n上一次规划的问题：\n{previous_feedback}\n请改进这些问题。"
        
        prompt = f"""用户问题：{user_question}

可用工具：
{tools_list}{feedback_prompt}

生成JSON格式的执行计划：
{{
  "fixed_question": "标准化问题描述",
  "thought": "规划思路：1)需求分解 2)工具选择 3)依赖关系",
  "steps": [
    {{
      "thought": "本步骤的详细思考",
      "title": "步骤标题（简短清晰，如：查询天气）",
      "content": "步骤详细描述，如依赖前序步骤用【步骤标题】引用",
      "tools": ["工具名称"] 或 null,
      "dependencies": ["前序步骤的标题"] 或 null
    }}
  ]
}}

关键规则：
1. dependencies必须是步骤标题的字符串数组，如["查询天气", "预订酒店"]，绝对不能是数字
2. 如果步骤独立无依赖，dependencies设为null
3. 步骤标题要简短（2-8字），便于引用
4. 步骤数量2-5个为宜

只返回JSON："""
        
        result = self.call_llm(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            system_prompt="你是任务规划专家。输出严格的JSON格式，dependencies必须是步骤标题字符串数组或null，不能是数字。"
        )
        
        if result:
            try:
                # 提取JSON
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0].strip()
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0].strip()
                
                plan = json.loads(result)
                
                # 验证并修复dependencies格式
                if 'steps' in plan:
                    for step in plan['steps']:
                        deps = step.get('dependencies')
                        if deps is not None:
                            # 如果不是列表，转换为null
                            if not isinstance(deps, list):
                                step['dependencies'] = None
                            else:
                                # 确保所有元素都是字符串，过滤数字
                                valid_deps = []
                                for d in deps:
                                    if isinstance(d, str) and d.strip():
                                        valid_deps.append(d.strip())
                                step['dependencies'] = valid_deps if valid_deps else null
                
                return plan
            except Exception as e:
                print(f"解析规划失败: {e}")
                return None
        return None
    
    def evaluate_plan_multi_reward(self, user_question, plan):
        """
        多源奖励评价（使用不同的评价模型）
        
        Reward来源：
        1. LLM评分（质量）- 使用eval_model避免自恋
        2. 结构奖励（合理性）
        3. 效率奖励（Token消耗）
        """
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        
        # 1. LLM评分（使用评价模型）
        prompt = f"""你是一个严格的任务规划评审专家。请客观评价以下规划方案。

用户问题：
{user_question}

规划方案：
{plan_json}

请从5个维度进行评分（每项0-10分）：

1. completeness（完整性）：规划是否完整覆盖用户的所有需求？
2. rationality（合理性）：步骤分解是否合理、逻辑是否清晰？
3. tool_usage（工具使用）：工具选择是否恰当、是否充分利用可用工具？
4. dependencies（依赖关系）：步骤间依赖关系是否正确？dependencies必须是步骤标题字符串数组，不能是数字
5. executability（可执行性）：规划是否可以实际执行？

评分标准：
- 9-10分：优秀，几乎完美
- 7-8分：良好，有小瑕疵
- 5-6分：及格，有明显问题
- 3-4分：较差，问题较多
- 0-2分：很差，严重问题

返回JSON格式（严格按照格式）：
{{
  "completeness": {{"score": 0-10, "reason": "具体理由"}},
  "rationality": {{"score": 0-10, "reason": "具体理由"}},
  "tool_usage": {{"score": 0-10, "reason": "具体理由"}},
  "dependencies": {{"score": 0-10, "reason": "具体理由"}},
  "executability": {{"score": 0-10, "reason": "具体理由"}},
  "total_score": 42,
  "overall_comment": "总体评价",
  "improvement_suggestions": "改进建议"
}}

注意：
1. 每个维度的score必须是0-10的数字
2. total_score是5个维度的总和
3. 请客观评分，不要都给高分或低分"""
        
        result = self.call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            system_prompt="你是严格、客观的规划评审专家。请根据实际质量给出差异化的评分，不要所有规划都给相同分数。",
            use_eval_model=True  # 使用评价模型
        )
        
        evaluation = None
        if result:
            try:
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0].strip()
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0].strip()
                evaluation = json.loads(result)
                
                # 验证评分
                if 'total_score' not in evaluation:
                    # 如果没有total_score，手动计算
                    total = 0
                    for key in ['completeness', 'rationality', 'tool_usage', 'dependencies', 'executability']:
                        if key in evaluation and 'score' in evaluation[key]:
                            total += evaluation[key]['score']
                    evaluation['total_score'] = total
                
            except Exception as e:
                print(f"解析评价结果失败: {e}")
                print(f"原始结果: {result[:200]}...")
                return None
        
        if not evaluation:
            return None
        
        # 2. 结构奖励
        structure_score = self._calculate_structure_reward(plan)
        
        # 3. 效率奖励
        efficiency_score = self._calculate_efficiency_reward(plan_json)
        
        # 综合奖励（加权）
        evaluation['structure_reward'] = round(structure_score, 2)
        evaluation['efficiency_reward'] = round(efficiency_score, 2)
        evaluation['final_score'] = round(
            evaluation['total_score'] * 0.7 +  # LLM评分 70%
            structure_score * 0.2 +             # 结构奖励 20%
            efficiency_score * 0.1,             # 效率奖励 10%
            2
        )
        
        return evaluation
    
    def _calculate_structure_reward(self, plan):
        """计算结构奖励"""
        score = 10.0
        
        steps = plan.get('steps', [])
        num_steps = len(steps)
        
        # 步骤数量（2-5最佳）
        if num_steps < 2:
            score -= 3
        elif num_steps > 5:
            score -= (num_steps - 5) * 0.5
        
        # 依赖关系正确性
        step_titles = [s.get('title', '') for s in steps]
        for step in steps:
            deps = step.get('dependencies')
            if deps:
                for dep in deps:
                    # 数字依赖严重扣分
                    if isinstance(dep, (int, float)):
                        score -= 3
                        print(f"  警告：发现数字依赖 {dep}")
                    # 无效引用扣分
                    elif dep not in step_titles:
                        score -= 1
                        print(f"  警告：无效依赖引用 {dep}")
        
        # 标题清晰度
        for step in steps:
            title = step.get('title', '')
            if len(title) < 2 or len(title) > 15:
                score -= 0.5
        
        return max(0, min(10, round(score, 2)))
    
    def _calculate_efficiency_reward(self, plan_json):
        """计算效率奖励（基于Token消耗）"""
        json_length = len(plan_json)
        
        if json_length < 500:
            score = 10
        elif json_length < 1000:
            score = 9
        elif json_length < 1500:
            score = 8
        elif json_length < 2000:
            score = 7
        else:
            score = max(5, 10 - (json_length - 2000) / 500)
        
        return round(score, 2)
    
    def generate_plan_with_retry(self, user_question, tools_dict, max_retries=2):
        """
        自我修正：低分触发重新规划
        
        Args:
            max_retries: 最大重试次数
        
        Returns:
            best_plan, best_evaluation
        """
        attempts = []
        
        for attempt in range(max_retries + 1):
            # 生成规划
            if attempt == 0:
                plan = self.generate_plan(user_question, tools_dict, temperature=0.7)
            else:
                # 使用上一次的反馈改进
                feedback = attempts[-1]['evaluation'].get('improvement_suggestions', '')
                plan = self.generate_plan(user_question, tools_dict, temperature=0.7 + attempt * 0.1, previous_feedback=feedback)
            
            if not plan:
                continue
            
            # 评价规划
            evaluation = self.evaluate_plan_multi_reward(user_question, plan)
            
            if not evaluation:
                continue
            
            attempts.append({
                'plan': plan,
                'evaluation': evaluation,
                'attempt': attempt
            })
            
            # 如果达到阈值，停止重试
            if evaluation['final_score'] >= self.score_threshold:
                break
        
        if not attempts:
            return None, None
        
        # 返回最佳规划
        best = max(attempts, key=lambda x: x['evaluation']['final_score'])
        return best['plan'], best['evaluation']
    
    def generate_preference_data(self, user_question, tools_dict, num_samples=3):
        """生成偏好数据（用于DPO训练）"""
        samples = []
        
        # 并发生成多个样本
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(num_samples):
                temperature = 0.7 + i * 0.1
                future = executor.submit(self.generate_plan, user_question, tools_dict, temperature)
                futures.append((future, temperature))
            
            for future, temp in futures:
                plan = future.result()
                if plan:
                    evaluation = self.evaluate_plan_multi_reward(user_question, plan)
                    if evaluation:
                        samples.append({
                            'plan': plan,
                            'evaluation': evaluation,
                            'temperature': temp
                        })
        
        if not samples:
            return None
        
        # 按final_score排序
        samples.sort(key=lambda x: x['evaluation']['final_score'], reverse=True)
        
        # 构建偏好数据
        preference_data = {
            "user_question": user_question,
            "samples": samples,
            "best_plan": samples[0]['plan'],
            "best_score": samples[0]['evaluation']['final_score'],
            "worst_plan": samples[-1]['plan'],
            "worst_score": samples[-1]['evaluation']['final_score'],
            "preference_pairs": []
        }
        
        # 生成偏好对（用于DPO）
        for i in range(1, len(samples)):
            preference_data['preference_pairs'].append({
                "chosen": samples[0]['plan'],
                "chosen_score": samples[0]['evaluation']['final_score'],
                "rejected": samples[i]['plan'],
                "rejected_score": samples[i]['evaluation']['final_score'],
                "score_diff": samples[0]['evaluation']['final_score'] - samples[i]['evaluation']['final_score'],
                "reward_breakdown": {
                    "chosen": {
                        "llm_score": samples[0]['evaluation']['total_score'],
                        "structure": samples[0]['evaluation']['structure_reward'],
                        "efficiency": samples[0]['evaluation']['efficiency_reward']
                    },
                    "rejected": {
                        "llm_score": samples[i]['evaluation']['total_score'],
                        "structure": samples[i]['evaluation']['structure_reward'],
                        "efficiency": samples[i]['evaluation']['efficiency_reward']
                    }
                }
            })
        
        return preference_data

def process_one_scenario(system, idx, scenario_data):
    """处理单个场景"""
    scenario = scenario_data['scenario']
    tools_dict = scenario_data['tools']
    
    try:
        print(f"\n{'='*60}")
        print(f"场景 {idx}: {scenario}")
        print(f"{'='*60}")
        
        # 生成用户问题
        user_question = system.generate_user_question(scenario, tools_dict)
        print(f"用户问题: {user_question}")
        
        # 生成规划并评价（带自我修正）
        preference_data = system.generate_preference_data(user_question, tools_dict, num_samples=3)
        
        if preference_data:
            print(f"\n样本评分:")
            for i, sample in enumerate(preference_data['samples'], 1):
                eval_data = sample['evaluation']
                print(f"  样本{i}: LLM={eval_data['total_score']}, "
                      f"结构={eval_data['structure_reward']}, "
                      f"效率={eval_data['efficiency_reward']}, "
                      f"最终={eval_data['final_score']}")
            
            return {
                "scenario_id": idx,
                "scenario": scenario,
                "tools": tools_dict,
                "tools_length": len(tools_dict),
                "user_question": user_question,
                "preference_data": preference_data
            }
    except Exception as e:
        print(f"场景 {idx} 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def process_all_scenarios(input_file, output_dir, max_scenarios=10):
    """处理所有场景"""
    with open(input_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    scenarios = scenarios[:max_scenarios]
    print(f"✓ 已加载 {len(scenarios)} 个场景\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    system = PlanAgentRL(plan_model='qwen-max', eval_model='doubao-seed', score_threshold=35)
    
    all_results = []
    
    for idx, scenario_data in enumerate(scenarios, 1):
        result = process_one_scenario(system, idx, scenario_data)
        if result:
            all_results.append(result)
        
        time.sleep(0.5)  # 避免API限流
    
    # 保存结果
    output_file = os.path.join(output_dir, 'plan_preference_data_rl.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ 完成！处理了 {len(all_results)}/{len(scenarios)} 个场景")
    print(f"✓ 结果已保存至: {output_file}")
    
    if all_results:
        total_samples = sum(len(r['preference_data']['samples']) for r in all_results)
        total_pairs = sum(len(r['preference_data']['preference_pairs']) for r in all_results)
        avg_best_score = sum(r['preference_data']['best_score'] for r in all_results) / len(all_results)
        
        # 统计得分分布
        all_scores = []
        for r in all_results:
            for sample in r['preference_data']['samples']:
                all_scores.append(sample['evaluation']['final_score'])
        
        print(f"\n统计信息:")
        print(f"  - 总场景数: {len(all_results)}")
        print(f"  - 总规划样本数: {total_samples}")
        print(f"  - 总偏好对数: {total_pairs}")
        print(f"  - 平均最佳得分: {avg_best_score:.2f}")
        print(f"  - 得分范围: {min(all_scores):.2f} - {max(all_scores):.2f}")
        print(f"  - 得分标准差: {(sum((s - avg_best_score)**2 for s in all_scores) / len(all_scores))**0.5:.2f}")
    print(f"{'='*60}")

def main():
    input_file = 'data/tool_chains_filtered_cluster.json'
    output_dir = 'data/plan_results'
    max_scenarios = 10
    
    print("="*60)
    print("Plan智能体系统（强化学习版）")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"处理场景数: {max_scenarios}")
    print(f"评分模式: 多源Reward (LLM + 结构 + 效率)")
    print("="*60 + "\n")
    
    start_time = time.time()
    process_all_scenarios(input_file, output_dir, max_scenarios)
    elapsed_time = time.time() - start_time
    
    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print(f"平均每场景: {elapsed_time/max_scenarios:.2f}秒")

if __name__ == '__main__':
    main()
