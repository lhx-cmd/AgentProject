import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 加载环境变量
load_dotenv('code/.env')

class PlanAgentSystemOptimized:
    """
    Plan智能体系统（优化版）
    - 并发处理提升速度
    - 简化prompt减少token消耗
    - 批量处理减少API调用
    """
    
    def __init__(self, use_dashscope=True):
        self.use_dashscope = use_dashscope
        self.client = None
        self.lock = threading.Lock()
        
        if use_dashscope:
            try:
                import dashscope
                dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
                self.dashscope = dashscope
                print("✓ 使用阿里云通义千问 (qwen-max-latest)")
            except ImportError:
                print("警告：未安装 dashscope 库")
                self.use_dashscope = False
        
        if not self.use_dashscope:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                print("✓ 使用OpenAI (gpt-4o)")
            except Exception as e:
                print(f"初始化失败: {e}")
    
    def call_llm(self, messages, temperature=0.7, system_prompt=None):
        """统一的LLM调用接口（使用qwen-max-latest）"""
        try:
            if self.use_dashscope:
                from dashscope import Generation
                
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                
                response = Generation.call(
                    model='qwen-max-latest',  # 使用最好的模型
                    messages=messages,
                    result_format='message',
                    temperature=temperature
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
            else:
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # OpenAI也使用最好的模型
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None
    
    def generate_user_question(self, scenario, tools_dict):
        """生成用户问题（简化版）"""
        # 只显示前3个工具
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
    
    def generate_plan(self, user_question, tools_dict, temperature=0.7):
        """生成规划（改进版 - 强调dependencies格式）"""
        tools_list = "\n".join([f"- {name}: {desc[:60]}..." for name, desc in list(tools_dict.items())[:5]])
        
        prompt = f"""用户问题：{user_question}

可用工具：
{tools_list}

生成JSON格式的执行计划：
{{
  "fixed_question": "标准化问题描述",
  "thought": "规划思路：1)需求分解 2)工具选择 3)依赖关系",
  "steps": [
    {{
      "thought": "本步骤的详细思考",
      "title": "步骤标题（简短清晰）",
      "content": "步骤详细描述，如依赖前序步骤用【步骤标题】引用",
      "tools": ["工具名称"] 或 null,
      "dependencies": ["前序步骤的标题"] 或 null
    }}
  ]
}}

重要规则：
1. dependencies必须是步骤标题的数组，如["查询天气", "预订酒店"]，不能是数字
2. 如果步骤独立，dependencies设为null
3. 步骤标题要简短清晰，便于引用
4. 步骤数量2-5个为宜

只返回JSON："""
        
        result = self.call_llm(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            system_prompt="你是任务规划专家。输出严格的JSON格式，dependencies必须是步骤标题数组或null。"
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
                            # 如果是数字或其他非法格式，转换为null
                            if not isinstance(deps, list):
                                step['dependencies'] = None
                            else:
                                # 确保所有元素都是字符串
                                step['dependencies'] = [str(d) for d in deps if d]
                                if not step['dependencies']:
                                    step['dependencies'] = None
                
                return plan
            except Exception as e:
                print(f"解析规划失败: {e}")
                return None
        return None
    
    def evaluate_plan(self, user_question, plan):
        """评价规划（简化版）"""
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        
        prompt = f"""评价以下规划（0-10分）：

问题：{user_question}
规划：{plan_json}

返回JSON：
{{
  "completeness": {{"score": "分数", "reason": "理由"}},
  "rationality": {{"score": "分数", "reason": "理由"}},
  "tool_usage": {{"score": "分数", "reason": "理由"}},
  "dependencies": {{"score": "分数", "reason": "理由"}},
  "executability": {{"score": "分数", "reason": "理由"}},
  "total_score": "分数",
  "overall_comment": "总评"
}}"""
        
        result = self.call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            system_prompt="你是规划评审专家，输出JSON格式。"
        )
        
        if result:
            try:
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0].strip()
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0].strip()
                return json.loads(result)
            except:
                return None
        return None
    
    def generate_one_sample(self, user_question, tools_dict, temperature):
        """生成单个样本（用于并发）"""
        plan = self.generate_plan(user_question, tools_dict, temperature)
        if plan:
            evaluation = self.evaluate_plan(user_question, plan)
            if evaluation:
                return {
                    "plan": plan,
                    "evaluation": evaluation,
                    "temperature": temperature
                }
        return None
    
    def generate_preference_data(self, user_question, tools_dict, num_samples=3):
        """并发生成多个样本"""
        samples = []
        temperatures = [0.7 + i * 0.1 for i in range(num_samples)]
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.generate_one_sample, user_question, tools_dict, temp): temp
                for temp in temperatures
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    samples.append(result)
        
        if not samples:
            return None
        
        # 按总分排序
        samples.sort(key=lambda x: x['evaluation']['total_score'], reverse=True)
        
        # 构建偏好数据
        preference_data = {
            "user_question": user_question,
            "samples": samples,
            "best_plan": samples[0]['plan'],
            "best_score": samples[0]['evaluation']['total_score'],
            "worst_plan": samples[-1]['plan'] if len(samples) > 1 else samples[0]['plan'],
            "worst_score": samples[-1]['evaluation']['total_score'] if len(samples) > 1 else samples[0]['evaluation']['total_score'],
            "preference_pairs": []
        }
        
        # 生成偏好对
        for i in range(1, len(samples)):
            preference_data['preference_pairs'].append({
                "chosen": samples[0]['plan'],
                "chosen_score": samples[0]['evaluation']['total_score'],
                "rejected": samples[i]['plan'],
                "rejected_score": samples[i]['evaluation']['total_score'],
                "score_diff": samples[0]['evaluation']['total_score'] - samples[i]['evaluation']['total_score']
            })
        
        return preference_data

def process_one_scenario(system, idx, scenario_data):
    """处理单个场景（用于并发）"""
    scenario = scenario_data['scenario']
    tools_dict = scenario_data['tools']
    
    try:
        # 生成用户问题
        user_question = system.generate_user_question(scenario, tools_dict)
        
        # 生成规划并评价
        preference_data = system.generate_preference_data(user_question, tools_dict, num_samples=3)
        
        if preference_data:
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
    
    return None

def process_all_scenarios(input_file, output_dir, max_scenarios=10, num_samples=3):
    """
    处理所有场景（优化版）
    """
    # 加载场景数据
    with open(input_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    # 限制场景数量
    scenarios = scenarios[:max_scenarios]
    
    print(f"✓ 已加载 {len(scenarios)} 个场景")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化系统
    system = PlanAgentSystemOptimized(use_dashscope=True)
    
    all_results = []
    
    # 顺序处理（避免API限流）
    for idx, scenario_data in enumerate(tqdm(scenarios, desc="处理场景"), 1):
        result = process_one_scenario(system, idx, scenario_data)
        if result:
            all_results.append(result)
            print(f"  ✓ 场景 {idx}: {result['scenario'][:30]}... | 得分: {result['preference_data']['best_score']}")
        
        # 短暂延迟避免限流
        time.sleep(0.3)
    
    # 保存结果
    output_file = os.path.join(output_dir, 'plan_preference_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ 完成！处理了 {len(all_results)}/{len(scenarios)} 个场景")
    print(f"✓ 结果已保存至: {output_file}")
    
    # 统计信息
    if all_results:
        total_samples = sum(len(r['preference_data']['samples']) for r in all_results)
        total_pairs = sum(len(r['preference_data']['preference_pairs']) for r in all_results)
        avg_best_score = sum(r['preference_data']['best_score'] for r in all_results) / len(all_results)
        
        print(f"\n统计信息:")
        print(f"  - 总场景数: {len(all_results)}")
        print(f"  - 总规划样本数: {total_samples}")
        print(f"  - 总偏好对数: {total_pairs}")
        print(f"  - 平均最佳得分: {avg_best_score:.2f}")
    print(f"{'='*60}")

def main():
    """主函数"""
    input_file = 'data/tool_chains_with_scenarios.json'
    output_dir = 'data/plan_results'
    max_scenarios = 10  # 先处理10个场景测试
    
    print("="*60)
    print("Plan智能体系统（优化版）")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"处理场景数: {max_scenarios}")
    print(f"每场景采样: 3次")
    print("="*60 + "\n")
    
    start_time = time.time()
    process_all_scenarios(input_file, output_dir, max_scenarios)
    elapsed_time = time.time() - start_time
    
    print(f"\n总耗时: {elapsed_time:.2f}秒")
    print(f"平均每场景: {elapsed_time/max_scenarios:.2f}秒")

if __name__ == '__main__':
    main()
