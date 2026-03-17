import sys
sys.path.append("../..")

from core.agent import MiniAIDECore
from core.interpreter import SandboxInterpreter
from config import DatasetConfig

def run():
    cfg = DatasetConfig(
        data_dir="./data",
        task_desc="Kaggle Santander客户交易预测，二分类，输出AUC-ROC曲线下面积",
        eval_metric="AUC-ROC",
        max_steps=10
    )

    agent = MiniAIDECore(cfg.task_desc, cfg.eval_metric)
    interpreter = SandboxInterpreter(cfg.data_dir)

    print("===== 运行：Santander Customer Transaction Prediction =====")
    for step in range(cfg.max_steps):
        print(f"\n迭代 {step+1}/{cfg.max_steps}")
        target_node = agent.select_next_node()

        if target_node is None:
            current_node = agent.draft()
        elif target_node.is_buggy:
            current_node = agent.debug(target_node)
        else:
            current_node = agent.improve(target_node)

        success, metric = interpreter.run(current_node.code)
        current_node.is_buggy = not success
        if success:
            current_node.metric = metric  # 关键修复：设置节点的metric
            print(f"✅ 成功 | 指标：{metric.value}")

    # 从所有成功运行的模型中选择指标最好的
    best = agent.journal.get_best_node()
    
    if best:
        print(f"\n【最终结果】最优指标：{best.metric.value}")
        
        # 保存最佳模型代码
        best_code_path = "./best_model.py"
        with open(best_code_path, "w", encoding="utf-8") as f:
            f.write(best.code)
        print(f"【保存】最优模型代码已保存到：{best_code_path}")
        
        # 运行最佳模型生成 submission.csv
        # 生成的代码会使用 './data/' 目录
        print("\n【运行】使用最佳模型生成 submission.csv...")
        import subprocess
        result = subprocess.run(
            ["python3", best_code_path],
            cwd=".",
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ 成功生成 submission.csv")
            print(result.stdout)
        else:
            print(f"❌ 生成 submission.csv 失败")
            print(f"错误信息：{result.stderr}")
            print(f"标准输出：{result.stdout}")
    else:
        print("\n【结果】没有成功运行的模型，无法生成 submission.csv")

if __name__ == "__main__":
    run()
