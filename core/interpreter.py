import os
import subprocess
import tempfile
from typing import Optional, Tuple
from .journal import MetricValue

class SandboxInterpreter:
    def __init__(self, data_dir: str, timeout: int = 120):
        self.data_dir = data_dir
        self.timeout = timeout

    def run(self, code: str) -> Tuple[bool, Optional[MetricValue]]:
        """
        运行代码，返回：(是否运行成功, 指标值)
        核心：临时目录隔离 + 子进程运行 + 指标解析
        """
        print("开始运行代码...")
        # 1. 创建临时目录（隔离运行环境）
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"创建临时目录：{tmp_dir}")
            # 复制数据集到临时目录（避免修改原数据）
            # 使用 './data/' 目录，与 prompt 中指定的路径一致
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            print(f"创建数据目录：{data_dir}")
            
            # 检查数据目录是否存在
            if not os.path.exists(self.data_dir):
                print(f"错误：数据目录 {self.data_dir} 不存在")
                return False, None
            
            # 复制数据
            print(f"复制数据从 {self.data_dir} 到 {data_dir}")
            os.system(f"cp -r {self.data_dir}/* {data_dir}/")
            
            # 检查复制是否成功
            copied_files = os.listdir(data_dir)
            print(f"复制的文件：{copied_files}")

            # 2. 写入代码文件
            code_path = os.path.join(tmp_dir, "run.py")
            print(f"写入代码到：{code_path}")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # 打印前几行代码以便调试
            print("生成的代码前10行：")
            with open(code_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"{i+1}: {line.rstrip()}")

            # 3. 子进程运行代码（直接执行，不使用沙盒）
            try:
                print("开始执行代码...")
                result = subprocess.run(
                    ["python3", code_path],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                print(f"代码执行完成，返回码：{result.returncode}")
                
                # 4. 解析运行结果
                if result.returncode != 0:
                    print(f"代码运行错误：{result.stderr}")
                    print(f"标准输出：{result.stdout}")
                    return False, None

                # 5. 解析指标（假设代码打印"Metric: 0.123"）
                print(f"解析指标，标准输出：{result.stdout}")
                metric_value = self._parse_metric(result.stdout)
                if metric_value is None:
                    print("未解析到指标值")
                    return False, None

                # 对于分类任务，准确率越高越好
                print(f"解析到指标值：{metric_value}")
                return True, MetricValue(value=metric_value, lower_is_better=False)
            except subprocess.TimeoutExpired:
                print("代码运行超时")
                return False, None
            except Exception as e:
                print(f"运行异常：{e}")
                import traceback
                traceback.print_exc()
                return False, None

    def _parse_metric(self, stdout: str) -> Optional[float]:
        """从代码输出中解析指标值（适配自定义打印格式）"""
        print("开始解析指标...")
        for line in stdout.split("\n"):
            print(f"检查行：{line}")
            if "Metric:" in line:
                try:
                    value = float(line.split("Metric:")[-1].strip())
                    print(f"解析到指标值：{value}")
                    return value
                except ValueError:
                    print(f"无法解析指标值：{line}")
                    return None
        print("未找到包含 'Metric:' 的行")
        return None
