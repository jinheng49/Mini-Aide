from dataclasses import dataclass

# ===================== ✅ 全局统一 LLM 配置（第三方平台） =====================
LLM_GLOBAL_CONFIG = {
    "API_KEY": "sk-HMloF0NeFeZKzaJ3Cb5f2c867b82441cB9B06601D937C661",       # 第三方平台密钥
    "BASE_URL": "https://aihubmix.com/v1",       # 第三方平台接口地址
    "MODEL": "glm-4.7"                     # 平台模型名称
}

# ===================== 全局通用配置 =====================
@dataclass
class GlobalConfig:
    timeout: int = 3600          # 代码沙盒超时时间
    debug_max_depth: int = 3     # 最大调试深度

# ===================== 数据集独立配置基类 =====================
@dataclass
class DatasetConfig:
    data_dir: str                # 数据集路径（每个数据集独立）
    task_desc: str               # 任务描述（独立）
    eval_metric: str             # 评估指标（独立）
    max_steps: int = 15          # 最大迭代步数（独立）