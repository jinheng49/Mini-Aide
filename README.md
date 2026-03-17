# Mini-Aide 项目介绍

## 项目概述

Mini-Aide 是一个基于大语言模型（LLM）的自动机器学习工具，能够自动生成、优化和调试机器学习模型代码，适用于各种 Kaggle 竞赛任务。该项目旨在简化机器学习模型的开发流程，通过自动化的方式提高模型性能和开发效率。

## 项目结构

```
Mini-Aide/
├── core/                # 核心功能模块
│   ├── agent.py         # 智能代理，负责模型生成和优化
│   ├── interpreter.py   # 代码解释器，负责运行和评估模型
│   ├── journal.py       # 日志管理，记录模型历史和性能
│   └── llm_backend.py   # LLM 后端，负责与大语言模型交互
├── examples/            # 示例任务
│   ├── digit_recognizer/                # 手写数字识别
│   ├── house-prices-advanced-regression-techniques/  # 房价预测
│   ├── ieee-fraud-detection/            # 欺诈检测
│   ├── otto_group/                      # 产品分类
│   ├── santander-customer-transaction-prediction/  # 客户交易预测
│   └── titanic/                         # 泰坦尼克号生存预测
├── config.py            # 配置文件
└── README.md            # 项目说明文档
```

## 核心功能

1. **自动模型生成**：通过 LLM 生成完整的机器学习模型代码
2. **模型优化**：基于评估结果自动改进模型性能
3. **错误修复**：自动检测和修复代码中的错误
4. **性能评估**：使用指定的评估指标评估模型性能
5. **结果保存**：自动保存最优模型代码和生成 submission.csv 文件

## 技术架构

### 1. 智能代理（MiniAIDECore）
- **Draft**：生成初始模型代码
- **Improve**：基于现有模型进行改进
- **Debug**：修复代码中的错误
- **SelectNextNode**：选择下一个要处理的模型节点

### 2. 代码解释器（SandboxInterpreter）
- 运行生成的代码
- 评估模型性能
- 捕获和处理错误

### 3. 日志管理（Journal）
- 记录所有生成的模型
- 跟踪模型性能
- 选择最优模型

### 4. LLM 后端
- 与大语言模型交互
- 生成和优化代码
- 处理模型改进和错误修复

## 使用方法

### 运行示例任务

 **进入示例目录**
```bash
cd examples/[task_name]
```

 **运行主脚本**
```bash
python run.py
```

 **查看结果**
- 最优模型代码会保存到 `best_model.py`
- 生成的提交文件会保存到 `submission.csv`

### 配置说明

在 `config.py` 文件中，可以配置以下参数：

- **LLM 配置**：API 密钥、基础 URL 和模型名称
- **全局配置**：代码沙盒超时时间、最大调试深度
- **数据集配置**：数据目录、任务描述、评估指标、最大迭代步数

## 参考资料

本项目参考了以下资料：

1. **原论文**：[AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/pdf/2502.13138)

2. **原论文仓库**：[WecoAI/aideml]( https://github.com/WecoAI/aideml)

3. **李宏毅老师深度学习**：
   - [ML2025spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php)
   - [hw2](https://www.kaggle.com/competitions/ml-2025-spring-hw-2)