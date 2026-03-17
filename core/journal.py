from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time

@dataclass
class MetricValue:
    """评估指标值（封装指标方向：越大越好/越小越好）"""
    value: float
    lower_is_better: bool = True  # 如RMSE越小越好，AUROC越大越好

    def is_better_than(self, other: "MetricValue") -> bool:
        if self.lower_is_better:
            return self.value < other.value
        return self.value > other.value

@dataclass
class Node:
    """Solution Tree 节点：每个节点对应一段生成的代码+运行状态"""
    code: str
    plan: str  # LLM 生成的建模思路（自然语言）
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    metric: Optional["MetricValue"] = None  # 代码运行后的评估指标
    is_buggy: bool = False  # 代码是否运行出错
    debug_depth: int = 0    # 调试迭代深度
    create_time: float = field(default_factory=time.time)

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)

class Journal:
    """状态机：追踪所有节点，管理 Solution Tree 生命周期"""
    def __init__(self):
        self.draft_nodes: List[Node] = []  # 初始草稿节点
        self.good_nodes: List[Node] = []   # 运行成功的节点
        self.buggy_nodes: List[Node] = []  # 运行出错的节点

    def add_node(self, node: Node):
        """添加节点并更新状态分类"""
        print(f"【Journal】添加节点 - is_buggy: {node.is_buggy}, metric: {node.metric}")
        if node.is_buggy:
            self.buggy_nodes.append(node)
            print(f"【Journal】添加到 buggy_nodes，总数: {len(self.buggy_nodes)}")
        else:
            self.good_nodes.append(node)
            print(f"【Journal】添加到 good_nodes，总数: {len(self.good_nodes)}")
        if node.parent is None:  # 根节点（草稿）
            self.draft_nodes.append(node)
            print(f"【Journal】添加到 draft_nodes，总数: {len(self.draft_nodes)}")

    def get_best_node(self) -> Optional[Node]:
        """获取当前最优节点（基于指标）"""
        print(f"【Journal】查找最佳节点 - good_nodes: {len(self.good_nodes)}")
        # 过滤出有metric的节点
        nodes_with_metric = [node for node in self.good_nodes if node.metric is not None]
        print(f"【Journal】有metric的节点数: {len(nodes_with_metric)}")
        if not nodes_with_metric:
            print(f"【Journal】没有找到有metric的节点")
            return None
        best = nodes_with_metric[0]
        for node in nodes_with_metric[1:]:
            if node.metric.is_better_than(best.metric):
                best = node
        print(f"【Journal】最佳节点指标: {best.metric.value}")
        return best

    def generate_summary(self) -> str:
        """生成节点摘要（供LLM迭代时参考历史）"""
        best_node = self.get_best_node()
        if not best_node:
            return "No valid solutions yet."
        return (
            f"Best metric so far: {best_node.metric.value} (lower_is_better={best_node.metric.lower_is_better})\n"
            f"Best solution plan: {best_node.plan[:100]}..."
        )
