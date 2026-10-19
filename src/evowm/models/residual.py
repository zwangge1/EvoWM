"""预测变化量（设计 docs/design/预测变化量_重训_2026-10-19.md）。

每一层的预测器从"直接给出下一个的表示"改成"前一个的表示 + 预测的变化量"：上游 JEPA.predict 在模型对象的
residual_predict 为真时，把输入的表示加到预测器输出上（改动登记在 hjepa/UPSTREAM.md）。这里只做两件事：
打开这个开关，并把预测器投影头的最后一个线性层置 0，使训练开始时模型就等于照抄表示。
保存的模型对象里仍然只有上游的类（多一个布尔属性）；没有这个属性的旧模型行为不变。
训练入口：配置里 evowm_residual: true 时，上游 main_hjepa.py 建好模型后调用 enable_residual。
"""
from __future__ import annotations

from torch import nn


def _last_linear(module: nn.Module) -> nn.Linear:
    lin = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not lin:
        raise ValueError("预测器的投影头里没有线性层，无法置 0")
    return lin[-1]


def enable_residual(model) -> list[int]:
    """对每一层打开 residual_predict，并把预测器投影头的最后一个线性层（权重与偏置）置 0。返回处理过的层号。"""
    levels = []
    for level in range(1, int(model.num_levels) + 1):
        jepa = model.get_level(level)
        head = getattr(jepa.predictor, "projector", None)
        if head is None or isinstance(head, nn.Identity):
            raise ValueError(f"第 {level} 层的预测器没有投影头（需要 projector.type: mlp）")
        lin = _last_linear(head)
        nn.init.zeros_(lin.weight)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)
        jepa.residual_predict = True
        levels.append(level)
    return levels


def is_residual(model) -> bool:
    """模型的每一层是否都在预测变化量（旧模型为 False）。"""
    return all(bool(getattr(model.get_level(k), "residual_predict", False)) for k in range(1, int(model.num_levels) + 1))
