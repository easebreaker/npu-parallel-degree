"""CV 流水结构对 CP 设计施加的硬约束。

这些约束**不依赖 b(N) 的拟合**——它们是纯结构的：
任务数（tile 对）与核数的比较。

三个上限：

1. 核数：`ceil(L/(N·BM)) × ceil(L/BN) ≥ AIC`
2. 流水深度：任务数 ≥ num_stages
3. MTE2 下界：`T(∞) = 4BhLd·s/BW`，故上界 `L/L*`

对应文档 `CV流水-对CP设计的约束.md`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 910B3 的典型值。实际应以 GetCoreNumAic() / GetCoreNumAiv() 取。
DEFAULT_AIC = 20
DEFAULT_BM = 128
DEFAULT_BN = 128


@dataclass(frozen=True)
class CVConfig:
    """硬件与 tile 配置。"""

    aic: int = DEFAULT_AIC
    bm: int = DEFAULT_BM
    bn: int = DEFAULT_BN
    has_l0c_to_ub: bool = False

    def __post_init__(self) -> None:
        if self.aic < 1 or self.bm < 16 or self.bn < 16:
            raise ValueError("核数与 tile 尺寸要为正（tile 至少 16，对应分形块）")
        if self.bm % 16 or self.bn % 16:
            raise ValueError("tile 尺寸应为 16 的倍数（Cube 的分形块约束）")

    def transfer_per_task(self) -> float:
        """一个 CV 任务的中转字节数（fp32 的 QK 结果）。

        `DAV_2201` 上没有 L0C→UB 直通，所以每个 QK tile 都要经 fixpipe
        搬到 GM 或 L1，再从那里到 UB。这是「每 tile 一次」的成本。
        """
        return self.bm * self.bn * 4.0


def tasks(length: float, degree: int, config: CVConfig, causal: bool = False) -> float:
    """CV 任务数：Q tile 数 × KV tile 数。

    CP 只切 Q，所以 `degree` 只影响第一个因子，KV 侧不动。
    `causal=True` 时只算下三角，约 `Q²/2`。
    """
    if degree < 1:
        raise ValueError("并行度至少 1")
    q = math.ceil(length / degree / config.bm)
    kv = math.ceil(length / config.bn)
    if causal:
        return q * (q + 1) / 2.0
    return float(q * kv)


def core_utilisation(length: float, degree: int, config: CVConfig, causal: bool = False) -> float:
    """核利用率上界：任务数 / 核数，封顶 1。"""
    return min(1.0, tasks(length, degree, config, causal) / config.aic)


def max_degree(length: float, config: CVConfig, causal: bool = False) -> float:
    """由「任务数 ≥ 核数」解出的并行度上限。

    非因果：`(L/(N·BM))·(L/BN) ≥ AIC`  =>  `N ≤ L²/(BM·BN·AIC)`
    因果：  `(L/(N·BM))²/2 ≥ AIC`       =>  `N ≤ L/(BM·√(2·AIC))`
    """
    if causal:
        return length / (config.bm * math.sqrt(2.0 * config.aic))
    return length * length / (config.bm * config.bn * config.aic)


def recommended_degree(length: float, config: CVConfig, causal: bool = True, cap: int = 8) -> int:
    """在给定卡数上限内，任务数够用的最大 2 的幂。

    返回 1 表示「不切」——包括「连 N=1 都不满核」的情形。
    """
    limit = min(max_degree(length, config, causal), float(cap))
    degree = 1
    while degree * 2 <= limit:
        degree *= 2
    return degree


def max_tile_for_cores(length: float, degree: int, config: CVConfig, causal: bool = False) -> int:
    """在给定 L 与 N 下，能让任务数 ≥ 核数的最大 tile 边长（16 的倍数）。

    这是「任务数不够时先缩 tile 而非减 N」这条建议的量化形式。
    """
    if causal:
        need = math.sqrt(2.0 * config.aic) * degree
    else:
        need = math.sqrt(config.aic * degree)
    tile = int(length / need)
    tile -= tile % 16
    return max(16, tile)


def total_transfer(length: float, degree: int, config: CVConfig, causal: bool = False) -> float:
    """全部任务的中转总字节数。

    在 `DAV_2201` 上每个 QK tile 都要搬一次；`has_l0c_to_ub=True` 时
    这条成本不进 GM，但仍占用 L0C->UB 的通道。
    """
    return tasks(length, degree, config, causal) * config.transfer_per_task()


@dataclass(frozen=True)
class PipelineFit:
    """CV 流水能否填满。"""

    degrees: tuple[int, ...]
    utilisation: tuple[float, ...]
    blocking: bool

    def summary(self) -> str:
        cells = "  ".join(
            f"N={d}: {u * 100:3.0f}%" for d, u in zip(self.degrees, self.utilisation)
        )
        tail = "任务数不足，CV 退化为串行" if self.blocking else "可重叠"
        return f"{cells}   {tail}"


def pipeline_fit(length: float, config: CVConfig, num_stages: int = 2, causal: bool = False) -> PipelineFit:
    """CV 流水深度是否够。

    需要「任务数 ≥ num_stages」才有重叠空间；否则 Cube 与 Vector 退化为串行，
    而 CrossCore 同步开销照付。
    """
    degrees = (1, 2, 4, 8)
    utils = tuple(core_utilisation(length, d, config, causal) for d in degrees)
    blocking = tasks(length, min(degrees), config, causal) < num_stages
    return PipelineFit(degrees, utils, blocking)


def degree_table(lengths: tuple[int, ...], config: CVConfig, causal: bool = True) -> list[tuple[int, float, int]]:
    """L -> (N 上限, 建议档位) 的表，供文档与报告引用。"""
    return [
        (length, max_degree(length, config, causal), recommended_degree(length, config, causal))
        for length in lengths
    ]
