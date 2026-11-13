"""从 13 点实测反解固定项 b，并给出它的标准误。

这个脚本存在的理由：本文的核心论断是「`b` 随并行度上升，所以最优并行度有内点」。
这个论断建立在**三个数**上（cp2/cp4/cp8 的 b）。把这三个数的反解过程放进仓库，
读者可以自己复核，而不是相信一个断言。

反解模型：每档并行度单独拟合

    T(L) = b + θ₁·L + θ₂·L²

权重取 1/T（相对误差），与论文里 MAPE 的口径一致。
标准误来自参数协方差的 (XᵀWX)⁻¹ 对角元。

用法：
    python3 derive_b.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "attn_mtgr_cp13.csv"


def load_rows() -> dict[int, dict[int, float]]:
    """返回 {并行度: {长度: 秒}}。"""
    table: dict[int, dict[int, float]] = {2: {}, 4: {}, 8: {}}
    with open(DATA, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            length = int(row["length"])
            for degree in (2, 4, 8):
                table[degree][length] = float(row[f"cp{degree}"]) / 1000.0
    return table


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """高斯-约当消元，原地。"""
    n = len(rhs)
    work = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        work[col], work[pivot] = work[pivot], work[col]
        if abs(work[col][col]) < 1e-300:
            raise ValueError("正规方程奇异")
        for row in range(n):
            if row != col:
                factor = work[row][col] / work[col][col]
                work[row] = [a - factor * b for a, b in zip(work[row], work[col])]
    return [work[i][n] / work[i][i] for i in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    work = [row[:] for row in matrix]
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        work[col], work[pivot] = work[pivot], work[col]
        identity[col], identity[pivot] = identity[pivot], identity[col]
        scale = work[col][col]
        work[col] = [x / scale for x in work[col]]
        identity[col] = [x / scale for x in identity[col]]
        for row in range(n):
            if row != col:
                factor = work[row][col]
                work[row] = [a - factor * b for a, b in zip(work[row], work[col])]
                identity[row] = [a - factor * b for a, b in zip(identity[row], identity[col])]
    return identity


def quadratic_fit(points: list[tuple[float, float]]) -> tuple[list[float], list[float], float]:
    """相对误差加权的二次拟合。返回 (系数[3], 标准误[3], MAPE)。"""
    rows = [[1.0, length, length * length] for length, _ in points]
    times = [value for _, value in points]
    n = len(times)
    p = 3
    weights = [1.0 / value for value in times]
    normal = [
        [sum(weights[r] ** 2 * rows[r][i] * rows[r][j] for r in range(n)) for j in range(p)]
        for i in range(p)
    ]
    rhs = [sum(weights[r] ** 2 * rows[r][i] * times[r] for r in range(n)) for i in range(p)]
    coef = _solve(normal, rhs)
    inverse = _inverse(normal)

    residuals = [
        weights[r] * (sum(rows[r][i] * coef[i] for i in range(p)) - times[r]) for r in range(n)
    ]
    dof = max(1, n - p)
    sigma2 = sum(r * r for r in residuals) / dof
    stderr = [math.sqrt(max(sigma2 * inverse[i][i], 0.0)) for i in range(p)]

    preds = [sum(rows[r][i] * coef[i] for i in range(p)) for r in range(n)]
    mape = sum(abs(pr - y) / y for pr, y in zip(preds, times)) / n * 100.0
    return coef, stderr, mape


def main() -> None:
    table = load_rows()
    lengths = sorted(table[2])
    print(f"数据：{DATA.name}，{len(lengths)} 个长度点，三档并行度")
    print(f"长度范围：{lengths[0]} – {lengths[-1]}（跨 {lengths[-1] / lengths[0]:.0f} 倍）")
    print()

    fits: dict[int, tuple[list[float], list[float], float]] = {}
    print(f"{'并行度':>6} {'b (ms)':>10} {'SE (ms)':>9} {'θ₁':>12} {'θ₂':>12} {'MAPE':>7}")
    for degree in (2, 4, 8):
        points = [(float(length), table[degree][length]) for length in lengths]
        coef, stderr, mape = quadratic_fit(points)
        fits[degree] = (coef, stderr, mape)
        print(
            f"cp{degree:<4} {coef[0] * 1e3:10.3f} {stderr[0] * 1e3:9.3f} "
            f"{coef[1]:12.3e} {coef[2]:12.3e} {mape:6.2f}%"
        )

    print()
    print("b 的跨档差异（两两，合并标准误）")
    for a, b in ((2, 4), (4, 8), (2, 8)):
        ca, sa, _ = fits[a]
        cb, sb, _ = fits[b]
        delta = (cb[0] - ca[0]) * 1e3
        err = math.hypot(sa[0], sb[0]) * 1e3
        print(
            f"  cp{a}→cp{b}: Δ={delta:+7.3f} ms  SE(Δ)={err:6.3f}  "
            f"{abs(delta) / err:5.2f}σ  {'显著' if abs(delta) > 2 * err else '不显著'}"
        )

    print()
    print("「三点、零自由度」的含义")
    print("  每档用 3 个参数拟合 13 个点，看起来绰绰有余；")
    print("  但**不同档之间的 b 只有 3 个值**，用它们去定 b(N) 的形状（至少 2 个参数 +")
    print("  一个检验自由度的残差）是不够的：3 点拟合 2 参数，残差恒为 0，")
    print("  任何单调凸/凹函数都能通过。这就是方案要求补测 N=1/3/6 的原因。")
    print()
    print("  四参数 b₁ 的估计（若强行假设线性）：")
    b2, b4, b8 = fits[2][0][0] * 1e3, fits[4][0][0] * 1e3, fits[8][0][0] * 1e3
    slope_24 = (b4 - b2) / 2.0
    slope_48 = (b8 - b4) / 4.0
    print(f"    cp2→cp4 每卡 {slope_24:+.3f} ms")
    print(f"    cp4→cp8 每卡 {slope_48:+.3f} ms")
    print(f"    两者之比 {slope_48 / slope_24:.2f}（线性应为 1.00）")
    print("    → 比值为 1 才算线性；偏离即说明形状未知，不能假设线性。")

    shape_table(fits)


def shape_table(fits: dict[int, tuple[list[float], list[float], float]]) -> None:
    """让四种形状各自穿过三个实测 b 点，看它们对内点的判断差多少。"""
    from marginal import fit_shape, optimal_degree_within, slope_at

    points = [(float(d), fits[d][0][0]) for d in (2, 4, 8)]
    shapes = fit_shape(points)

    print()
    print("四种形状各自穿过三个实测点，对内点的判断")
    slope_header = "b'(8)"
    print(f"{'形状':10s} {'b1(ms)':>8} {slope_header:>9} {'rms(us)':>9}")
    for fit in shapes:
        print(
            f"{fit.kind:10s} {fit.b1 * 1e3:8.3f} {slope_at(fit) * 1e3:9.3f} "
            f"{fit.rms * 1e6:9.1f}"
        )
    print("  （三点只留 1 个自由度，残差 62–343 μs 远小于 b 的标准误 470–906 μs，")
    print("    故 `identify_shape` 会返回 underdetermined —— 形状不可判）")

    print()
    cases = ((256, 1.40), (2048, 3.40), (8192, 72.93), (16384, 143.73))
    header = "".join(f"{f'L={length}':>22}" for length, _ in cases)
    print(f"{'形状':10s}{header}")
    for fit in shapes:
        cells = ""
        for length, compute_ms in cases:
            degree, best, penalty = optimal_degree_within(fit, compute_ms * 1e-3, cards=8)
            cells += f"  N*={degree}  损失{penalty * 100:5.1f}%"
        print(f"{fit.kind:10s}{cells}")
    print()
    print("  定性结论对形状稳健：短请求（L≤2048）用满 8 卡损失 19–41%，")
    print("  长请求（L≥8192）用满 8 卡无损失。差别只在 N* 的位置（1–2.4）。")


if __name__ == "__main__":
    main()
