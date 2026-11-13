"""CV 流水对 CP 的约束：核数上限、流水深度、缩 tile 的代价。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvflow import (
    CVConfig,
    core_utilisation,
    max_degree,
    max_tile_for_cores,
    pipeline_fit,
    recommended_degree,
    tasks,
    total_transfer,
)

CFG = CVConfig()


class TaskCountTests(unittest.TestCase):
    def test_cp_only_shrinks_the_query_side(self) -> None:
        # KV 侧不动：N 只作用于 Q tile 数
        one = tasks(8192, 1, CFG)
        eight = tasks(8192, 8, CFG)
        self.assertAlmostEqual(one / eight, 8.0, places=6)

    def test_causal_counts_half(self) -> None:
        full = tasks(4096, 2, CFG, causal=False)
        masked = tasks(4096, 2, CFG, causal=True)
        self.assertLess(masked, full / 2 + 1)

    def test_rejects_bad_degree(self) -> None:
        with self.assertRaises(ValueError):
            tasks(1024, 0, CFG)


class CoreLimitTests(unittest.TestCase):
    def test_short_requests_cannot_fill_cores(self) -> None:
        # L=256 用 cp8：只有 2 个任务给 20 个核
        self.assertLess(core_utilisation(256, 8, CFG), 0.2)

    def test_long_requests_fill_cores(self) -> None:
        self.assertAlmostEqual(core_utilisation(8192, 8, CFG), 1.0, places=6)

    def test_max_degree_matches_the_solved_formula(self) -> None:
        # 因果掩码下 N <= L/(BM*sqrt(2*AIC))
        expected = 2048 / (128 * math.sqrt(2 * 20))
        self.assertAlmostEqual(max_degree(2048, CFG, causal=True), expected, places=6)

    def test_recommended_degree_is_a_power_of_two(self) -> None:
        for length in (256, 512, 1024, 2048, 4096, 8192, 16384):
            degree = recommended_degree(length, CFG, causal=True)
            self.assertIn(degree, (1, 2, 4, 8), length)

    def test_recommended_degree_grows_with_length(self) -> None:
        shorts = [recommended_degree(n, CFG) for n in (256, 1024, 2048)]
        longs = [recommended_degree(n, CFG) for n in (8192, 16384)]
        self.assertLessEqual(max(shorts), min(longs))

    def test_short_requests_are_not_split(self) -> None:
        # L<=1024 因果掩码下连 N=2 都不该用
        for length in (256, 512, 1024):
            self.assertEqual(recommended_degree(length, CFG), 1, length)


class TileShrinkTests(unittest.TestCase):
    def test_tile_must_shrink_when_tasks_are_insufficient(self) -> None:
        # L=2048, N=8, BM=128 时任务数不足；需要缩到 32 才够
        self.assertLess(tasks(2048, 8, CFG, True), CFG.aic)
        tile = max_tile_for_cores(2048, 8, CFG, causal=True)
        self.assertLessEqual(tile, 32)
        shrunk = CVConfig(bm=tile, bn=tile)
        self.assertGreaterEqual(tasks(2048, 8, shrunk, True), CFG.aic)

    def test_tile_is_a_multiple_of_sixteen(self) -> None:
        for length in (1024, 2048, 4096, 8192):
            self.assertEqual(max_tile_for_cores(length, 4, CFG) % 16, 0)

    def test_total_transfer_is_roughly_invariant_to_tile(self) -> None:
        # 缩 tile 不减少总中转字节，只增加次数
        big = total_transfer(2048, 8, CVConfig(bm=128, bn=128), True)
        small = total_transfer(2048, 8, CVConfig(bm=32, bn=32), True)
        self.assertLess(small / big, 2.0, "总中转量应大致守恒")

    def test_small_tiles_mean_more_syncs(self) -> None:
        big = tasks(2048, 8, CVConfig(bm=128, bn=128), True)
        small = tasks(2048, 8, CVConfig(bm=32, bn=32), True)
        self.assertGreater(small, big * 5)


class PipelineTests(unittest.TestCase):
    def test_pipeline_fit_reports_utilisation(self) -> None:
        fit = pipeline_fit(1024, CFG, 2, causal=True)
        self.assertEqual(len(fit.degrees), 4)
        self.assertEqual(len(fit.utilisation), 4)
        self.assertAlmostEqual(fit.utilisation[0], 1.0, places=6)
        self.assertLess(fit.utilisation[-1], 0.2)

    def test_blocking_detected_for_tiny_workload(self) -> None:
        # L=64 单 Q tile、单 KV tile
        self.assertTrue(pipeline_fit(64, CFG, 2, causal=True).blocking)


class ConfigTests(unittest.TestCase):
    def test_tile_must_align_to_fractal(self) -> None:
        with self.assertRaises(ValueError):
            CVConfig(bm=100)

    def test_noc_l0c_to_ub_is_default(self) -> None:
        self.assertFalse(CFG.has_l0c_to_ub)


if __name__ == "__main__":
    unittest.main()
