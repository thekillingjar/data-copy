"""Host model checks; not a substitute for Ascend compilation/simulation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import profile as profiler


def model(rows, cols, kib, pretranspose=False):
    plan = profiler.tile_plan(rows, cols, kib)
    a = list(range(rows * cols))
    b = [None] * len(a)
    for c0 in range(0, cols, plan['tile_cols']):
        count = min(plan['tile_cols'], cols - c0)
        ub = [None] * (count * rows)
        for c in range(count):
            assert c * rows * 4 % 32 == 0
            for block in range(rows):
                # DMA blockLen=4, GM gap=(cols-1)*4, Compact destination.
                source_byte = (c0 + c) * 4 + block * (4 + (cols - 1) * 4)
                ub[c * rows + block] = a[source_byte // 4]
        assert len(ub) * 4 <= plan['tile_bytes']
        b[c0 * rows:(c0 + count) * rows] = ub
    if pretranspose:
        output = []
        for c0 in range(0, cols, plan['tile_cols']):
            count = min(plan['tile_cols'], cols - c0)
            output.extend(b[c0 * rows:(c0 + count) * rows])
        return output
    return b


class TilingTest(unittest.TestCase):
    def test_three_paths_and_tail(self):
        for rows, cols, small, large in [(256, 768, 8, 240), (8, 37, 1, 2),
                                          (128, 513, 4, 128), (4088, 19, 16, 240)]:
            expected = [r * cols + c for c in range(cols) for r in range(rows)]
            for kib, pre in [(small, False), (large, False), (large, True)]:
                with self.subTest(rows=rows, cols=cols, kib=kib, pre=pre):
                    self.assertEqual(model(rows, cols, kib, pre), expected)

    def test_budget(self):
        self.assertEqual(profiler.tile_plan(256, 768, 8)['tile_count'], 96)
        self.assertEqual(profiler.tile_plan(256, 768, 240)['tile_count'], 4)
        for shape in [(7, 100, 8), (256, 768, 256), (4096, 100, 8),
                      (512, 768, 1), (8, 0, 8)]:
            with self.assertRaises(ValueError):
                profiler.tile_plan(*shape)

    def test_schema_and_phase_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'op_summary_test.csv'
            path.write_text('Op Name,aiv_total_cycles\ndatacopy_pretranspose,120\ndatacopy_contiguous,30\n')
            self.assertEqual(profiler.workflow_cycles(folder, 'pretranspose'), (120, 30, 150))
            path.write_text('Op Name,Task Duration(us)\ndatacopy_small,1.2\n')
            with self.assertRaises(RuntimeError):
                profiler.read_cycles(folder, 'datacopy_small')
            path.write_text('Op Name,aiv_total_cycles\ndatacopy_small,10\ndatacopy_small,20\n')
            with self.assertRaises(RuntimeError):
                profiler.read_cycles(folder, 'datacopy_small')

    def test_collection_flow_with_synthetic_profiler(self):
        # Synthetic values are fixtures, never measured performance.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'demo'
            binary.touch()
            msprof = root / 'tools' / 'profiler' / 'bin' / 'msprof'
            msprof.parent.mkdir(parents=True)
            msprof.write_text('#!/bin/sh\nexit 0\n')
            msprof.chmod(0o755)
            out = root / 'results'
            calls = []

            def fake_run(command, stdout, **kwargs):
                self.assertEqual(command[0], str(msprof))
                mode = command[6]
                store = int(command[10])
                calls.append((mode, store))
                prof = Path(command[4].split('=', 1)[1])
                prof.mkdir(parents=True)
                rows = ('datacopy_pretranspose,120\ndatacopy_contiguous,30\n'
                        if mode == 'pretranspose' else f'datacopy_{mode},100\n')
                (prof / 'op_summary.csv').write_text('Op Name,aiv_total_cycles\n' + rows)
                stdout.write('verification=PASS\n' if store else 'verification=not_requested\n')
                return subprocess.CompletedProcess(command, 0)

            argv = ['profile.py', '--binary', str(binary), '--soc-version', 'test',
                    '--output', str(out), '--trials', '1', '--msprof', str(msprof)]
            with patch.object(sys, 'argv', argv), patch.object(profiler.subprocess, 'run', fake_run), patch('builtins.print'):
                profiler.main()
            self.assertEqual(calls, [(m, s) for s in [1, 0] for m in ['small', 'large', 'pretranspose']])
            result = json.loads((out / 'summary.json').read_text())
            self.assertEqual(result['results']['pretranspose']['total_cycles'], 150)
            self.assertEqual(result['pretranspose_over_large'], 1.5)


if __name__ == '__main__':
    unittest.main()
