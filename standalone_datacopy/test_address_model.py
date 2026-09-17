"""Host model checks; not a substitute for Ascend compilation/simulation."""
import json
import shlex
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
            path.write_text('Op Name,aiv_total_cycles,aiv_mte2_ratio,aiv_mte3_ratio\n'
                            'datacopy_pretranspose,120,50,25\ndatacopy_contiguous,30,0.8,0\n')
            metrics, stages = profiler.workflow_metrics(folder, 'pretranspose')
            self.assertEqual(metrics['mte2_cycles'], 84)
            self.assertEqual(metrics['mte3_cycles'], 30)
            self.assertEqual(metrics['aiv_total_cycles'], 150)
            self.assertEqual(stages[0]['mte2_ratio_field'], 'aiv_mte2_ratio')
            path.write_text('Op Name,aiv_total_cycles\ndatacopy_small,100\n')
            with self.assertRaises(ValueError):
                profiler.read_metrics(folder, 'datacopy_small')
            path.write_text('Op Name,aiv_total_cycles,aiv_mte2_ratio\n'
                            'datacopy_small,10,0.5\ndatacopy_small,20,0.5\n')
            with self.assertRaises(RuntimeError):
                profiler.read_metrics(folder, 'datacopy_small')

    def test_ratio_conventions(self):
        for value, expected in [('50', 0.5), ('0.5', 0.5), ('0.5%', 0.005), ('0', 0)]:
            self.assertEqual(profiler.pipe_ratio({'aiv_mte2_ratio': value}, 'mte2')[0], expected)
        ratio, field = profiler.pipe_ratio({'aiv_mte2_ratio': 'N/A',
                                            'aiv_mte2_time(us)': '2', 'aiv_time(us)': '8'}, 'mte2')
        self.assertEqual(ratio, 0.25)
        self.assertIn('/', field)
        with self.assertRaises(ValueError):
            profiler.pipe_ratio({'aiv_mte2_ratio': '-1'}, 'mte2')

    def test_collection_flow_with_synthetic_profiler(self):
        # Synthetic values are fixtures, never measured performance.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'demo with spaces'
            binary.touch()
            msprof = root / 'tools' / 'profiler' / 'bin' / 'msprof'
            msprof.parent.mkdir(parents=True)
            msprof.write_text('#!/bin/sh\nexit 0\n')
            msprof.chmod(0o755)
            out = root / 'results'
            calls = []

            def fake_run(command, stdout, **kwargs):
                self.assertEqual(command[0], str(msprof))
                self.assertEqual(len(command), 3)
                application = shlex.split(command[2].split('=', 1)[1])
                self.assertEqual(application[0], str(binary.resolve()))
                mode = application[1]
                store = int(application[5])
                calls.append((mode, store))
                prof = Path(command[1].split('=', 1)[1])
                prof.mkdir(parents=True)
                rows = ('datacopy_pretranspose,120,0.5\ndatacopy_contiguous,30,0.8\n'
                        if mode == 'pretranspose' else f'datacopy_{mode},100,0.5\n')
                (prof / 'op_summary.csv').write_text('Op Name,aiv_total_cycles,aiv_mte2_ratio\n' + rows)
                # Application stdout is deliberately absent from profiler log.
                stdout.write('Profiling finished\n')
                if store:
                    Path(application[7]).write_text('verification=PASS\n')
                return subprocess.CompletedProcess(command, 0)

            argv = ['profile.py', '--binary', str(binary), '--soc-version', 'test',
                    '--output', str(out), '--trials', '1', '--msprof', str(msprof)]
            with patch.object(sys, 'argv', argv), patch.object(profiler.subprocess, 'run', fake_run), patch('builtins.print'):
                profiler.main()
            self.assertEqual(calls, [(m, s) for s in [1, 0] for m in ['small', 'large', 'pretranspose']])
            result = json.loads((out / 'summary.json').read_text())
            self.assertEqual(result['results']['pretranspose']['mte2_cycles'], 84)
            self.assertEqual(result['pretranspose_over_large_mte2'], 1.68)

            # A profiler exit code of zero (or a PASS string in its stdout)
            # must not bypass a missing application-owned verification file.
            missing_out = root / 'missing_report'
            argv[argv.index(str(out))] = str(missing_out)
            def no_report(command, stdout, **kwargs):
                stdout.write('verification=PASS\n')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(sys, 'argv', argv), patch.object(profiler.subprocess, 'run', no_report), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'verification report'):
                    profiler.main()


if __name__ == '__main__':
    unittest.main()
