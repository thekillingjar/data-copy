#!/usr/bin/env python3
"""Profile three complete single-core transfer workflows, including GM pretranspose."""
import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path


def read_cycles(folder, kernel):
    values = []
    for path in folder.rglob('op_summary*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if not any(kernel in value for value in row.values() if value):
                    continue
                value = row.get('aiv_total_cycles', '').strip()
                if value and value not in ('N/A', 'NA', '--'):
                    number = float(value)
                    if math.isfinite(number) and number > 0:
                        values.append(number)
    if len(values) != 1:
        raise RuntimeError(f'{folder}: expected exactly one {kernel} aiv_total_cycles row, got {len(values)}. '
                           'Inspect op_summary schema; no time-to-cycle conversion is performed.')
    return values[0]


def tile_plan(rows, cols, kib):
    if rows < 8 or rows > 4088 or rows % 8 or not 1 <= cols <= 65535:
        raise ValueError('rows must be 8..4088 and divisible by 8; cols must be 1..65535')
    if not 1 <= kib <= 240:
        raise ValueError('tile KiB must be 1..240 (16 KiB reserved from 256 KiB UB)')
    if rows * cols * 4 > 64 * 1024 * 1024:
        raise ValueError('Input limit: 64 MiB')
    tile_cols = min(cols, kib * 1024 // (rows * 4))
    if tile_cols == 0:
        raise ValueError('Tile budget cannot hold one full output row')
    return dict(tile_cols=tile_cols, tile_bytes=tile_cols * rows * 4,
                tile_count=(cols + tile_cols - 1) // tile_cols)


def workflow_cycles(folder, mode):
    if mode == 'pretranspose':
        pre = read_cycles(folder, 'datacopy_pretranspose')
        load = read_cycles(folder, 'datacopy_contiguous')
        return pre, load, pre + load
    load = read_cycles(folder, 'datacopy_' + mode)
    return 0, load, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--msprof', default='msprof',
                        help='msprof executable path; default: resolve msprof from PATH')
    parser.add_argument('--soc-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--cols', type=int, default=768)
    parser.add_argument('--small-kib', type=int, default=8)
    parser.add_argument('--large-kib', type=int, default=240)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--analyze-only', action='store_true', help='Reparse existing output without executing kernels')
    args = parser.parse_args()
    try:
        small = tile_plan(args.rows, args.cols, args.small_kib)
        large = tile_plan(args.rows, args.cols, args.large_kib)
    except ValueError as exc:
        parser.error(str(exc))
    if args.trials < 1 or not 0 <= args.device <= 255:
        parser.error('trials >= 1; device: 0..255')
    if small['tile_bytes'] >= large['tile_bytes']:
        parser.error('Actual small tile must be smaller than large tile')
    modes = ['small', 'large', 'pretranspose']
    config = dict(rows=args.rows, cols=args.cols, small_kib=args.small_kib,
                  large_kib=args.large_kib, trials=args.trials, device=args.device,
                  soc_version=args.soc_version, binary=str(args.binary.resolve()),
                  input_bytes=args.rows * args.cols * 4, hardware_ub_bytes=256 * 1024,
                  small=small, large=large)
    if args.analyze_only:
        saved = json.loads((args.output / 'config.json').read_text())
        if config != saved:
            parser.error('Analysis arguments must match saved config.json')
    else:
        msprof = shutil.which(args.msprof)
        if msprof is None:
            parser.error(f'msprof executable not found or not executable: {args.msprof}. '
                         'Use --msprof /path/to/tools/profiler/bin/msprof')
        msprof = str(Path(msprof).absolute())
        print(f'Using msprof: {msprof}')
        binary = str(args.binary.resolve(strict=True))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

        def run(mode, store, directory):
            directory.mkdir()
            kib = args.small_kib if mode == 'small' else args.large_kib
            command = [msprof, 'op', 'simulator', f'--soc-version={args.soc_version}',
                       f'--output={directory.resolve() / "prof"}', binary, mode,
                       str(args.rows), str(args.cols), str(kib), str(store), str(args.device)]
            (directory / 'command.json').write_text(json.dumps(command, indent=2))
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if store and 'verification=PASS' not in (directory / 'run.log').read_text():
                raise RuntimeError(f'Correctness check did not pass: {directory}')

        # Validation stores every tile and compares every output element.
        # These three invocations are excluded from timed results.
        for mode in modes:
            run(mode, 1, args.output / f'check_{mode}')
        for trial in range(args.trials):
            # Rotate order; every invocation starts a fresh process and full workflow.
            for mode in modes[trial % 3:] + modes[:trial % 3]:
                run(mode, 0, args.output / f'{mode}_trial{trial}')

    records = []
    for trial in range(args.trials):
        for mode in modes:
            pre, load, total = workflow_cycles(args.output / f'{mode}_trial{trial}', mode)
            plan = small if mode == 'small' else large
            records.append(dict(mode=mode, trial=trial, rows=args.rows, cols=args.cols,
                                **plan, pretranspose_cycles=pre, load_cycles=load,
                                total_cycles=total))
    with (args.output / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for mode in modes:
        selected = [row for row in records if row['mode'] == mode]
        summary[mode] = {key: statistics.median(row[key] for row in selected)
                         for key in ['pretranspose_cycles', 'load_cycles', 'total_cycles']}
        summary[mode]['min_total_cycles'] = min(row['total_cycles'] for row in selected)
        summary[mode]['max_total_cycles'] = max(row['total_cycles'] for row in selected)
    result = dict(metric='sum of single-core aiv_total_cycles per full workflow; excludes host/inter-kernel gaps',
                  results=summary,
                  small_over_large=summary['small']['total_cycles'] / summary['large']['total_cycles'],
                  pretranspose_over_large=summary['pretranspose']['total_cycles'] / summary['large']['total_cycles'])
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
