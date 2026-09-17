#!/usr/bin/env python3
"""Profile three complete single-core transfer workflows, including GM pretranspose."""
import argparse
import csv
import json
import math
import shutil
import shlex
import re
import statistics
import subprocess
from pathlib import Path


def numeric(value):
    try:
        number = float(str(value).strip().rstrip('%'))
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def pipe_ratio(row, pipe):
    # Same precedence and bare-number convention as HW_GE_ATT NDDMA.
    zero = None
    for field in [f'aiv_{pipe}_ratio', f'{pipe}_exe_ratio', f'aic_{pipe}_ratio']:
        raw = row.get(field, '')
        value = numeric(raw)
        if value is None:
            continue
        ratio = value / 100 if '%' in str(raw) or value > 1 else value
        if not 0 <= ratio <= 1:
            raise ValueError(f'Invalid {field}: {raw}')
        if ratio > 0:
            return ratio, field
        zero = (ratio, field)
    for numerator, denominator in [(f'aiv_{pipe}_time(us)', 'aiv_time(us)'),
                                   (f'{pipe}_exe_time(us)', 'aiv_time(us)'),
                                   (f'{pipe}_exe_time(us)', 'Task Duration(us)')]:
        n, d = numeric(row.get(numerator)), numeric(row.get(denominator))
        if n is not None and d is not None and d > 0 and 0 < n <= d:
            return n / d, f'{numerator}/{denominator}'
    if zero is not None:
        return zero
    raise ValueError(f'Missing {pipe} ratio/time fields; cannot derive {pipe} cycles')


def read_metrics(folder, kernel):
    matches = []
    for path in sorted(folder.rglob('op_summary*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                row = {key.strip(): value for key, value in row.items() if key is not None}
                name = row.get('Op Name', '')
                if not re.search(r'(?<![A-Za-z0-9_])' + re.escape(kernel) + r'(?![A-Za-z0-9_])', name):
                    continue
                total = numeric(row.get('aiv_total_cycles'))
                if total is None or total <= 0:
                    raise ValueError(f'{path}:{line}: missing positive aiv_total_cycles')
                block_dim = numeric(row.get('Block Dim', row.get('BlockDim')))
                if block_dim is not None and block_dim != 1:
                    raise ValueError(f'{path}:{line}: expected single-core Block Dim=1')
                ratio, field = pipe_ratio(row, 'mte2')
                duration = numeric(row.get('Task Duration(us)'))
                if duration is not None and duration < 0:
                    raise ValueError(f'{path}:{line}: negative Task Duration(us)')
                result = dict(kernel=kernel, source_file=str(path), source_row=line,
                              task_duration_us=duration,
                              aiv_total_cycles=total, mte2_ratio=ratio,
                              mte2_ratio_field=field, mte2_cycles=total * ratio)
                try:
                    r3, f3 = pipe_ratio(row, 'mte3')
                    result.update(mte3_ratio=r3, mte3_ratio_field=f3, mte3_cycles=total * r3)
                except ValueError:
                    result.update(mte3_ratio=None, mte3_ratio_field='', mte3_cycles=None)
                matches.append(result)
    if len(matches) != 1:
        raise RuntimeError(f'{folder}: expected exactly one {kernel} row, got {len(matches)}. '
                           'Inspect op_summary names and exported schema.')
    return matches[0]


def build_command(msprof, folder, application):
    # Matches HW_GE_ATT/NDDMA/.../round2/scripts/run_round2_collection.py.
    return [msprof, f'--output={folder}', f'--application={shlex.join(application)}']


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


def workflow_metrics(folder, mode):
    kernels = ['datacopy_pretranspose', 'datacopy_contiguous'] if mode == 'pretranspose' else ['datacopy_' + mode]
    stages = [read_metrics(folder, kernel) for kernel in kernels]
    pre = stages[0]['mte2_cycles'] if mode == 'pretranspose' else 0
    load = stages[-1]['mte2_cycles']
    mte3 = [stage['mte3_cycles'] for stage in stages]
    durations = [stage['task_duration_us'] for stage in stages]
    return dict(pretranspose_mte2_cycles=pre, load_mte2_cycles=load,
                pretranspose_duration_us=durations[0] if mode == 'pretranspose' else 0,
                load_duration_us=durations[-1],
                task_duration_us=sum(durations) if all(v is not None for v in durations) else None,
                mte2_cycles=pre + load,
                aiv_total_cycles=sum(stage['aiv_total_cycles'] for stage in stages),
                mte3_cycles=sum(mte3) if all(v is not None for v in mte3) else None), stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--msprof', '--msprof-bin', dest='msprof', default='msprof',
                        help='msprof executable path; default: resolve msprof from PATH')
    parser.add_argument('--soc-version', default='', help='Optional metadata only; device profiling does not require it')
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
    config = dict(profiling_mode='application', metric='mte2_cycles', rows=args.rows, cols=args.cols, small_kib=args.small_kib,
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
            report = directory.resolve() / 'verification.txt'
            application = [binary, mode, str(args.rows), str(args.cols), str(kib),
                           str(store), str(args.device)]
            if store:
                application.append(str(report))
            command = build_command(msprof, directory.resolve() / 'prof', application)
            (directory / 'command.json').write_text(json.dumps(command, indent=2))
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if store and (not report.exists() or report.read_text().strip() != 'verification=PASS'):
                raise RuntimeError(f'No successful application verification report: {report}. '
                                   f'Check {directory / "run.log"}; rebuild demo_datacopy with the updated source.')

        # Validation stores every tile and compares every output element.
        # These three invocations are excluded from timed results.
        for mode in modes:
            run(mode, 1, args.output / f'check_{mode}')
        for trial in range(args.trials):
            # Rotate order; every invocation starts a fresh process and full workflow.
            for mode in modes[trial % 3:] + modes[:trial % 3]:
                run(mode, 0, args.output / f'{mode}_trial{trial}')

    records, stage_records = [], []
    for trial in range(args.trials):
        for mode in modes:
            metrics, stages = workflow_metrics(args.output / f'{mode}_trial{trial}', mode)
            stage_records.extend(dict(mode=mode, trial=trial, **stage) for stage in stages)
            plan = small if mode == 'small' else large
            records.append(dict(mode=mode, trial=trial, rows=args.rows, cols=args.cols,
                                **plan, **metrics))
    with (args.output / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output / 'stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stage_records[0]))
        writer.writeheader()
        writer.writerows(stage_records)
    summary = {}
    for mode in modes:
        selected = [row for row in records if row['mode'] == mode]
        summary[mode] = {key: statistics.median(row[key] for row in selected)
                         for key in ['pretranspose_mte2_cycles', 'load_mte2_cycles', 'mte2_cycles', 'aiv_total_cycles']}
        summary[mode]['mte3_cycles'] = (statistics.median(row['mte3_cycles'] for row in selected)
                                         if all(row['mte3_cycles'] is not None for row in selected) else None)
        summary[mode]['min_mte2_cycles'] = min(row['mte2_cycles'] for row in selected)
        summary[mode]['max_mte2_cycles'] = max(row['mte2_cycles'] for row in selected)
        for key in ['pretranspose_duration_us', 'load_duration_us', 'task_duration_us']:
            values = [row[key] for row in selected]
            summary[mode][key] = statistics.median(values) if all(v is not None for v in values) else None
        durations = [row['task_duration_us'] for row in selected]
        summary[mode]['min_task_duration_us'] = min(durations) if all(v is not None for v in durations) else None
        summary[mode]['max_task_duration_us'] = max(durations) if all(v is not None for v in durations) else None
    denominator = summary['large']['mte2_cycles']
    result = dict(metric='mte2_cycles = aiv_total_cycles * normalized MTE2 ratio (HW_GE_ATT NDDMA convention)',
                  results=summary,
                  small_over_large_mte2=(summary['small']['mte2_cycles'] / denominator if denominator else None),
                  pretranspose_over_large_mte2=(summary['pretranspose']['mte2_cycles'] / denominator if denominator else None))
    duration_base = summary['large']['task_duration_us']
    result['duration_metric'] = 'sum of Task Duration(us) per workflow; excludes inter-kernel gaps and host overhead'
    for mode in ['small', 'pretranspose']:
        duration = summary[mode]['task_duration_us']
        result[f'{mode}_over_large_duration'] = duration / duration_base if duration is not None and duration_base else None
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
