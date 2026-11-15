"""Standalone entry point for the experiments reported in the PG-VPS paper."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'configs/experiments.json').read_text())


def rows_for(dataset):
    rows = []
    for task in CONFIG['datasets'][dataset]['tasks']:
        key, friction, mass, mode, *setting = task
        case = CONFIG['cases'][key]
        suffix = setting[0] if setting else mode
        row = dict(case, friction=friction, mass=mass, mode=mode,
                   name=f'{key}_{friction}_{mass}_{suffix}',
                   cfg=dict(CONFIG['common'], object_mass=mass))
        if setting:
            row['setting'] = setting[0]
        rows.append(row)
    return rows


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def execute(task):
    dataset, row, out = task
    from scripts import episode
    original = episode.make_controller
    if 'setting' in row:
        from research_platform.pgvps_mpc import PGVPSCapture
        def build(r):
            controller = PGVPSCapture(r['estimated_vertices'], r['seed'] + 200000,
                                     r['cfg']['servo_kv'])
            core = controller.core
            setting = CONFIG['sensitivity'][r['setting']]
            core.STRICT_RATIO = setting['strict']
            core.WINDOW = setting['window']
            core.MIN_RELATIVE_PROGRESS = setting['progress']
            core.progress_history = deque(maxlen=setting['window'])
            core.switch_ratio = setting['strict']
            return controller
        episode.make_controller = build
    try:
        if dataset == 'pgtaiv_public_baselines':
            from scripts.baseline_episode import run
            result = run((out, row, CONFIG['public']))
        else:
            result = episode.run((out, row))
    finally:
        episode.make_controller = original
    atomic_json(out / (row['name'] + '_result.json'), result)
    return result


def fingerprint():
    paths = sorted(ROOT.rglob('*.py')) + sorted((ROOT / 'configs').glob('*.json'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if 'results' not in p.relative_to(ROOT).parts}
    versions = {name: importlib.metadata.version(name)
                for name in ['numpy', 'scipy', 'mujoco', 'matplotlib']}
    return dict(sources=hashes, versions=versions, python=sys.version.split()[0])


def simulate(args, smoke=False):
    marker = fingerprint()
    for dataset, spec in CONFIG['datasets'].items():
        if args.study != 'all' and spec['group'] != args.study:
            continue
        rows = rows_for(dataset)
        if smoke:
            unique = {}
            for row in rows:
                unique.setdefault((row['mode'], row.get('setting')), row)
            rows = list(unique.values())
            rows = [dict(r, cfg=dict(r['cfg'], steps=3)) for r in rows]
        out = args.output / ('smoke_results' if smoke else 'results') / dataset
        out.mkdir(parents=True, exist_ok=True)
        provenance = out / 'provenance.json'
        if provenance.exists():
            if not args.resume:
                raise SystemExit(f'{out} already exists; use --resume or a new --output.')
            if json.loads(provenance.read_text()) != marker:
                raise SystemExit(f'Source/config/environment mismatch: cannot resume {out}')
        elif any(out.iterdir()):
            raise SystemExit(f'Refusing to mix untracked files in {out}')
        atomic_json(provenance, marker)
        atomic_json(out / 'protocol.json', dict(tasks=rows, smoke_only=smoke))
        selected = rows[:args.limit] if args.limit else rows
        pending = [r for r in selected if not all((out / (r['name'] + suffix)).exists()
                   for suffix in ['_result.json', '.npz', '_plans.json'])]
        tasks = [(dataset, r, out) for r in pending]
        print(f'{dataset}: {len(tasks)} pending / {len(rows)} required', flush=True)
        def collect(result):
            print(result['name'], 'success=', result['success'],
                  'cost=', round(result['cost'], 6), flush=True)
        if args.workers == 1:
            for task in tasks:
                collect(execute(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(execute, tasks):
                    collect(result)
        completed = [json.loads((out / (r['name'] + '_result.json')).read_text())
                     for r in rows if (out / (r['name'] + '_result.json')).exists()]
        atomic_json(out / 'summary.json', completed)
        atomic_json(out / 'completion.json', dict(expected=len(rows), completed=len(completed),
                                                complete=len(rows) == len(completed), smoke_only=smoke))


def require_complete(output):
    for dataset in CONFIG['datasets']:
        file = output / 'results' / dataset / 'summary.json'
        if not file.exists():
            raise SystemExit(f'Missing {file}; complete the experiments first.')
        actual = {r['name'] for r in json.loads(file.read_text())}
        expected = {r['name'] for r in rows_for(dataset)}
        if actual != expected:
            raise SystemExit(f'Incomplete experiment: {dataset} ({len(actual)}/{len(expected)})')


def analyze(output):
    require_complete(output)
    from scripts import analyze_pgvps as analysis
    analysis.ROOT = output
    analysis.OUT = output / 'results/pgvps_analysis'
    analysis.main()


def figures(output):
    require_complete(output)
    if not (output / 'results/pgvps_analysis/analysis.json').exists():
        raise SystemExit('Run analyze before figures.')
    from scripts import make_pgtaiv_control_figures as plots
    plots.ROOT = output
    plots.DATA = output / 'results/pgvps_independent_confirmatory'
    plots.OUT = output / 'figures'
    plots.main()
    from scripts import make_mujoco_environment_figure as scene
    scene.ROOT = output
    scene.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'run', 'analyze', 'figures', 'all', 'smoke', 'validate', 'benchmark'])
    parser.add_argument('--study', choices=['all', 'independent', 'ablation', 'sensitivity', 'public'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='Working directory containing generated results/ and figures/.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, help='Debug only: full episodes per dataset; analysis requires complete data.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    if args.command == 'all' and (args.study != 'all' or args.limit):
        parser.error('all requires every study without --limit')
    if args.command == 'list':
        for name, spec in CONFIG['datasets'].items():
            print(spec['group'], name, len(spec['tasks']))
    elif args.command == 'validate':
        from scripts.validate_decision_bounds import run
        run(args.output)
    elif args.command == 'benchmark':
        from scripts.benchmark_budget import run
        run(args.output)
    elif args.command in ['run', 'smoke']:
        simulate(args, smoke=args.command == 'smoke')
    elif args.command == 'analyze':
        analyze(args.output)
    elif args.command == 'figures':
        figures(args.output)
    else:
        simulate(args)
        from scripts.validate_decision_bounds import run as validate
        from scripts.benchmark_budget import run as benchmark
        validate(args.output)
        benchmark(args.output)
        analyze(args.output)
        figures(args.output)


if __name__ == '__main__':
    main()
