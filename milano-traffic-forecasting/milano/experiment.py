"""Run rolling-origin direct point forecasts, or a clearly labelled smoke test."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import load_series, features, sample_indices, arrays
from .train import fit, predict, metrics
from .report import plot_predictions


def run(series_path, config, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = load_series(series_path)
    n = len(frame)
    config = dict(config)
    if smoke:
        config.update(lookback_steps=6, horizon_hours=[1], epochs=2,
                      seeds=[42], minimum_samples_per_split=2)
    train_end = int(n * config['train_fraction'])
    val_end = int(n * (config['train_fraction'] + config['validation_fraction']))
    if not 0 < train_end < val_end < n:
        raise ValueError('Invalid chronological splits')
    plans = {}
    y = frame.internet.to_numpy()
    for hours in config['horizon_hours']:
        horizon = int(hours * 6)
        plans[hours] = sample_indices(y, config['lookback_steps'], horizon, train_end, val_end)
    manifest = {'mode': 'PIPELINE_SMOKE_NOT_RESEARCH' if smoke else 'research',
                'config': config, 'series_sha256': hashlib.sha256(Path(series_path).read_bytes()).hexdigest(),
                'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__, 'device': 'cpu',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).parent.glob('*.py'))},
                'train_stop_exclusive': str(frame.index[train_end]),
                'validation_stop_exclusive': str(frame.index[val_end]),
                'sample_counts': {h: {s:len(p) for s,p in plan.items()} for h,plan in plans.items()}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if any(len(p) < config['minimum_samples_per_split'] for plan in plans.values() for p in plan.values()):
        raise ValueError('Insufficient data for configured horizons/splits; see manifest.json. Use --smoke for day-one pipeline verification.')
    results = []
    for hours, splits in plans.items():
        for enhanced in [False, True]:
            name = 'calendar_gru' if enhanced else 'traffic_gru'
            x, y, scaler = features(frame, train_end, enhanced)
            batches = {s: arrays(x, y, pairs, config['lookback_steps'], scaler) for s,pairs in splits.items()}
            for seed in config['seeds']:
                model, history = fit(batches['train'], batches['validation'], config, seed)
                prediction = predict(model, batches['test'][0], scaler)
                origins, targets = splits['test'].T
                actual = y[targets]
                score = metrics(actual, prediction)
                row = {'model': name, 'horizon_hours': hours, 'seed': seed, 'n_test': len(targets),
                       'mode': manifest['mode'], **score}
                results.append(row)
                prefix = f'{name}_{hours}h_seed{seed}'
                table = pd.DataFrame({'origin_time': frame.index[origins], 'target_time': frame.index[targets],
                                      'actual': actual, 'predicted': prediction})
                table.to_csv(output/f'{prefix}_predictions.csv', index=False)
                pd.DataFrame(history).to_csv(output/f'{prefix}_history.csv', index=False)
                torch.save({'state_dict': model.state_dict(), 'scaler': scaler,
                            'config': config, 'enhanced': enhanced, 'seed': seed,
                            'horizon_hours': hours}, output/f'{prefix}.pt')
                plot_predictions(table, output/f'{prefix}.png', f'{manifest["mode"]} | {prefix}')
                print(json.dumps(row), flush=True)
        # Persistence is an independent sanity baseline in the same original units.
        origins, targets = splits['test'].T
        results.append({'model': 'persistence', 'horizon_hours': hours, 'seed': -1,
                        'n_test': len(targets), 'mode': manifest['mode'], **metrics(y[targets], y[origins])})
    scores = pd.DataFrame(results)
    scores.to_csv(output/'metrics.csv', index=False)
    scores.groupby(['model','horizon_hours'])[['r2','rmse','mae']].agg(['mean','std']).to_csv(output/'summary.csv')
    base = scores[scores.model == 'traffic_gru'].set_index(['horizon_hours','seed'])
    improved = scores[scores.model == 'calendar_gru'].set_index(['horizon_hours','seed'])
    delta = improved[['r2','rmse','mae']] - base[['r2','rmse','mae']]
    delta.rename(columns=lambda c: f'delta_{c}_calendar_minus_traffic').to_csv(output/'paired_differences.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/experiment.json'))
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.series, json.loads(args.config.read_text()), args.out, args.smoke)


if __name__ == '__main__':
    main()
