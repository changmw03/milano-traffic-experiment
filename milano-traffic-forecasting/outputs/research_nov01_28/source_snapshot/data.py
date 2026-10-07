"""Chronological samples, training-only scaling, no future filling."""
import numpy as np
import pandas as pd


def load_series(path):
    frame = pd.read_csv(path)
    frame.index = pd.to_datetime(frame.pop('timestamp'), utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Timestamps must be sorted and unique')
    if len(frame) < 2 or not np.all((frame.index[1:] - frame.index[:-1]) == pd.Timedelta(minutes=10)):
        raise ValueError('Expected a complete 10-minute grid, retaining missing values')
    return frame


def features(frame, train_end, enhanced):
    y = frame.internet.to_numpy(dtype=np.float64)
    observed = y[:train_end][np.isfinite(y[:train_end])]
    if len(observed) < 2:
        raise ValueError('Insufficient observed training values')
    mean, std = float(observed.mean()), float(observed.std())
    std = std if std > 1e-8 else 1.0
    x = ((y - mean) / std)[:, None]
    if enhanced:
        local = frame.index.tz_convert('Europe/Rome')
        hour = (local.hour + local.minute / 60) / 24
        day = local.dayofweek / 7
        calendar = np.column_stack([np.sin(2*np.pi*hour), np.cos(2*np.pi*hour),
                                    np.sin(2*np.pi*day), np.cos(2*np.pi*day),
                                    (local.dayofweek >= 5).astype(float)])
        x = np.column_stack([x, calendar])
    return x.astype(np.float32), y, {'mean': mean, 'std': std, 'fit_stop_exclusive': train_end}


def sample_indices(y, lookback, horizon, train_end, val_end):
    """Origin = most recent observed bin; target = origin + horizon.

    Purge boundary-crossing forecasts: validation/test origins must already
    belong to their period, so model selection never uses future labels
    relative to an evaluated forecast origin. Past context may cross a boundary.
    """
    splits = {name: [] for name in ('train', 'validation', 'test')}
    for origin in range(lookback - 1, len(y) - horizon):
        target = origin + horizon
        if not np.isfinite(y[origin-lookback+1:origin+1]).all() or not np.isfinite(y[target]):
            continue
        if target < train_end:
            split = 'train'
        elif origin >= train_end and target < val_end:
            split = 'validation'
        elif origin >= val_end:
            split = 'test'
        else:
            continue
        splits[split].append((origin, target))
    return {name: np.asarray(pairs, dtype=np.int64).reshape(-1, 2) for name, pairs in splits.items()}


def arrays(x, y, pairs, lookback, scaler):
    inputs = np.stack([x[o-lookback+1:o+1] for o, _ in pairs])
    targets = ((y[pairs[:, 1]] - scaler['mean']) / scaler['std']).astype(np.float32)
    return inputs, targets
