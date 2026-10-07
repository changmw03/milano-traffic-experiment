"""Synthetic integration check only: never reported as real-data research."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from milano.experiment import run


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        n = 144 * 14
        frame = pd.DataFrame({'timestamp': pd.date_range('2013-11-01', periods=n, freq='10min', tz='UTC'),
                              'internet': 10 + np.sin(np.arange(n)*2*np.pi/144)})
        frame.to_csv(root/'synthetic.csv', index=False)
        config = json.loads(Path('configs/experiment.json').read_text())
        config.update(seeds=[42], epochs=1, hidden_size=4, minimum_samples_per_split=2)
        run(root/'synthetic.csv', config, root/'run')
        result = pd.read_csv(root/'run/metrics.csv')
        assert len(result) == 15
        assert np.isfinite(result[['r2','rmse','mae']]).all().all()
        assert set(result.horizon_hours) == {1,3,6,12,24}
        for hours in config['horizon_hours']:
            a = pd.read_csv(root/f'run/traffic_gru_{hours}h_seed42_predictions.csv')
            b = pd.read_csv(root/f'run/calendar_gru_{hours}h_seed42_predictions.csv')
            pd.testing.assert_series_equal(a.target_time, b.target_time)
            pd.testing.assert_series_equal(a.actual, b.actual)
        print('PASS: synthetic integration, 2 GRUs x 5 horizons; equal evaluation targets')


if __name__ == '__main__':
    main()
