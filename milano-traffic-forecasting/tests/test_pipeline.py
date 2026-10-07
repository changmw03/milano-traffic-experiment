import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from milano.preprocess import aggregate
from milano.data import features, sample_indices, load_series


class PipelineTests(unittest.TestCase):
    def test_csv_time_resolution_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'series.csv'
            frame = pd.DataFrame({'timestamp': pd.date_range('2013-11-01', periods=10, freq='10min', tz='UTC'), 'internet': np.ones(10)})
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_series(path)), 10)
            frame.drop(index=3).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, '10-minute'):
                load_series(path)

    def test_scaler_never_uses_future(self):
        frame = pd.DataFrame({'internet': np.arange(100, dtype=float)},
                             index=pd.date_range('2013-11-01', periods=100, freq='10min', tz='UTC'))
        before, _, scale1 = features(frame, 70, True)
        frame.iloc[70:, 0] = 1e9
        after, _, scale2 = features(frame, 70, True)
        self.assertEqual(scale1, scale2)
        np.testing.assert_array_equal(before[:70], after[:70])
        self.assertEqual(before.shape[1], 6)

    def test_horizons_and_boundary_purge(self):
        y = np.arange(3000, dtype=float)
        for horizon in [6, 18, 36, 72, 144]:
            splits = sample_indices(y, 144, horizon, 2100, 2550)
            for pairs in splits.values():
                self.assertTrue(len(pairs) > 0)
                self.assertTrue(np.all(pairs[:, 1] - pairs[:, 0] == horizon))
            self.assertLess(splits['train'][:, 1].max(), 2100)
            self.assertGreaterEqual(splits['validation'][:, 0].min(), 2100)
            self.assertLess(splits['validation'][:, 1].max(), 2550)
            self.assertGreaterEqual(splits['test'][:, 0].min(), 2550)

    def test_missing_windows_are_excluded(self):
        y = np.arange(100, dtype=float)
        y[20] = np.nan
        splits = sample_indices(y, 6, 6, 70, 85)
        for pairs in splits.values():
            for origin, target in pairs:
                self.assertTrue(np.isfinite(y[origin-5:origin+1]).all())
                self.assertTrue(np.isfinite(y[target]))

    def test_raw_country_sum_blanks_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sms-call-internet-mi-2013-11-01.txt'
            lines = ['1\t1383260400000\t39\t\t\t\t\t2',
                     '1\t1383260400000\t33\t\t\t\t\t3',
                     '1\t1383260400000\t0\t\t\t\t\t',
                     '1\t1383261000000\t39\t\t\t\t\t']
            path.write_text('\n'.join(lines)+'\n')
            frame, q = aggregate([path])
            self.assertEqual(frame.internet.iloc[0], 5)
            self.assertTrue(np.isnan(frame.internet.iloc[1]))
            self.assertEqual(len(frame), 144)
            self.assertEqual(q['absent_slots'], 142)
            self.assertEqual(q['present_but_all_internet_blank_slots'], 1)
            path.write_text('\n'.join(lines+[lines[0]])+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                aggregate([path])

    def test_real_day_is_not_a_research_dataset(self):
        y = np.ones(144)
        for h in [6, 18, 36, 72, 144]:
            self.assertTrue(all(len(p) == 0 for p in sample_indices(y, 144, h, 100, 122).values()))


if __name__ == '__main__':
    unittest.main()
