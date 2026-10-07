"""Stream raw TSV files without changing the originals."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['square_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
           'call_in', 'call_out', 'internet']


def aggregate(paths, square=1):
    records, manifests, days = [], [], []
    total, missing = 0, Counter()
    selected_keys = set()
    for path in sorted(set(Path(p).resolve() for p in paths)):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        if not match:
            raise ValueError(f'Filename must contain YYYY-MM-DD: {path}')
        day = pd.Timestamp(match[1], tz='Europe/Rome')
        if day in days:
            raise ValueError('Multiple files for the same date; refusing double counting')
        days.append(day)
        lower, upper = day.tz_convert('UTC').value // 10**6, (day + pd.DateOffset(days=1)).tz_convert('UTC').value // 10**6
        digest, md5, rows = hashlib.sha256(), hashlib.md5(), 0
        with path.open('rb') as f:
            for line_no, raw in enumerate(f, 1):
                digest.update(raw)
                md5.update(raw)
                values = raw.decode('utf-8-sig').rstrip('\r\n').split('\t')
                if len(values) != 8:
                    raise ValueError(f'{path.name}:{line_no}: expected 8 tab-separated fields')
                try:
                    sid, timestamp, country = map(int, values[:3])
                    activity = [float(v) if v else np.nan for v in values[3:]]
                except ValueError as exc:
                    raise ValueError(f'{path.name}:{line_no}: invalid number') from exc
                if not 1 <= sid <= 10000 or country < 0:
                    raise ValueError(f'{path.name}:{line_no}: invalid ID')
                if timestamp % 600000 or not lower <= timestamp < upper:
                    raise ValueError(f'{path.name}:{line_no}: timestamp outside date or 10-minute grid')
                for name, raw_value, number in zip(COLUMNS[3:], values[3:], activity):
                    if raw_value and (not math.isfinite(number) or number < 0):
                        raise ValueError(f'{path.name}:{line_no}: invalid {name}')
                    if not raw_value:
                        missing[name] += 1
                rows += 1
                if sid == square:
                    key = (timestamp, country)
                    if key in selected_keys:
                        raise ValueError(f'Duplicate selected-square timestamp/country: {key}')
                    selected_keys.add(key)
                    records.append((timestamp, country, activity[-1]))
        total += rows
        manifests.append({'path': str(path), 'bytes': path.stat().st_size,
                          'sha256': digest.hexdigest(), 'md5': md5.hexdigest(), 'rows': rows})
        print(f'CHECKED {path.name}: {rows:,} rows', flush=True)
    if not records:
        raise ValueError(f'No records for Square ID {square}')
    raw = pd.DataFrame(records, columns=['timestamp_ms', 'country_code', 'internet'])
    raw['timestamp'] = pd.to_datetime(raw.timestamp_ms, unit='ms', utc=True)
    grouped = raw.groupby('timestamp').internet
    # Blank country-level activities contribute nothing to a sum; an all-blank
    # timestamp remains unknown, rather than silently becoming zero.
    frame = pd.DataFrame({'internet': grouped.sum(min_count=1),
                          'country_rows': grouped.size(), 'internet_nonblank_rows': grouped.count()})
    index = pd.date_range(min(days), max(days) + pd.DateOffset(days=1), freq='10min', inclusive='left').tz_convert('UTC')
    frame = frame.reindex(index)
    frame.index.name = 'timestamp'
    frame['timestamp_local'] = frame.index.tz_convert('Europe/Rome').astype(str)
    frame['square_id'] = square
    quality = {
        'square_id': square, 'delimiter': 'TAB', 'header': False, 'columns': COLUMNS,
        'files': manifests, 'raw_rows_all_squares': total,
        'blank_activity_fields_all_squares': dict(missing),
        'selected_rows': len(raw), 'selected_duplicate_keys': 0,
        'duplicate_check_scope': 'selected square only',
        'selected_internet_blank_rows': int(raw.internet.isna().sum()),
        'selected_country_codes': sorted(raw.country_code.unique().tolist()),
        'expected_slots': len(frame),
        'absent_slots': int(frame.country_rows.isna().sum()),
        'present_but_all_internet_blank_slots': int(((frame.country_rows > 0) & frame.internet.isna()).sum()),
        'missing_internet_slots': int(frame.internet.isna().sum()),
        'zero_internet_slots': int(frame.internet.eq(0).sum()),
        'start_local': str(index[0].tz_convert('Europe/Rome')),
        'end_local': str(index[-1].tz_convert('Europe/Rome')),
        'internet_summary': frame.internet.describe().to_dict(),
        'missing_policy': 'sum nonblank country entries; all blank/absent -> NaN; no imputation',
    }
    return frame, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Files or quoted glob patterns')
    parser.add_argument('--square', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('outputs/day1'))
    args = parser.parse_args()
    import glob
    paths = [p for pattern in args.input for p in glob.glob(pattern)]
    if not paths:
        parser.error('No input files matched')
    frame, quality = aggregate(paths, args.square)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'series.csv')
    (args.out / 'quality.json').write_text(json.dumps(quality, indent=2, allow_nan=False), encoding='utf-8')
    from .report import plot_series
    plot_series(frame, args.out / 'traffic.png', args.square)
    print(json.dumps(quality, indent=2))


if __name__ == '__main__':
    main()
