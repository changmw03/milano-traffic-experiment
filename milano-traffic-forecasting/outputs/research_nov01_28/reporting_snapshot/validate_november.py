"""Validate the fixed 28-day source inventory and run existing aggregation."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from milano.preprocess import aggregate
from milano.report import plot_series
from milano.data import sample_indices, features


def main():
    out = ROOT/'outputs/nov01_28'
    out.mkdir(exist_ok=True)
    if (out/'series.csv').exists():
        raise ValueError('Validated series already exists; refusing to overwrite it')
    metadata = json.loads((ROOT/'raw/dataverse_metadata.json').read_text())
    official = {i['dataFile']['filename']: i['dataFile'] for i in metadata['data']['latestVersion']['files']}
    paths = []
    for day in range(1, 29):
        name = f'sms-call-internet-mi-2013-11-{day:02d}.txt'
        candidates = [ROOT/'raw'/name]
        if day == 1:
            candidates.append(Path.home()/'Downloads'/name)
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise ValueError(f'Expected exactly one file for {name}, found {found}')
        path = found[0]
        if path.stat().st_size != official[name]['filesize']:
            raise ValueError(f'Size mismatch: {name}')
        paths.append(path)
    if len({official[p.name]['checksum']['value'] for p in paths}) != 28:
        raise ValueError('Duplicate file content in official selected metadata')
    (out/'input_files.json').write_text(json.dumps([str(p) for p in paths], indent=2))
    print('INVENTORY OK: 28 dates, one file per date, official sizes matched', flush=True)
    frame, quality = aggregate(paths, square=1)
    for record in quality['files']:
        name = Path(record['path']).name
        if record['md5'] != official[name]['checksum']['value']:
            raise ValueError(f'Official checksum mismatch: {name}')
    quality['official_md5_all_match'] = True
    quality['duplicate_file_content'] = len({r['md5'] for r in quality['files']}) != 28
    quality['interval_minutes'] = sorted(set((frame.index[1:] - frame.index[:-1]).total_seconds()/60))
    quality['daily_slots'] = {str(k):int(v) for k,v in frame.groupby(frame.index.tz_convert('Europe/Rome').date).size().items()}
    quality['nonfinite_observed_values'] = int((~np.isfinite(frame.internet.dropna())).sum())
    # Potential outliers are flagged for inspection only, never removed.
    q1,q3 = frame.internet.quantile([.25,.75])
    quality['iqr_outlier_count_not_removed'] = int(((frame.internet<q1-1.5*(q3-q1)) | (frame.internet>q3+1.5*(q3-q1))).sum())
    config = json.loads((ROOT/'configs/experiment.json').read_text())
    n = len(frame)
    train_end = int(n*config['train_fraction'])
    val_end = int(n*(config['train_fraction']+config['validation_fraction']))
    quality['split_rows'] = {'train':train_end,'validation':val_end-train_end,'test':n-val_end}
    quality['split_boundaries_local'] = {'validation_start':str(frame.index[train_end].tz_convert('Europe/Rome')),
                                        'test_start':str(frame.index[val_end].tz_convert('Europe/Rome'))}
    quality['train_scaler'] = features(frame,train_end,False)[2]
    quality['sample_counts'] = {h:{s:len(p) for s,p in sample_indices(frame.internet.to_numpy(),144,h*6,train_end,val_end).items()} for h in config['horizon_hours']}
    frame.to_csv(out/'series.csv')
    (out/'quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False),encoding='utf-8')
    plot_series(frame,out/'traffic.png',1)
    blockers = []
    if len(frame) != 4032 or len(quality['daily_slots']) != 28:
        blockers.append('Unexpected data period/point count')
    if quality['interval_minutes'] != [10.0]:
        blockers.append('Irregular time intervals')
    if quality['missing_internet_slots']:
        blockers.append('Missing aggregate Internet observations')
    if quality['nonfinite_observed_values'] or quality['duplicate_file_content']:
        blockers.append('Invalid values or duplicate file content')
    quality['training_blockers'] = blockers
    (out/'quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in quality.items() if k not in ('files','selected_country_codes','daily_slots')},indent=2),flush=True)
    if blockers:
        raise SystemExit('STOP BEFORE TRAINING: '+ '; '.join(blockers))
    print('PASS: fixed 28-day series ready for training',flush=True)


if __name__ == '__main__':
    main()
