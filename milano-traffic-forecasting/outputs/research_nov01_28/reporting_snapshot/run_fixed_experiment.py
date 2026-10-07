"""Run unchanged experiment logic in independent CPU processes by horizon."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/research_nov01_28'


def launch(horizon):
    with (OUT/f'{horizon}h.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-m','milano.experiment','--series','outputs/nov01_28/series.csv',
                        '--config',str(OUT/f'config_{horizon}h.json'),
                        '--out',str(OUT/f'{horizon}h')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return horizon


def main():
    quality = json.loads((ROOT/'outputs/nov01_28/quality.json').read_text())
    if quality['training_blockers'] or quality['expected_slots'] != 4032 or quality['square_id'] != 1:
        raise ValueError('Data validation has not passed')
    config = json.loads((ROOT/'configs/experiment.json').read_text())
    assert config['seeds'] == [42,43,44] and config['horizon_hours'] == [1,3,6,12,24]
    OUT.mkdir(exist_ok=False)
    (OUT/'source_snapshot').mkdir()
    for path in (ROOT/'milano').glob('*.py'):
        shutil.copy2(path, OUT/'source_snapshot'/path.name)
    shutil.copy2(ROOT/'requirements-lock.txt', OUT/'requirements-lock.txt')
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'square_id':1,
                'period_local':['2013-11-01','2013-11-28'],'config':config,
                'execution':'3 independent CPU processes maximum; horizon partitions only',
                'series_sha256':hashlib.sha256((ROOT/'outputs/nov01_28/series.csv').read_bytes()).hexdigest(),
                'status':'running','completed_horizons':[]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for horizon in config['horizon_hours']:
        (OUT/f'config_{horizon}h.json').write_text(json.dumps(dict(config,horizon_hours=[horizon]),indent=2))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(launch,h):h for h in config['horizon_hours']}
        while pending:
            done,_ = concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                h = pending.pop(future)
                future.result()
                manifest['completed_horizons'].append(h)
                (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
                print(f'COMPLETED horizon={h}h; {len(manifest["completed_horizons"])}/5',flush=True)
            if not done:
                histories = len(list(OUT.glob('*h/*_history.csv')))
                print(f'PROGRESS {histories}/30 GRU fits completed, {time.monotonic()-started:.0f}s elapsed',flush=True)
    scores = pd.concat([pd.read_csv(OUT/f'{h}h/metrics.csv') for h in config['horizon_hours']],ignore_index=True)
    assert len(scores) == 35
    scores.to_csv(OUT/'metrics.csv',index=False)
    scores.groupby(['model','horizon_hours'])[['r2','rmse','mae']].agg(['mean','std']).to_csv(OUT/'summary.csv')
    base = scores[scores.model=='traffic_gru'].set_index(['horizon_hours','seed'])
    enhanced = scores[scores.model=='calendar_gru'].set_index(['horizon_hours','seed'])
    delta = enhanced[['r2','rmse','mae']]-base[['r2','rmse','mae']]
    delta.columns = ['delta_r2','delta_rmse','delta_mae']
    delta['rmse_reduction_percent'] = (base.rmse-enhanced.rmse)/base.rmse*100
    delta['mae_reduction_percent'] = (base.mae-enhanced.mae)/base.mae*100
    delta.to_csv(OUT/'paired_differences.csv')
    delta.groupby('horizon_hours').agg(['mean','std']).to_csv(OUT/'paired_differences_summary.csv')
    manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE: fixed conditions; 30 GRU fits and 5 persistence evaluations',flush=True)


if __name__ == '__main__':
    main()
