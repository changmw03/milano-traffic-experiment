"""Verify saved test predictions against the frozen series and metrics."""
import json
import numpy as np
import pandas as pd


def audit(out, series, scores, horizons, seeds):
    histories = []
    checks = 0
    for h in horizons:
        reference = None
        for model in ['traffic_gru','calendar_gru']:
            for seed in seeds:
                prefix = f'{model}_{h}h_seed{seed}'
                table = pd.read_csv(out/f'{h}h/{prefix}_predictions.csv')
                origins = pd.to_datetime(table.origin_time,utc=True)
                targets = pd.to_datetime(table.target_time,utc=True)
                assert ((targets-origins)==pd.Timedelta(hours=h)).all()
                actual = series.loc[targets,'internet'].to_numpy()
                np.testing.assert_allclose(table.actual,actual,rtol=1e-12)
                current = (list(table.origin_time),list(table.target_time))
                if reference is None:
                    reference = current
                assert current == reference, 'Models/seeds evaluated on different samples'
                # GRU predictions are float32. CSV uses their shortest round-trip
                # representation; restore float32 before comparing with the
                # original in-memory metrics, rather than treating the printed
                # decimal as a new float64 prediction.
                predicted = table.predicted.to_numpy(dtype=np.float32).astype(np.float64)
                difference = actual-predicted
                values = [1-np.sum(difference**2)/np.sum((actual-actual.mean())**2),
                          np.sqrt(np.mean(difference**2)),np.mean(np.abs(difference))]
                row = scores[(scores.model==model)&(scores.seed==seed)&(scores.horizon_hours==h)].iloc[0]
                np.testing.assert_allclose(values,row[['r2','rmse','mae']].to_numpy(dtype=float),rtol=1e-8,atol=1e-10)
                history = pd.read_csv(out/f'{h}h/{prefix}_history.csv')
                histories.append({'model':model,'horizon_hours':h,'seed':seed,'epochs_run':len(history),
                                  'best_epoch':int(history.loc[history.validation_mse_scaled.idxmin(),'epoch']),
                                  'best_validation_mse_scaled':float(history.validation_mse_scaled.min())})
                checks += 1
        persistence = series.loc[origins,'internet'].to_numpy()
        difference = actual-persistence
        values = [1-np.sum(difference**2)/np.sum((actual-actual.mean())**2),
                  np.sqrt(np.mean(difference**2)),np.mean(np.abs(difference))]
        row = scores[(scores.model=='persistence')&(scores.horizon_hours==h)].iloc[0]
        np.testing.assert_allclose(values,row[['r2','rmse','mae']].to_numpy(dtype=float),rtol=1e-8,atol=1e-10)
        pd.DataFrame({'origin_time':origins,'target_time':targets,'actual':actual,'predicted':persistence}).to_csv(out/f'{h}h/persistence_{h}h_predictions.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'training_summary.csv',index=False)
    result = {'gru_prediction_files_checked':checks,'persistence_horizons_checked':len(horizons),
              'identical_targets_across_models_and_seeds':True,'direct_horizon_alignment':True,
              'actual_values_match_frozen_series':True,'all_saved_metrics_recomputed_and_matched':True}
    (out/'result_audit.json').write_text(json.dumps(result,indent=2))
    return result
