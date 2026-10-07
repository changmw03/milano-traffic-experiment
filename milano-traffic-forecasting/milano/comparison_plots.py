"""Descriptive figures only; never select models or tune on test results."""
from . import report  # Configure writable matplotlib cache and Agg backend.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

LABELS = {'traffic_gru':'Traffic GRU','calendar_gru':'Calendar GRU','persistence':'Persistence'}
COLORS = {'traffic_gru':'#247ba0','calendar_gru':'#e17c05','persistence':'#66756b'}


def performance(scores, out):
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,metric in zip(axes,['r2','rmse','mae']):
        for model in LABELS:
            table = scores[scores.model==model].groupby('horizon_hours')[metric].agg(['mean','std'])
            ax.errorbar(table.index,table['mean'],yerr=table['std'].fillna(0),
                        marker='o',capsize=3,label=LABELS[model],color=COLORS[model])
        ax.set(xlabel='Forecast horizon (hours)',ylabel=metric.upper(),xticks=[1,3,6,12,24])
        ax.grid(alpha=.2)
    axes[0].axhline(0,color='black',alpha=.25,linewidth=.8)
    axes[-1].legend(fontsize=9)
    fig.suptitle('Square 1 | Fixed 28-day experiment | Mean +/- sample SD over seeds 42, 43, 44\nPersistence is deterministic; error bars represent seed variation, not confidence intervals',fontsize=11)
    fig.savefig(out/'horizon_performance.png',dpi=170)
    plt.close(fig)


def differences(delta, out):
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['delta_r2','delta_rmse','delta_mae']):
        g = delta.groupby('horizon_hours')[metric].agg(['mean','std'])
        ax.errorbar(g.index,g['mean'],yerr=g['std'],marker='o',capsize=4,color='#8655a3')
        ax.axhline(0,color='black',linewidth=1)
        ax.set(xlabel='Horizon (hours)',ylabel=metric,xticks=[1,3,6,12,24])
        ax.grid(alpha=.2)
    fig.suptitle('Paired difference: Calendar GRU minus Traffic GRU\nPositive R2 / negative RMSE and MAE indicate improvement',fontsize=11)
    fig.savefig(out/'paired_differences.png',dpi=170)
    plt.close(fig)


def forecast_comparison(out, series, horizons):
    fig, axes = plt.subplots(5,1,figsize=(14,15),layout='constrained')
    for ax,h in zip(axes,horizons):
        base = pd.read_csv(out/f'{h}h/traffic_gru_{h}h_seed42_predictions.csv')
        enhanced = pd.read_csv(out/f'{h}h/calendar_gru_{h}h_seed42_predictions.csv')
        times = pd.to_datetime(base.target_time,utc=True)
        origins = pd.to_datetime(base.origin_time,utc=True)
        persistence = series.loc[origins,'internet'].to_numpy()
        ax.plot(times,base.actual,label='Actual',color='#222222',linewidth=1.3)
        ax.plot(times,base.predicted,label=LABELS['traffic_gru'],color=COLORS['traffic_gru'],alpha=.85,linewidth=.9)
        ax.plot(times,enhanced.predicted,label=LABELS['calendar_gru'],color=COLORS['calendar_gru'],alpha=.85,linewidth=.9)
        ax.plot(times,persistence,label='Persistence',color=COLORS['persistence'],alpha=.65,linewidth=.8)
        ax.set(title=f'{h}h horizon | {len(times)} matched test targets',ylabel='Internet activity')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M',tz=ZoneInfo('Europe/Rome')))
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=9)
    axes[-1].set_xlabel('Target time (Europe/Rome)')
    fig.suptitle('Actual vs predictions | Seed 42 chosen before training\nModels evaluated at identical test timestamps within each horizon',fontsize=12)
    fig.savefig(out/'actual_vs_predictions_seed42.png',dpi=140)
    plt.close(fig)
