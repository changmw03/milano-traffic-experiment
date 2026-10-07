"""Figures use original activity units and Milan local time."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/.matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo


def plot_series(frame, path, square):
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    ax.plot(frame.index, frame.internet, color='#177f85', linewidth=1.8)
    ax.set(title=f'Milano | Square {square} | 10-minute Internet activity',
           xlabel='Local time (Europe/Rome)', ylabel='Internet activity (not MB or Mbps)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M', tz=ZoneInfo('Europe/Rome')))
    ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_predictions(table, path, title):
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    ax.plot(table.target_time, table.actual, label='Actual', linewidth=1)
    ax.plot(table.target_time, table.predicted, label='Predicted', linewidth=1)
    ax.set(title=title, ylabel='Internet activity', xlabel='Target time (UTC)')
    ax.legend()
    fig.savefig(path, dpi=130)
    plt.close(fig)
