"""Plot the real pair contributions and the two shared synthetic recordings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    for extension in ['png', 'pdf']:
        fig.savefig(HERE / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def real_plot(data):
    pairs = pd.DataFrame(data['real']['pairs'])
    pairs['pair'] = np.arange(1, len(pairs) + 1)
    pairs['local_fraction'] = pairs.duration_s / (pairs.duration_s + pairs.following_gap_s)
    pairs['duration_weight'] = pairs.duration_s / pairs.duration_s.sum()
    pairs['contribution_to_score'] = pairs.local_fraction * pairs.duration_weight
    pairs.to_csv(HERE / 'real_pair_contributions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=.77, bottom=.22, left=.09, right=.98, wspace=.35)
    fig.suptitle('The real recording contains four eligible pairs', x=.09, ha='left',
                 fontsize=20, fontweight='bold', y=.96)
    fig.text(.09, .85, f"Team A, August 3, 2020.  Activity–gap concentration = {data['real']['activity_gap_concentration']:.6f}")
    axes[0].scatter(pairs.duration_s, pairs.following_gap_s, s=85, color='#007C78')
    for row in pairs.itertuples():
        axes[0].annotate(str(row.pair), (row.duration_s, row.following_gap_s),
                         xytext=(8, 4), textcoords='offset points')
    axes[0].set(xlabel='Activity duration D (s)', ylabel='Following gap G (s)',
                xlim=(1.45, 2.55), ylim=(0, 640))
    bars = axes[1].barh(pairs.pair, pairs.contribution_to_score, color='#007C78')
    axes[1].bar_label(bars, fmt='%.5f', padding=5, fontsize=11)
    axes[1].set(xlabel='Contribution to the final score', ylabel='Pair number',
                yticks=pairs.pair, xlim=(0, .0205), xticks=[0, .005, .010, .015, .020])
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
    fig.text(.09, .08, 'The 2.4-second activity followed by a 44.5-second gap contributes most to this score.', fontsize=12)
    fig.text(.09, .025, 'Gaps separate retained activities. Four pairs support this illustration. Training boundaries remain unconfirmed.', fontsize=10.5)
    save(fig, 'real_pair_contributions')


def synthetic_plot(data):
    traces = pd.read_csv(HERE / 'synthetic_raw_signals.csv')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(top=.79, bottom=.17, left=.09, right=.98, hspace=.60)
    fig.suptitle('Same window summaries, different activity–gap pairing', x=.09, ha='left',
                 fontsize=19, fontweight='bold', y=.96)
    fig.text(.09, .88, 'Constructed signals. Both Jack scores = −1.80/min. The candidate separates the pairings.')
    for ax, case, label in zip(axes, data['synthetic'], ['A', 'B']):
        raw = traces.loc[traces.case == case['case']].copy()
        time = pd.to_timedelta(raw.time).dt.total_seconds()
        ax.step(time, raw.speed, where='post', color='#526675', linewidth=1.7)
        for pair in case['pairs']:
            start, end, next_start = pair['elapsed_start_s'], pair['end_s'], pair['next_start_s']
            ax.axvspan(start, end, color='#007C78', alpha=.18)
            ax.text((start + end) / 2, 5.55, f"D={pair['duration_s']:g}s", ha='center', fontsize=11)
            ax.text((end + next_start) / 2, 1.0, f"G={pair['following_gap_s']:g}s", ha='center', color='#A65D19', fontsize=11)
        ax.text(23, 5.55, 'final activity', ha='center', fontsize=10, color='#526675')
        ax.set_title(f"Example {label}     C = {case['activity_gap_concentration']:.2f}", loc='left', fontsize=14, pad=10)
        ax.set(ylabel='Speed (m/s)', ylim=(-.3, 7.0), yticks=[0, 5], xlim=(0, 30))
        ax.grid(axis='x', alpha=.15)
    axes[-1].set_xlabel('Elapsed seconds, first 30 seconds of each 120-second signal')
    fig.text(.09, .07, 'Paired activity durations are {2, 8}s and gaps are {2, 8}s in both cases. The final activity has no following pair.', fontsize=11)
    fig.text(.09, .025, 'The remaining 90 seconds have zero speed. These abrupt signals test the calculations, not athlete physiology.', fontsize=11)
    save(fig, 'synthetic_pairing_comparison')


if __name__ == '__main__':
    data = json.loads((HERE / 'comparison.json').read_text(encoding='utf-8'))
    real_plot(data)
    synthetic_plot(data)
    print('Saved real and synthetic pairing figures as PNG and PDF.')
