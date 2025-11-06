import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from matplotlib import rc
from dataclasses import dataclass


clrs = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22','#17becf']

sb.set_style('whitegrid')
sb.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

rc('text', usetex=False)


def plot_entropy(results, separate_ID=False):
    common_kwargs = dict(stat='probability', kde=False, bins=12, binrange=[0, 2.4], label="dummy", legend=False,
                         element="step", alpha=0.7)
    id_kwargs = dict(
              hue="category", multiple="stack",
              palette=[sb.color_palette()[0], sb.color_palette()[4]]) if separate_ID else dict(color=sb.color_palette()[0])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(2.5, 2.5/1.6), gridspec_kw={'height_ratios': [1, 3]}, constrained_layout=True)
    fig.subplots_adjust(hspace=0.30)  # adjust space between axes

    for ax in axes:
        sb.histplot(data=results.bimacs, x="entropy", **id_kwargs,
                    **common_kwargs, ax=ax)
        sb.histplot(results.fashion_mnist, color=sb.color_palette()[1],
                    **common_kwargs, ax=ax)

    axes[0].set_ylim(0.4, 0.55)  # outliers only
    axes[1].set_ylim(0, .15)  # most of the data

    axes[0].spines['bottom'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[0].set_ylabel("")
    axes[1].set_ylabel("Fraction", fontsize=12)
    axes[1].set_xlabel("Entropy", fontsize=12)

    axes[1].yaxis.set_label_coords(-0.04, 0.5, fig.transFigure)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)

    fig.set_size_inches(2.5, 2.5/1.6)

    return fig, axes

separate_ID = False



plot_entropy(test_entropy, separate_ID=separate_ID)
plt.savefig('pdfs/entropy_hist_test.pdf', bbox_inches='tight')
plt.savefig('pngs/entropy_hist_test.png', bbox_inches='tight')
