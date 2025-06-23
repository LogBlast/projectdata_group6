import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import *

class LabelVisualiation :
    def display_labels_proportion_to_target(df : pd.DataFrame, target : str = 'EngagementLevel') :
        for label in LABELS_LIST:
            if label == target :
                continue

            crosstab = pd.crosstab(df[label], df[target], normalize='index') * 100
            barplot = crosstab.plot(kind='bar', stacked=True)

            if pd.api.types.is_numeric_dtype(df[label]):
                xticks = barplot.get_xticks()
                tick_labels = [lbl.get_text() for lbl in barplot.get_xticklabels()]
                if len(xticks) >= 2:
                    barplot.set_xticks([xticks[0], xticks[-1]])
                    barplot.set_xticklabels([tick_labels[0], tick_labels[-1]])

            plt.title(f'Engagement Level by {label}')
            plt.xlabel(label)
            plt.ylabel('Percentage')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.close()