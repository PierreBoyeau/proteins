import numpy as np
import seaborn as sns


def visualize_attention(sequence, attention, log_viz=True):
    if len(attention) > len(sequence):
        attention_shr = attention[-len(sequence):]
    elif len(attention) == len(sequence):
        attention_shr = attention.copy()
    else:
        raise ValueError
    attention_shr = attention_shr.reshape(1, -1)
    if log_viz:
        attention_shr = np.log(attention_shr)

    annot = np.array(list(sequence))  # .reshape(1, -1)
    ax = sns.heatmap(attention_shr, yticklabels=False)
    ax.set_xticklabels(labels=annot, rotation=0)
    return ax