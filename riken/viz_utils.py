import numpy as np
import matplotlib.pyplot as plt
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


def to_2d_array(arr, ncols):
    nrows = (len(arr) // ncols) + 1
    arr2d = [[0]*ncols for _ in range(nrows)]
    for idx, val in enumerate(arr):
        i = idx // ncols
        j = idx % ncols
        arr2d[i][j] = val
    return np.array(arr2d)


def visualize_attention_2d(sequence, attention, ncols=50):
    heatmap_2d = to_2d_array(attention, ncols=ncols)
    nrows, ncols = heatmap_2d.shape
    str_2d = to_2d_array(sequence, ncols=ncols)
    ax = sns.heatmap(data=heatmap_2d, annot=str_2d, fmt='', xticklabels=False, yticklabels=False)
    return ax
