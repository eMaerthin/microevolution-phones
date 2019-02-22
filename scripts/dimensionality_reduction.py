from functools import reduce
from itertools import compress

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.manifold import TSNE


def fit_tsne(list_X, n_components=2, perplexity=30, n_iter=1000, n_iter_without_progress=300):
    assert (n_components is 2)
    X = np.concatenate(list_X)
    z, _ = reduce(lambda cum, cur: (cum[0] + ([cum[1]] * len(cur)), cum[1] + 1), list_X, ([], 0))
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, n_iter_without_progress=n_iter_without_progress).fit_transform(X)
    Xs = []
    for i in range(len(list_X)):
        Xs.append([xj for xj, zj in zip(tsne, z) if zj == i])
    return Xs


def draw_tsne(tsne_Xs, ys, n_components=2, title='tsne', filter_phonemes=None):
    assert(n_components is 2)
    len_x_vector = len(tsne_Xs)
    assert(len_x_vector == len(ys))
    plt.figure()
    colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]

    for i in range(0, len_x_vector):
        subset = [True] * len(ys[i])
        if filter_phonemes:
            subset = [any(substring in phoneme for substring in filter_phonemes) for phoneme in ys[i]]
        x1, x2 = zip(*tsne_Xs[i])
        x1 = list(compress(x1, subset))
        x2 = list(compress(x2, subset))
        y = list(compress(ys[i], subset))
        plt.subplot(len_x_vector, 1, i + 1, aspect='equal')
        plt.title(title)
        color = [colors[hash(v_y) % len(colors)] for v_y in y]
        plt.scatter(x1, x2, c=color, s=1)
    plt.show()
