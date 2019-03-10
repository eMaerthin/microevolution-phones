from functools import reduce
from itertools import product
from random import shuffle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from scipy.interpolate import interp2d
from scipy.spatial.distance import (pdist, squareform)
from skimage.filters.rank import windowed_histogram
from skimage.morphology import disk
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

NORMALIZATION_FACTOR = 1.0  # 8.0  # 255.0


def fit_tsne(list_X, n_components=2, perplexity=30, n_iter=1000, n_iter_without_progress=300):
    assert (n_components is 2)
    X = np.concatenate(list_X)
    z, _ = reduce(lambda cum, cur: (cum[0] + ([cum[1]] * len(cur)), cum[1] + 1), list_X, ([], 0))
    tsne = TSNE(n_jobs=4, n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, n_iter_without_progress=n_iter_without_progress).fit_transform(X)
    Xs = []
    for i in range(len(list_X)):
        Xs.append([xj for xj, zj in zip(tsne, z) if zj == i])
    return Xs


def fit_pca(list_X, n_components=2, verbose=0):
    pca = PCA(n_components=n_components)
    X = np.concatenate(list_X)
    pca.fit(X)
    if verbose > 0:
        print(f' pca explained variance ratio: {pca.explained_variance_ratio_}')
        print(f' pca singular values: {pca.singular_values_}')
    return pca

def draw_centers(Xs, ys, result_path, n_components=2, verbose=0):
    assert(n_components is 2)
    fig, ax = plt.subplots(nrows=1, ncols=1)


def draw_composition(Xs, ys, titles, result_path, n_components=2, verbose=0,
              out_shape=None, agg_shape=None, drawing_subsamples=False, suptitle=None):
    if not out_shape:
        out_shape = (1, len(ys))
    if not agg_shape:
        agg_shape = (1, 1)
    assert(agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1] is len(ys))
    assert(n_components is 2)
    len_x_vector = len(Xs)
    assert(len_x_vector == len(ys))
    fig, axes = plt.subplots(nrows=out_shape[0], ncols=out_shape[1], sharex=True, sharey=True, squeeze=False)
    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
    mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed')

    axes1d = axes.flatten()
    r = np.arange(len(ys)).reshape(out_shape[0] * agg_shape[0], out_shape[1] * agg_shape[1])
    a1, a2 = agg_shape
    indices = [r[a1 * i:a1 * (i + 1), a2 * j: a2 * (j + 1)].reshape(-1) for i, j in product(range(out_shape[0]),
                                                                                            range(out_shape[1]))]
    # print(indices)
    x_list = []
    y_list = []
    t_list = []

    for i in indices:
        x_list.append(sum((Xs[j] for j in i), []))
        y_list.append(sum((ys[j] for j in i), []))
        t_list.append(f'S{1 + 2 * (i // 15) }_D{1 + (i % 15)}') #titles[i[0]])
    if drawing_subsamples:
        min_samples = np.inf
        for yi in y_list:
            min_samples = min(min_samples, len(yi))

    if suptitle:
        suptitle = suptitle + f' remaining pts: {min_samples}'
    colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
    metrics = []
    print(x_list[0])
    min_x1, min_x2, max_x1, max_x2 = \
        reduce(lambda cum, cur: (min(cum[0], min(cur[0])),
                                 min(cum[1], min(cur[1])),
                                 max(cum[2], max(cur[0])),
                                 max(cum[3], max(cur[1]))),
               zip(*x_list), (np.inf, np.inf, -np.inf, -np.inf))
    print(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
    vmax=None
    for ax, title, x, y in zip(axes1d, t_list, x_list, y_list):
        x1, x2 = zip(*x)
        # y = ys[i]
        remaining_labels = sorted(list(set(y)))
        '''for label in remaining_labels:
            p1 = [p for l, p in zip(y, x1) if l is label]
            p2 = [p for l, p in zip(y, x2) if l is label]
            cx = np.mean(p1)
            cy = np.mean(p2)
            radius = max(np.std(p1),np.std(p2))
            circle = plt.Circle((cx, cy), radius, color='gray', zorder=1)
            ax.add_artist(circle)'''
        if verbose > 0:
            print(f'remaining labels: {remaining_labels} (#pts in total: {len(y)})')
        # plt.subplot(len_x_vector, 1, i + 1, aspect='equal')
        # ax.title(title)
        color = [colors[remaining_labels.index(v_y) % len(colors)] for v_y in y]
        if drawing_subsamples:
            together = list(zip(color, x1, x2))
            shuffle(together)
            color, x1, x2 = zip(*together)
            color = color[:min_samples]
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            if verbose > 0:
                print(f'remaining pts: {len(color)}')
        ax.scatter(x1, x2, c=color, s=1, zorder=2, alpha=0.0015)
        bins = [7, 7]  # [41, 41]  # [7, 7]  # [41, 41]
        disk_radius = 2  # 6  # 6  # 3  # 6
        h, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins, density=False,
                                             range=[[min_x1, max_x1], [min_x2, max_x2]]) #, bins=(xedges, yedges))
        h = h.astype(np.uint8)
        hist_img = windowed_histogram(h, disk(disk_radius))
        #hist_img_max = np.argmax(hist_img, axis=2)
        hist_img_max = h.copy()

        for ix, iy in np.ndindex(hist_img_max.shape):
            hist_img_max[ix, iy] = np.sum([NORMALIZATION_FACTOR * i_elem * elem for i_elem, elem in enumerate(hist_img[ix][iy])])
        # print(hist_img_max)
        h = hist_img_max.T #h.T
        if not vmax:
            vmax = np.max(h)
        # print(h)
        # print(f'{np.max(h)} vs {1/81/(x_edges[1]-x_edges[0])/(y_edges[1]-y_edges[0])}')
        m1, m2 = np.meshgrid(x_edges, y_edges)
        print(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0, vmax=vmax, cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics)))/NORMALIZATION_FACTOR
    print(distance_matrix)
    if suptitle:
        # suptitle=suptitle + '\nDistances from the first picture: '+np.array2string(distance_matrix[0], precision=1)
        fig.suptitle(suptitle, fontsize=8)
    plt.savefig(result_path, dpi='figure', frameon=True, bbox_inches=None)
    # plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False)
    axes1d = axes.flatten()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    part2_x = [np.array(metrics)]
    part2_tsne = fit_tsne(part2_x)
    print(f'tsne: {part2_tsne}')
    p2_x1, p2_x2 = zip(*part2_tsne[0])
    kf = out_shape[1] // agg_shape[0]
    color=['red']*kf + ['green']*kf + ['blue']*kf
    for ax in axes1d:
        ax.scatter(p2_x1, p2_x2, c=color)
        for (i, (x, y)) in enumerate(zip(p2_x1, p2_x2)):
            ax.text(x, y, f'S{1 + 2 * (i // kf) }_D{1 + (i % kf)}')
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', frameon=True, bbox_inches=None)
    # plt.show()



