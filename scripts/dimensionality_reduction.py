from functools import reduce
from itertools import product
from random import shuffle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from scipy.interpolate import interp2d
from scipy.spatial.distance import (pdist, squareform)
from skimage.filters.rank import windowed_histogram
from skimage.morphology import disk
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

NORMALIZATION_FACTOR = 1.0  # 8.0  # 255.0


def fit_tsne(list_X, n_components=2, perplexity=30, n_iter=1000, n_iter_without_progress=300, verbose=0):
    assert (n_components is 2)
    X = np.concatenate(list_X)
    z, _ = reduce(lambda cum, cur: (cum[0] + ([cum[1]] * len(cur)), cum[1] + 1), list_X, ([], 0))
    tsne = TSNE(n_jobs=4, n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, n_iter_without_progress=n_iter_without_progress)
    tsne_out = tsne.fit_transform(X)
    if verbose > 0:
        print(f'tsne.kl_divergence: {tsne.kl_divergence_}')
    Xs = []
    for i in range(len(list_X)):
        Xs.append([xj for xj, zj in zip(tsne_out, z) if zj == i])
    return Xs


def fit_pca(list_X, n_components=2, verbose=0):
    pca = PCA(n_components=n_components)
    X = np.concatenate(list_X)
    pca.fit(X)
    if verbose > 0:
        print(f' pca explained variance ratio: {pca.explained_variance_ratio_}')
        cum = [pca.explained_variance_ratio_[0]]
        for i in range(1, len(pca.explained_variance_ratio_)):
            cum.append(cum[i - 1] + pca.explained_variance_ratio_[i])
        print(f' pca explained variance ratio cum: {cum}')
        print(f' pca singular values: {pca.singular_values_}')
    return pca

def draw_centers(Xs, ys, result_path, n_components=2, verbose=0):
    assert(n_components is 2)
    fig, ax = plt.subplots(nrows=1, ncols=1)


def parse_validation(validation=None):
    validated = False
    if validation:
        pass  # TODO
    return validated


def draw_composition(Xs, ys, result_path, n_components=2, verbose=0,
                     out_shape=None, agg_shape=None, drawing_subsamples=False,
                     sup_title=None, validation=None):
    if not out_shape:
        out_shape = (1, len(ys))
    if not agg_shape:
        agg_shape = (1, 1)
    estimated_series_size = agg_shape[1] * out_shape[1]
    assert(agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1] is len(ys))
    assert n_components is 2, f'it should be 2 - currently it is {n_components}'
    len_x_vector = len(Xs)
    assert(len_x_vector == len(ys))
    fig_rows, fig_cols = out_shape
    validation_correct = parse_validation(validation)
    if validation_correct:
        fig_cols += 1
    fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, sharex=True, sharey=True, squeeze=False)
    mng = plt.get_current_fig_manager()
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
        title = f'S{1 + 2 * (i[0] // estimated_series_size)}_D{1 + (i[0] % estimated_series_size)}'
        if len(i) > 1:
            title = title + f' - S{1 + 2 * (i[-1] // estimated_series_size) }_D{1 + (i[-1] % estimated_series_size)}'
        t_list.append(title)
    if drawing_subsamples:
        min_samples = np.inf
        for yi in y_list:
            if verbose > 0:
                print(f'checking minimum len(y). current: {len(yi)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(yi))

    if sup_title:
        sup_title = sup_title + f' remaining pts: {min_samples}'
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
    vmax = None
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
        bins = [41, 41]  # [7, 7]  # [41, 41]
        disk_radius = 6  # 6  # 3  # 6
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
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    print(distance_matrix)
    if sup_title:
        # sup_title = sup_title + '\nDistances from the first picture: '+np.array2string(distance_matrix[0], precision=1)
        fig.suptitle(sup_title, fontsize=8)
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
    color = ['red']*kf + ['green']*kf + ['blue']*kf
    for ax in axes1d:
        ax.scatter(p2_x1, p2_x2, c=color)
        for (i, (x, y)) in enumerate(zip(p2_x1, p2_x2)):
            ax.text(x, y, f'S{1 + 2 * (i // kf) }_D{1 + (i % kf)}')
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', frameon=True, bbox_inches=None)
    # plt.show()


def draw_X(Xs, result_path, n_components=2, verbose=0,
           out_shape=None, agg_shape=None, drawing_subsamples=False,
           sup_title=None, validation=None):
    if not out_shape:
        out_shape = (1, len(Xs))
    if not agg_shape:
        agg_shape = (1, 1)
    estimated_series_size = agg_shape[1] * out_shape[1]
    assert agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1] is len(Xs), f'{agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1]} vs {len(Xs)}'
    assert n_components is 2, f'it should be 2 - currently it is {n_components}'
    fig_rows, fig_cols = out_shape
    validation_correct = parse_validation(validation)
    if validation_correct:
        fig_cols += 1
    plt.rcParams.update({'font.size': 7})
    fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, sharex=True, sharey=True, squeeze=False)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed')

    axes1d = axes.flatten()
    r = np.arange(len(Xs)).reshape(out_shape[0] * agg_shape[0], out_shape[1] * agg_shape[1])
    a1, a2 = agg_shape
    indices = [r[a1 * i:a1 * (i + 1), a2 * j: a2 * (j + 1)].reshape(-1) for i, j in product(range(out_shape[0]),
                                                                                            range(out_shape[1]))]
    # print(indices)
    x_list = []
    t_list = []

    for i in indices:
        x_list.append(sum((Xs[j] for j in i), []))
        title = f'S{1 + (i[0] // estimated_series_size)}_D{1 + (i[0] % estimated_series_size)}'
        #if len(i) > 1:
        #    title = title + f' - S{1 + (i[-1] // estimated_series_size) }_D{1 + (i[-1] % estimated_series_size)}'
        t_list.append(title)
    if drawing_subsamples:
        min_samples = np.inf
        for xi in x_list:
            if verbose > 0:
                print(f'checking minimum len(x). current: {len(xi)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(xi))

    if sup_title:
        sup_title = sup_title + f' remaining pts: {min_samples}'
    metrics = []
    min_x1, min_x2, max_x1, max_x2 = \
        reduce(lambda cum, cur: (min(cum[0], min(cur[0])),
                                 min(cum[1], min(cur[1])),
                                 max(cum[2], max(cur[0])),
                                 max(cum[3], max(cur[1]))),
               zip(*x_list), (np.inf, np.inf, -np.inf, -np.inf))
    if verbose > 0:
        print(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
    vmax = None
    for ax, title, x in zip(axes1d, t_list, x_list):
        x1, x2 = zip(*x)
        if drawing_subsamples:
            together = list(zip(x1, x2))
            shuffle(together)
            x1, x2 = zip(*together)
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            if verbose > 0:
                print(f'remaining pts: {len(x1)}')
        ax.scatter(x1, x2, c='red', s=1, zorder=2, alpha=0.0015)
        bins = [30, 30]  # [41, 41]  # [7, 7]  # [41, 41]
        disk_radius = 2  # 6  # 6  # 3  # 6
        h, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins, density=False,
                                             range=[[min_x1, max_x1], [min_x2, max_x2]]) #, bins=(xedges, yedges))
        h = h.astype(np.uint8)
        hist_img = windowed_histogram(h, disk(disk_radius))
        hist_img_max = h.copy()

        for ix, iy in np.ndindex(hist_img_max.shape):
            hist_img_max[ix, iy] = np.sum([NORMALIZATION_FACTOR * i_elem * elem for i_elem, elem in enumerate(hist_img[ix][iy])])
        h = hist_img_max.T #h.T
        if not vmax:
            vmax = np.max(h)
        m1, m2 = np.meshgrid(x_edges, y_edges)
        if verbose > 1:
            print(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0, vmax=vmax, cmap='gray_r')  # cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    if verbose > 0:
        print(distance_matrix)
    if sup_title:
        # sup_title = sup_title + '\nDistances from the first picture: '+np.array2string(distance_matrix[0], precision=1)
        fig.suptitle(sup_title, fontsize=8)
    plt.savefig(result_path, dpi=1200, frameon=True, bbox_inches=None)
    # plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False)
    axes1d = axes.flatten()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    part2_x = [np.array(metrics)]
    part2_tsne = fit_tsne(part2_x)
    if verbose > 0:
        print(f'tsne: {part2_tsne}')
    p2_x1, p2_x2 = zip(*part2_tsne[0])
    kf = out_shape[1] // agg_shape[0]
    colours = ['red', 'green', 'blue', 'black', 'magenta', 'yellow']
    color = [i for s in [[c] * out_shape[1] for c in colours[:out_shape[0]]] for i in s]
    for ax in axes1d:
        ax.scatter(p2_x1, p2_x2, c=color)
        for (i, (x, y)) in enumerate(zip(p2_x1, p2_x2)):
            ax.text(x, y, f'S{1 + (i // kf) }_D{1 + (i % kf)}')
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', frameon=True, bbox_inches=None)
    # plt.show()


def draw_X2(Xs, result_path, n_components=2, verbose=0,
           out_shape=None, agg_lists=None, drawing_subsamples=False,
           sup_title=None, validation=None):
    def split_seq(seq, size):
        newseq = []
        split_size = 1.0 / size * len(seq)
        for i in range(size):
            newseq.append(seq[int(round(i * split_size)):int(round((i + 1) * split_size))])
        return newseq

    if not out_shape:
        out_shape = (1, len(Xs))  # (rows, columns)
    if not agg_lists:
        agg_lists = split_seq(list(range(len(Xs))), out_shape[0] * out_shape[1])
    symmetric_diff = set([item for s in agg_lists for item in s]).symmetric_difference(range(len(Xs)))
    assert len(symmetric_diff) is 0, f'agg_lists: {agg_lists}, symmetric_difference with range({len(Xs)}) is: {symmetric_diff}'
    assert len(agg_lists) == (out_shape[0] * out_shape[1]), f'{len(agg_lists)} vs {out_shape[0] * out_shape[1]}'
    assert n_components is 2, f'it should be 2 - currently it is {n_components}'
    fig_rows, fig_cols = out_shape
    validation_correct = parse_validation(validation)
    if validation_correct:
        fig_cols += 1
    plt.rcParams.update({'font.size': 7})
    fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, sharex=True, sharey=True, squeeze=False)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed')

    axes1d = axes.flatten()
    indices = agg_lists

    x_list = []
    t_list = []

    for i, ith_agg_list in enumerate(indices):
        x_list.append(sum((Xs[j] for j in ith_agg_list), []))
        title = f'{i} ({len(ith_agg_list)})'
        t_list.append(title)
    if drawing_subsamples:
        min_samples = np.inf
        for xi in x_list:
            if verbose > 0:
                print(f'checking minimum len(x). current: {len(xi)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(xi))

        if sup_title:
            sup_title = sup_title + f' remaining pts: {min_samples}'
    metrics = []
    min_x1, min_x2, max_x1, max_x2 = \
        reduce(lambda cum, cur: (min(cum[0], min(cur[0])),
                                 min(cum[1], min(cur[1])),
                                 max(cum[2], max(cur[0])),
                                 max(cum[3], max(cur[1]))),
               zip(*x_list), (np.inf, np.inf, -np.inf, -np.inf))
    if verbose > 0:
        print(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
    # vmax = None
    for ax, title, x in zip(axes1d, t_list, x_list):
        x1, x2 = zip(*x)
        if drawing_subsamples:
            together = list(zip(x1, x2))
            shuffle(together)
            x1, x2 = zip(*together)
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            if verbose > 0:
                print(f'remaining pts: {len(x1)}')
        ax.scatter(x1, x2, c='red', s=1, zorder=2, alpha=0.0015)
        bins = [30, 30]  # [41, 41]  # [7, 7]  # [41, 41]
        disk_radius = 2  # 6  # 6  # 3  # 6
        h, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins, density=False,
                                             range=[[min_x1, max_x1], [min_x2, max_x2]]) #, bins=(xedges, yedges))
        h = h.astype(np.uint8)
        hist_img = windowed_histogram(h, disk(disk_radius))
        hist_img_max = h.copy()

        for ix, iy in np.ndindex(hist_img_max.shape):
            hist_img_max[ix, iy] = np.sum([NORMALIZATION_FACTOR * i_elem * elem for i_elem, elem in enumerate(hist_img[ix][iy])])
        h = hist_img_max.T #h.T
        # if not vmax:
        #    vmax = np.max(h)
        m1, m2 = np.meshgrid(x_edges, y_edges)
        if verbose > 1:
            print(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0,  # vmax=vmax,
                      cmap='gray_r')  # cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    if verbose > 0:
        print(distance_matrix)
    if sup_title:
        # sup_title = sup_title + '\nDistances from the first picture: '+np.array2string(distance_matrix[0], precision=1)
        fig.suptitle(sup_title, fontsize=8)
    plt.savefig(result_path, dpi=1200, frameon=True, bbox_inches=None)
    # plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False)
    axes1d = axes.flatten()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    part2_x = [np.array(metrics)]
    part2_tsne = fit_tsne(part2_x)
    if verbose > 0:
        print(f'tsne: {part2_tsne}')
    p2_x1, p2_x2 = zip(*part2_tsne[0])
    colours = ['red', 'green', 'blue', 'black', 'magenta', 'yellow', 'purple']
    color = [i for s in [[c] * out_shape[1] for c in colours[:out_shape[0]]] for i in s]
    for ax in axes1d:
        ax.scatter(p2_x1, p2_x2, c=range(len(t_list)), cmap='gray_r') #, c=color)
        for (i, (x, y, title)) in enumerate(zip(p2_x1, p2_x2, t_list)):
            ax.text(x, y, title)
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', frameon=True, bbox_inches=None)
    animated_path_dynamics = f'{result_path[:-4]}_animated_dynamics.mp4'
    animate_part2_tsne = True
    fps = 10
    dpi = 300
    if animate_part2_tsne:
        writer = FFMpegWriter(fps=fps)
        fig1 = plt.figure()
        margin = 2
        min_x1 = min(p2_x1) - margin
        min_x2 = min(p2_x2) - margin
        max_x1 = max(p2_x1) + margin
        max_x2 = max(p2_x2) + margin
        l, = plt.plot([], [], 'k-o')
        plt.xlim(min_x1, max_x1)
        plt.ylim(min_x2, max_x2)
        with writer.saving(fig1, animated_path_dynamics, dpi):
            for i in tqdm(range(len(p2_x1))):
                l.set_data(p2_x1[max(0, i - 5):i], p2_x2[max(0, i - 5):i])
                writer.grab_frame()

    # plt.show()

def animate_language_change(tsne_X, animated_result_path, verbose=0,
                            points=20000, jump=1000, reverted=False, dpi=100,
                            fps=60, save_pngs=False, format='mp4', metadata=None,
                            scatter_alpha = 0.0, meshgrid_alpha = 1.0):
    '''
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    '''
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    assert save_pngs is False, 'currently we do not support saving pngs'
    fig = plt.figure()
    if reverted:
        tsne_X = tsne_X[::-1]

    def to_list_if_needed(maybe_list):
        if isinstance(maybe_list, list):
            return maybe_list
        return maybe_list.tolist()
    x_list = [to_list_if_needed(x) for s in tsne_X for x in s]
    #x_list = []
    #x_list.append(sum((x for x in tsne_X), []))

    print(x_list[:5])
    x1, x2 = zip(*x_list)
    min_x1 = min(x1)
    min_x2 = min(x2)
    max_x1 = max(x1)
    max_x2 = max(x2)

    bins = [30, 30]  # [41, 41]  # [7, 7]  # [41, 41]
    disk_radius = 2  # 6  # 6  # 3  # 6

    vmax = None
    with writer.saving(fig, animated_result_path, dpi):
        for start_idx in tqdm(range(0, len(x_list) - points, jump)):
            plt.clf()

            plt.xlim(min_x1, max_x1)
            plt.ylim(min_x2, max_x2)
            x = x_list[start_idx:start_idx + points]
            x1, x2 = zip(*x)
            if scatter_alpha > 0.0:
                plt.plot(x1, x2, 'k.', alpha=scatter_alpha)
            if meshgrid_alpha > 0.0:

                h, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins, density=False,
                                                     range=[[min_x1, max_x1], [min_x2, max_x2]])  # , bins=(xedges, yedges))
                h = h.astype(np.uint8)

                hist_img = windowed_histogram(h, disk(disk_radius))

                hist_img_max = h.copy()

                for ix, iy in np.ndindex(hist_img_max.shape):
                    hist_img_max[ix, iy] = np.sum(
                        [NORMALIZATION_FACTOR * i_elem * elem for i_elem, elem in enumerate(hist_img[ix][iy])])
                h = hist_img_max.T  # h.T
                m1, m2 = np.meshgrid(x_edges, y_edges)
                if not vmax:
                    vmax = np.max(h)
                plt.pcolormesh(m1, m2, h, zorder=1, cmap='gray_r', vmin=0.0, vmax=vmax,
                               alpha=meshgrid_alpha)  # cmap='RdBu')
            writer.grab_frame()
