from collections import OrderedDict
from functools import reduce
from itertools import product
import logging
from operator import attrgetter
from random import shuffle

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from scipy.spatial.distance import (pdist, squareform)
from skimage.filters.rank import windowed_histogram
from skimage.morphology import disk
from sklearn.decomposition import PCA
from tqdm import tqdm

from schemas import *

logger = logging.getLogger()
NORMALIZATION_FACTOR = 1.0  # 8.0  # 255.0

def fit_tsne(x, n_components=2, perplexity=30, n_iter=1000, n_iter_without_progress=300):
    assert (n_components is 2)
    tsne = TSNE(n_jobs=4, n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, n_iter_without_progress=n_iter_without_progress)
    tsne_out = tsne.fit_transform(x)
    return tsne_out


def fit_pca(list_X, n_components=2):
    pca = PCA(n_components=n_components)
    X = np.concatenate(list_X)
    pca.fit(X)
    logger.info(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
    cum = [pca.explained_variance_ratio_[0]]
    for i in range(1, len(pca.explained_variance_ratio_)):
        cum.append(cum[i - 1] + pca.explained_variance_ratio_[i])
    logger.info(f'PCA explained variance ratio cum: {cum}')
    logger.info(f'PCA singular values: {pca.singular_values_}')
    return pca


def parse_validation(validation=None):
    validated = False
    if validation:
        pass  # TODO
    return validated


def draw_composition(Xs, ys, result_path, out_shape=None, agg_shape=None,
                     drawing_subsamples=False, sup_title=None, validation=None):
    if not out_shape:
        out_shape = (1, len(ys))
    if not agg_shape:
        agg_shape = (1, 1)
    estimated_series_size = agg_shape[1] * out_shape[1]
    assert(agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1] is len(ys))
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
            logger.debug(f'Checking minimum len(y). current: {len(yi)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(yi))

    if sup_title:
        sup_title = sup_title + f' remaining pts: {min_samples}'
    colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
    metrics = []
    min_x1, min_x2, max_x1, max_x2 = \
        reduce(lambda cum, cur: (min(cum[0], min(cur[0])),
                                 min(cum[1], min(cur[1])),
                                 max(cum[2], max(cur[0])),
                                 max(cum[3], max(cur[1]))),
               zip(*x_list), (np.inf, np.inf, -np.inf, -np.inf))
    logger.debug(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
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
        logger.debug(f'remaining labels: {remaining_labels} (#pts in total: {len(y)})')
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
            logger.debug(f'remaining pts: {len(color)}')
        ax.scatter(x1, x2, c=color, s=1, zorder=2, alpha=0.0015)
        bins = [41, 41]  # [7, 7]  # [41, 41]
        disk_radius = 6  # 6  # 3  # 6
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
        logger.debug(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0, vmax=vmax, cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    logger.debug(f'Distance matrix: {distance_matrix}')
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
    logger.debug(f'tsne: {part2_tsne}')
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


def draw_X(Xs, result_path, out_shape=None, agg_shape=None, drawing_subsamples=False,
           sup_title=None, validation=None):
    if not out_shape:
        out_shape = (1, len(Xs))
    if not agg_shape:
        agg_shape = (1, 1)
    estimated_series_size = agg_shape[1] * out_shape[1]
    assert agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1] is len(Xs), f'{agg_shape[0] * agg_shape[1] * out_shape[0] * out_shape[1]} vs {len(Xs)}'
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
    x_list = []
    t_list = []

    for i in indices:
        x_list.append(sum((Xs[j] for j in i), []))
        title = f'S{1 + (i[0] // estimated_series_size)}_D{1 + (i[0] % estimated_series_size)}'
        t_list.append(title)
    if drawing_subsamples:
        min_samples = np.inf
        for xi in x_list:
            logger.debug(f'checking minimum len(x). current: {len(xi)}, minimum so far: {min_samples}')
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
    logger.debug(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
    vmax = None
    for ax, title, x in zip(axes1d, t_list, x_list):
        x1, x2 = zip(*x)
        if drawing_subsamples:
            together = list(zip(x1, x2))
            shuffle(together)
            x1, x2 = zip(*together)
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            logger.debug(f'remaining pts: {len(x1)}')
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
        logger.debug(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0, vmax=vmax, cmap='gray_r')  # cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    logger.debug(f'distance matrix: {distance_matrix}')
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
    logger.debug(f'tsne: {part2_tsne}')
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


def draw_X2(Xs, result_path, out_shape=None, agg_lists=None, drawing_subsamples=False,
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
    logger.debug(f'Indices: {indices}')
    x_list = []
    t_list = []

    for i, ith_agg_list in enumerate(indices):
        x_list.append(sum((Xs[j] for j in ith_agg_list), []))
        title = f'{i} ({len(ith_agg_list)})'
        t_list.append(title)
    if drawing_subsamples:
        min_samples = np.inf
        for xi in x_list:
            logger.debug(f'checking minimum len(x). current: {len(xi)}, minimum so far: {min_samples}')
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
    logger.debug(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')
    # vmax = None
    for ax, title, x in zip(axes1d, t_list, x_list):
        x1, x2 = zip(*x)
        if drawing_subsamples:
            together = list(zip(x1, x2))
            shuffle(together)
            x1, x2 = zip(*together)
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            logger.debug(f'remaining pts: {len(x1)}')
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
        logger.debug(f'shape: {m1.shape} vs {m2.shape} vs {h.shape}')
        ax.pcolormesh(m1, m2, h, zorder=1, vmin=0.0,  # vmax=vmax,
                      cmap='gray_r')  # cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
        ax.set(aspect='equal')
        ax.set_title(title, fontsize=8)
        metrics.append(h.reshape(-1))
    distance_matrix = squareform(pdist(np.array(metrics))) / NORMALIZATION_FACTOR
    logger.debug(f'distance matrix: {distance_matrix}')
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
    logger.debug(f'tsne: {part2_tsne}')
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


def preprocess_events(events, base_dir):
    """
    preprocess_events populates list of Converters and scales timestamps to [0,1]
    :param events:
    :return:
    """
    colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
    subjects = [event.subject for event in events]
    labels = [event.label for event in events]
    unique_subjects = sorted(list(set(subjects)))
    unique_labels = sorted(list(set(labels)))
    converters = dict()
    for subject in unique_subjects:
        color = colors[unique_subjects.index(subject) % len(colors)]
        color_labels = {label: colors[unique_labels.index(label) % len(colors)] for label in labels}
        timestamp_min = min([event.timestamp for event in events if event.subject == subject])
        timestamp_max = max([event.timestamp for event in events if event.subject == subject])
        time_scale = timestamp_max - timestamp_min
        converter = Converter(subject, timestamp_min,
                              timestamp_max, time_scale,
                              color, color_labels)
        converters[subject] = converter

    def update_timestamp(event):
        c = converters[event.subject]
        return (event.timestamp - c.timestamp_min) / c.time_scale
    scaled_events = [event._replace(timestamp=update_timestamp(event)) for event in events]
    sorted_events = sorted(scaled_events, key=attrgetter('timestamp'))
    return sorted_events, converters


def split_events_by_offset(events, split_params):
    split = []
    frame_id = []
    points = split_params.get('points', 20000)
    jump = split_params.get('jump', 1000)
    for start_idx in range(0, len(events), jump):
        length = min(points, len(events) - start_idx)
        split.append(events[start_idx:start_idx + length].copy())
        frame_id.append(start_idx/points)
    return split, frame_id


def split_events_by_lifespan(events, split_params):
    split = []
    frame_id = []
    lifespan = split_params.get('lifespan', 120)
    subjects = set((event.subject for event in events))
    for subject in subjects:
        subject_events = [event for event in events if event.subject == subject]
        def by_timestamp_raw(obj):
            return obj.timestamp_raw
        subject_events.sort(key=by_timestamp_raw)
        start_t = subject_events[0].timestamp_raw
        stop_t = subject_events[-1].timestamp_raw
        t = start_t
        while t <= stop_t:
            current_events = [event for event in subject_events if t <= event.timestamp_raw < t + lifespan]
            if len(current_events) > 0:
                split.append(current_events)
                frame_id.append((t - start_t) / lifespan)
            t = min([event.timestamp_raw for event in events if event.timestamp_raw >= t + lifespan] + [stop_t + 1.0])
    split_id = zip(split, frame_id)
    sorted_id = sorted(split_id, key=lambda x: x[1])
    (split_sorted, frame_id_sorted) = tuple(zip(*sorted_id))
    return split_sorted, frame_id_sorted


def split_events_by_recordings(events, split_params):
    """
    The implementation of the split strategy "recordings" where each blob corresponds to unique (subject,sample) tuple
    and has id corresponding to which sample of the given subject it is
    (assuming samples are ascending with the timestamp)
    :param events: events that need a split
    :param split_params: not used
    :return:
    """
    split = []
    frame_id = []
    samples = OrderedDict.fromkeys([(event.subject, event.sample) for event in events])
    subjects = set((event.subject for event in events))
    subject_counters = {subject: 0 for subject in subjects}
    for subject, sample in samples.keys():
        split.append([event for event in events if event.sample == sample and event.subject == subject])
        frame_id.append(subject_counters[subject])
        subject_counters[subject] += 1
    return split, frame_id


def split_events_by_fitting_to_shape(events, split_params):
    out_shape = split_params.get('out_shape', (1, 1))
    strategy_params = split_params.get('additional_params', None)

    if out_shape is (1, 1) or not strategy_params:
        split = [events]
        frame_id = [0]
        return split, frame_id
    subjects = list(set(event.subject for event in events))
    subjects.sort()
    # the above line is faster than OrderedDict.fromkeys(event.subject for event in events)

    assert(out_shape[0] == len(subjects))

    keys = range(out_shape[0] * out_shape[1])
    items = [(key, []) for key in keys]
    split_dict = OrderedDict(items)  # .fromkeys(range(out_shape[0] * out_shape[1]))  # [None] * out_shape[0] * out_shape[1]

    for i, subject in enumerate(subjects):
        subject_events = [event for event in events if event.subject == subject]
        logger.info(f' strategy params: {strategy_params} , len(sub_ev): {len(subject_events)}')
        current_splits, current_frame_ids = split_events(subject_events, strategy_params)
        logger.info(f' subject: {subject} -> {len(current_frame_ids)}')

        for current_split, current_frame_id in zip(current_splits, current_frame_ids):
            max_frame_id = current_frame_ids[-1]
            index = i * out_shape[1] + ( (current_frame_id * out_shape[1]) // (max_frame_id + 1) )
            if index in split_dict:
                split_dict[index].extend(current_split)
            else:
                raise IndexError('invalid index')

    split = list(split_dict.values())
    frame_id = list(split_dict.keys())
    return split, frame_id  # tuple(zip(*sorted_id))  # (split_sorted, frame_id_sorted)





def split_events(events, split_params):
    strategy = split_params.get('strategy', 'offset')
    split_func = {'offset': split_events_by_offset,
                  'lifespan': split_events_by_lifespan,
                  'recordings': split_events_by_recordings,
                  'fit_to_shape': split_events_by_fitting_to_shape}
    return split_func[strategy](events, split_params)


def compute_histogram(events, bins, selem):
    xs = np.array([event.x for event in events])
    x1, x2 = zip(*xs)
    h, _, _ = np.histogram2d(x1, x2, bins=bins, density=False)  # , bins=(xedges, yedges))
    hist_img = windowed_histogram(h.astype(np.uint8), selem)
    hist_img_max = h.copy()
    for ix, iy in np.ndindex(hist_img_max.shape):
        hist_img_max[ix, iy] = np.sum(
            [NORMALIZATION_FACTOR * i_elem * elem for i_elem, elem in enumerate(hist_img[ix][iy])])
    h = hist_img_max.T  # h.T
    return h


def animate_language_change(events, converters, animated_result_path, split_params, bins=(30, 30),
                            disk_radius=2, reverted=False, dpi=100, fps=60, save_pngs=False, metadata=None,
                            scatter_alpha=0.0, meshgrid_alpha=1.0, separate_subject=None):
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    assert save_pngs is False, 'currently we do not support saving pngs'
    fig = plt.figure()
    if reverted:
        events[:] = events[::-1]
    event_split, frame_ids = split_events(events, split_params)
    xs = np.array([event.x for event in events])
    min_x1, min_x2 = np.amin(xs, axis=0)
    max_x1, max_x2 = np.amax(xs, axis=0)
    x_edges = np.linspace(min_x1, max_x1, bins[0], endpoint=True)
    y_edges = np.linspace(min_x2, max_x2, bins[1], endpoint=True)
    m1, m2 = np.meshgrid(x_edges, y_edges)
    results = []
    vmax = 0
    for event_packet, frame_ids in zip(event_split, frame_ids):
        histogram = compute_histogram(event_packet, (x_edges, y_edges), disk(disk_radius))
        results.append((histogram, event_packet, frame_ids))
        vmax = max(vmax, np.max(histogram))

    with writer.saving(fig, animated_result_path, dpi):
        for (h, event_packet, frame_ids) in tqdm(results):
            plt.clf()
            plt.xlim(min_x1, max_x1)
            plt.ylim(min_x2, max_x2)
            event_data = [(event.x[0], event.x[1], converters[event.subject].color_labels[event.label],
                           event.subject, event.label) for event in event_packet]
            x1, x2, color, subject, labels = zip(*event_data)
            if scatter_alpha > 0.0:
                if separate_subject:
                    plt.scatter(x1, x2, c=color, alpha=scatter_alpha)  # , label = separate_subject)
                else:
                    for s in converters.keys():
                        plt.scatter(x1, x2, c=color, alpha=scatter_alpha, label=s)
            # for label, x, y in zip(labels, x1, x2):
            #     plt.annotate(label, xy=(x, y), bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.35))
            if meshgrid_alpha > 0.0:
                plt.pcolormesh(m1, m2, h, zorder=1, cmap='gray_r', vmin=0.0, vmax=vmax,
                               alpha=meshgrid_alpha)  # cmap='RdBu')
            # plt.legend(list(converters.keys()), loc='upper right')
            # legend = plt.gca().get_legend()
            # [handle.set_color(converters[handle.get_label()].color) for handle in legend.legendHandles]
            title = f'frame: {frame_ids:.2f}'
            if separate_subject:
                title = 'subject: ' + separate_subject + ' - ' + title
            plt.gca().set_title(title)

            writer.grab_frame()


def draw_metrics(result_path, events, converters, split_params=None,
                bins=(30, 30), disk_radius=2,
                out_shape=None, drawing_subsamples=False,
                sup_title=None, validation=None):

    event_split, frame_ids = split_events(events, split_params)
    xs = np.array([event.x for event in events])
    min_x1, min_x2 = np.amin(xs, axis=0)
    max_x1, max_x2 = np.amax(xs, axis=0)
    logger.debug(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')

    x_edges = np.linspace(min_x1, max_x1, bins[0], endpoint=True)
    y_edges = np.linspace(min_x2, max_x2, bins[1], endpoint=True)
    min_samples = np.inf
    if drawing_subsamples:
        for event_packet in event_split:
            logger.debug(f'Checking minimum len(x). current: {len(event_packet)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(event_packet))

    titles = []
    metrics = []
    subjects_ids = []
    subjects = list(set([event.subject for event in events]))

    for event_packet, frame_idx in zip(event_split, frame_ids):

        if drawing_subsamples:
            shuffle(event_packet)
            event_packet = event_packet[:min_samples]
        histogram = compute_histogram(event_packet, (x_edges, y_edges), disk(disk_radius))
        hs_vector = histogram.reshape(-1)
        metrics.append(hs_vector)
        titles.append(f'{event_packet[0].subject}_{frame_idx}')  # -{event_packet[0].sample}')
        subjects_ids.append(1 + subjects.index(event_packet[0].subject) + 0.1/(1+frame_idx))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False)
    axes1d = axes.flatten()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    part2_x = np.array(metrics)
    perplexity = 5
    #if part2_x.shape[0] < 100:
    #    perplexity = 5
    part2_tsne = fit_tsne(part2_x, perplexity=perplexity)
    logger.debug(f'tsne: {part2_tsne}')
    p2_x1, p2_x2 = zip(*part2_tsne)
    colours = ['red', 'green', 'blue', 'black', 'magenta', 'yellow', 'purple']
    color = [i for s in [[c] * out_shape[1] for c in colours[:out_shape[0]]] for i in s]
    for ax in tqdm(axes1d):
        # ax.scatter(p2_x1, p2_x2, c=range(len(titles)), cmap='gray_r')  # , c=color)
        ax.scatter(p2_x1, p2_x2, c=subjects_ids)  # , c=color)
        #for (i, (x, y, title)) in enumerate(zip(p2_x1, p2_x2, titles)):
        #    ax.text(x, y, title)
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', bbox_inches=None)

def draw_events(result_path, events, converters, split_params=None,
                bins=(30, 30), disk_radius=2,
                out_shape=None, drawing_subsamples=False,
                sup_title=None, validation=None):

    split_params = {'strategy': 'fit_to_shape',
                    'additional_params': split_params,
                    'out_shape': out_shape}
    event_split, frame_ids = split_events(events, split_params)
    xs = np.array([event.x for event in events])
    min_x1, min_x2 = np.amin(xs, axis=0)
    max_x1, max_x2 = np.amax(xs, axis=0)
    logger.debug(f'limits: {(min_x1, min_x2, max_x1, max_x2)}')

    x_edges = np.linspace(min_x1, max_x1, bins[0], endpoint=True)
    y_edges = np.linspace(min_x2, max_x2, bins[1], endpoint=True)
    m1, m2 = np.meshgrid(x_edges, y_edges)
    vmax = 0
    if drawing_subsamples:
        min_samples = np.inf
        for event_packet in event_split:
            logger.debug(f'checking minimum len(x). current: {len(event_packet)}, minimum so far: {min_samples}')
            min_samples = min(min_samples, len(event_packet))

        if sup_title:
            sup_title = sup_title + f' remaining pts: {min_samples}'

    result_frames = {}
    titles = []
    for event_packet, frame_idx in zip(event_split, frame_ids):
        if drawing_subsamples:
            shuffle(event_packet)
            event_packet = event_packet[:min_samples]
        histogram = compute_histogram(event_packet, (x_edges, y_edges), disk(disk_radius))
        hs_vector = histogram.reshape(-1)
        assert frame_idx not in result_frames
        assert 0 <= frame_idx < out_shape[0] * out_shape[1]
        result_frames[frame_idx] = (histogram, hs_vector, event_packet)
        titles.append(f'{event_packet[0].sample}')  # -{event_packet[0].sample}')

        vmax = max(vmax, np.max(histogram))

    fig_rows, fig_cols = out_shape
    validation_correct = parse_validation(validation) # validation is used to put validation information into the last column
    if validation_correct:
        fig_cols += 1
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=fig_rows, ncols=fig_cols, sharex=True, sharey=True, squeeze=False)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed')

    axes1d = axes.flatten()
    metrics = []
    for i, ax in tqdm(enumerate(axes1d)):
        ax.set(aspect='equal')

        if i not in result_frames:
            continue
        (histogram, hs_vector, event_packet) = result_frames[i]
        metrics.append(hs_vector)
        # titles.append(f'{set([event.subject for event in event_packet])}')  # -{event_packet[0].sample}')
        x1 = [event.x[0] for event in event_packet]
        x2 = [event.x[1] for event in event_packet]
        # TODO: collect all parameters and enable their custom configuration
        ax.scatter(x1, x2, c='red', s=1, zorder=2, alpha=0.0015)
        # ax.set_title(title, fontsize=8)
        ax.pcolormesh(m1, m2, histogram, zorder=1, vmin=0.0,  # vmax=vmax,
                      cmap='gray_r')  # cmap='RdBu') #, shading='gouraud') #150) #0.0001) # cmap='RdBu',
    plt.savefig(result_path, dpi=450, bbox_inches=None)

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False)
    axes1d = axes.flatten()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    part2_x = np.array(metrics)
    part2_tsne = fit_tsne(part2_x, perplexity=1)

    logger.debug(f'tsne: {part2_tsne}')
    p2_x1, p2_x2 = zip(*part2_tsne)
    colours = ['red', 'green', 'blue', 'black', 'magenta', 'yellow', 'purple']
    color = [i for s in [[c] * out_shape[1] for c in colours[:out_shape[0]]] for i in s]
    for ax in tqdm(axes1d):
        ax.scatter(p2_x1, p2_x2, c=range(len(titles)), cmap='gray_r')  # , c=color)
        for (i, (x, y, title)) in enumerate(zip(p2_x1, p2_x2, titles)):
            ax.text(x, y, title)
    result_path_dynamics = f'{result_path[:-4]}_dynamics.png'
    plt.savefig(result_path_dynamics, dpi='figure', bbox_inches=None)
    # animated_path_dynamics = f'{result_path[:-4]}_animated_dynamics.mp4'
