from collections import OrderedDict
import json
import logging
from os.path import join
from pickle import (dumps, loads)

import numpy as np
import pandas as pd

from chain import Chain
from chains.mfcc_local import MfccLocal
from decorators import (check_if_already_done, timeit)
from dimensionality_reduction import (draw_events, draw_metrics, animate_language_change, fit_tsne, fit_pca, preprocess_events)
from schemas import *
logger = logging.getLogger()


class TsneMfccLocal(Chain):
    allow_sample_layer_concurrency = False
    requirements = [MfccLocal]

    def __init__(self):
        super(TsneMfccLocal, self).__init__()
        self._events = []

    @staticmethod
    def sample_result_filename(out_sample_path):
        return MfccLocal.filenames_to_skip_sample(out_sample_path)[0]

    @staticmethod
    def sample_filename_prerequisites():
        return [MfccLocal.sample_result_filename]

    @staticmethod
    def filenames_to_skip_dataset(out_results_dir):
        return [join(out_results_dir, 'global_tsne_input_XY.pickle')]

    def sample_layer(self, subject, sample_json_filename, sample_settings):
        pprint_indent = self.process_settings.get('pprint_indent', 4)

        output_path_pattern = join(self.results_dir, subject, sample_json_filename)

        mfcc_result_path = self.sample_filename_prerequisites()[0](output_path_pattern)
        mfcc_csv_result_path = self.sample_result_filename(output_path_pattern)

        @timeit
        @check_if_already_done(mfcc_csv_result_path, lambda v: v)  # , ignore_already_done=True)
        def store_mfcc_result_as_csv(mfcc_result_path, mfcc_csv_result_path):
            schema = MfccLocalSchema()
            with open(mfcc_result_path, 'r') as f_csv:
                json_file = json.load(f_csv)
                result = schema.load(json_file)
                logger.debug(json.dumps(result, indent=pprint_indent))
                df = pd.DataFrame.from_dict(result['mfcc_info'])
                series_to_merge = df.mfcc.apply(pd.Series).add_suffix('_mfcc_features')
                df = series_to_merge.merge(df, left_index=True,
                                           right_index=True).drop(['mfcc'], axis=1)
                # filter nans:
                for c in df.filter(regex='features$', axis=1).columns:
                    df[:] = df[df[c].notnull()]
                df.to_csv(mfcc_csv_result_path)
                return True

        store_mfcc_result_as_csv(mfcc_result_path, mfcc_csv_result_path)

        df = pd.read_csv(mfcc_csv_result_path)
        # df = df[df.i == 0]
        labels = df.word.values
        metadata = sample_settings.get('metadata')
        timestamps = [metadata.get('published_timestamp', 0.0) + delta for delta in df.start.values]
        df = df[[col for col in df.columns if col.endswith('_mfcc_features')]]
        xs = df.values

        self._events.extend([Event(x, label, timestamp, timestamp,
                                   subject, sample_json_filename)
                             for (x, label,
                                  timestamp) in zip(xs, labels,
                                                    timestamps)])

    def dataset_preprocess(self, dataset):
        self._events = []

    def dataset_postprocess(self, dataset):
        global_tsne_input_path = self.filenames_to_skip_dataset(self.results_dir)[0]

        n_iter = self.process_settings.get('tsne_iterations', 500)
        perplexity = self.process_settings.get('tsne_perplexity', 40)
        pca_components = self.process_settings.get('pca_components', 10)
        method = self.process_settings.get('visualization_method', 'pca_tsne')
        filter = list(self.process_settings.get('tsne_phoneme_filter', self.process_settings.get('visualization_phoneme_filter', '')))
        filter_local = list(self.process_settings.get('draw_phoneme_filter', self.process_settings.get('visualization_phoneme_filter_local', '')))
        out_shape = self.process_settings.get('visualization_out_shape', (1, 1))
        drawing_subsamples = self.process_settings.get('visualization_drawing_subsamples', False)
        agg_lists = self.process_settings.get('visualization_agg_lists', None)
        points_in_single_pointcloud = self.process_settings.get('points_in_single_pointcloud', 500)
        offset_points_between_frames = self.process_settings.get('offset_between_frames', 500)
        reverted = self.process_settings.get('animation_reverted', False)
        fps = self.process_settings.get('animation_fps', 20)
        out_video_format = self.process_settings.get('animation_format', 'mp4')
        dpi = self.process_settings.get('animation_dpi', 100)
        separate_subjects = self.process_settings.get('animation_separate_subjects', False)
        draw_how_many_columns = self.process_settings.get('draw_how_many_columns', 1)
        split_lifespan = self.process_settings.get('split_lifespan', 120)
        split_strategy = self.process_settings.get('split_strategy', 'offset') # it may be "frame"
        want_to_draw_events = True

        @timeit
        @check_if_already_done(global_tsne_input_path)
        def prepare_global_tsne_input(out_path):
            with open(out_path, 'wb') as result_f:
                pickled = dumps(self._events)
                result_f.write(pickled)

        prepare_global_tsne_input(global_tsne_input_path)
        with open(global_tsne_input_path, 'rb') as f:
            self._events = loads(f.read())

        working_dir = self.results_dir

        if filter and len(filter) > 0:
            self._events = [event for event in self._events if event.label in filter]

        filter_hash = '_'.join(sorted(filter))

        def cache_fit_tsne(tsne_mfcc_result_path, events):
            x = np.array([event.x for event in events])
            x_tsne = fit_tsne(x, n_iter=n_iter, perplexity=perplexity)
            # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            tsne_events = [event._replace(x=tsne) for tsne, event in zip(x_tsne, self._events)]
            pickle_dumped = dumps(tsne_events)
            with open(tsne_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)

        def cache_pca(pca_mfcc_result_path, events):
            x = np.array([event.x for event in events])
            pca = fit_pca(x, n_components=2)
            x_pca = pca.transform(x)
            pca_events = [event._replace(x=x_new) for x_new, event in zip(x_pca, self._events)]
            pickle_dumped = dumps(pca_events)
            with open(pca_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)

        def cache_pca_tsne(pca_tsne_mfcc_result_path, events):
            x = np.array([event.x for event in events])
            pca = fit_pca([x], n_components=pca_components)
            x_pca = pca.transform(x)
            cache_fit_tsne(pca_tsne_mfcc_result_path, x_pca)
        pca_tsne_pickle_filename = f'pca_tsne_mfcc_result_{filter_hash}_{n_iter}_{perplexity}_{pca_components}.pickle'
        result_path_dict = {'tsne': join(working_dir, f'tsne_mfcc_result_{filter_hash}_{n_iter}_{perplexity}.pickle'),
                            'pca': join(working_dir, f'pca_mfcc_result_{filter_hash}_{pca_components}.pickle'),
                            'pca_tsne': join(working_dir, pca_tsne_pickle_filename)}

        cache_fun_dict = {'tsne': cache_fit_tsne,
                     'pca': cache_pca,
                     'pca_tsne': cache_pca_tsne}

        validation = None
        logger.info(f'method: {method}')
        result_path = result_path_dict[method]
        cache_fun = timeit(check_if_already_done(result_path)
                           (cache_fun_dict[method]))
        cache_fun(result_path, self._events)

        with open(result_path, 'rb') as f:
            # converter of timestamps should be applied in a better place perhaps
            vis_events, converters = preprocess_events(loads(f.read()), self.base_dir)

            # TODO(marcin) clean up is needed
            #  '''
            # phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
            #                 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            #                 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
            # for phoneme in phoneme_list:
            #     filter_local = [phoneme]
            #     '''
            if filter_local and len(filter_local):
                vis_events[:] = [event for event in vis_events if event.label in filter_local]

            filter_local_hash = '_'.join(sorted(filter_local))
            split_params = {'strategy': split_strategy, 'lifespan': split_lifespan,
                            'points': points_in_single_pointcloud, 'jump': offset_points_between_frames}

            if want_to_draw_events:
                shape_hash = str(out_shape[0]) + '-' + str(out_shape[1])

                draw_result_path = join(working_dir,
                                        f'draw_{method}_mfcc_result_{filter_hash}_{shape_hash}_{filter_local_hash}.png')
                sup_title = f'{method}_mfcc trained on: {filter} -> visualization filter: {filter_local}'
                number_of_subjects = len(set([event.subject for event in vis_events]))
                out_shape = (number_of_subjects, draw_how_many_columns)

                metrics_result_path = draw_result_path[:-4] + '_metrics.png'
                draw_metrics(metrics_result_path, events=vis_events, converters=converters, split_params=split_params,
                             out_shape=out_shape, drawing_subsamples=drawing_subsamples,
                             sup_title=sup_title, validation=validation)

                draw_events(draw_result_path, events=vis_events, converters=converters, split_params=split_params,
                            out_shape=out_shape, drawing_subsamples=drawing_subsamples,
                            sup_title=sup_title, validation=validation)


            for (meshgrid_alpha, scatter_alpha) in [(0.0, 1.0), (1.0, 0.0)]:
                prefix = join(working_dir, f'animated_{method}_mfcc_result_{filter_hash}_{filter_local_hash}')
                suffix = f'.{out_video_format}'
                params = OrderedDict({'reverted': reverted, 'fps': fps, 'save_pngs': False,
                                      'scatter_alpha': scatter_alpha, 'meshgrid_alpha': meshgrid_alpha,
                                      'dpi': dpi
                                      })
                if separate_subjects:
                    for subject in converters.keys():
                        params['separate_subject'] = subject
                        path_from_params = '_'.join([f'{key}_{value}'
                                                     for key, value in params.items()]
                                                    ).replace('/', '_')
                        animated_result_path = prefix + path_from_params + suffix
                        subject_events = [event for event in vis_events if event.subject == subject]
                        animate_language_change(events=subject_events, converters=converters,
                                                animated_result_path=animated_result_path,
                                                split_params=split_params, **params)
                else:
                    path_from_params = '_'.join([f'{key}_{value}'
                                                 for key, value in params.items()]).replace('/', '_')
                    animated_result_path = prefix + path_from_params + suffix
                    animate_language_change(events=vis_events, converters=converters,
                                            animated_result_path=animated_result_path,
                                            split_params=split_params, **params)

""" old stuff:
            # '''
            # ranges = [
            #     (49, 51), (52, 54), (55, 57),
            #     (58, 60), (61, 63), (64, 65),
            #     (66, 71), (72, 76), (77, 81),
            #     (11, 14), (15, 18), (19, 21),
            #     (0, 3), (4, 7), (8, 10),
            #     (35, 39), (40, 44), (45, 48),
            #     (22, 26), (27, 30), (31, 34)
            # ]
            # agg_lists = [list(range(a, b + 1)) for (a, b) in ranges]
            # '''
"""