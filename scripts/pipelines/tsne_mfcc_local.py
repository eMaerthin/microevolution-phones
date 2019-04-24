from itertools import compress
import json
from os.path import join
from pickle import (dumps, loads)

import pandas as pd

from decorators import (check_if_already_done, timeit)
from dimensionality_reduction import (draw_X2, animate_language_change, fit_tsne, fit_pca)
from marshmallow import pprint
from schemas import *
from pipeline import Pipeline
from pipelines.mfcc_pipeline import MfccPipeline

class TsnePipelineMfcc(Pipeline):
    def __init__(self, verbose, subjects_dir):
        super(TsnePipelineMfcc, self).__init__(verbose=verbose, subjects_dir=subjects_dir)
        self._x = [] # TODO(mbodych): let's improve code readability by changing name of this variable
        self._y = [] # TODO(mbodych): let's improve code readability by changing name of this variable

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_tsne_mfcc_result.json'

    @staticmethod
    def filename_prerequisites():
        return [MfccPipeline.result_filename]

    @staticmethod
    def filename_prerequisites_postprocessed():
        return [MfccPipeline.result_filename_postprocessed]

    def series_pipeline(self, series_json_path, series_settings):
        schema = MfccSchema()
        mfcc_result_path = self.filename_prerequisites()[0](series_json_path)
        mfcc_csv_result_path = f'{mfcc_result_path[:-5]}.csv'

        @timeit
        @check_if_already_done(mfcc_csv_result_path, self.verbose)  # , ignore_already_done=True)
        def store_mfcc_result_as_csv(mfcc_result_path, mfcc_csv_result_path):
            print(f'result_path: {mfcc_csv_result_path}')
            with open(mfcc_result_path, 'r') as f_csv:
                print(f' mfcc_result_path: {mfcc_result_path}')
                json_file = json.load(f_csv)
                result = schema.load(json_file)
                if self.verbose > 1:
                    pprint(result, indent=4)
                df = pd.DataFrame.from_dict(result['mfcc_info'])
                df = df.mfcc.apply(pd.Series).add_suffix('_mfcc_features').merge(df, left_index=True,
                                                                                 right_index=True).drop(['mfcc'],
                                                                                                        axis=1)
                # filter nans:
                for c in df.filter(regex='features$', axis=1).columns:
                    df = df[df[c].notnull()]
                df.to_csv(mfcc_csv_result_path)

        store_mfcc_result_as_csv(mfcc_result_path, mfcc_csv_result_path)

        df = pd.read_csv(mfcc_csv_result_path)
        df = df[df.i == 0]
        self._y.append(df.word.values)

        df = df[[col for col in df.columns if col.endswith('_mfcc_features')]]
        self._x.append(df.values)

    def subject_pipeline(self, working_dir, series, common_settings):
        series = self.sort_series(series)  # , '{}.json')
        for series_json_filename in series:
            settings = self.load_settings(working_dir, series_json_filename, self.verbose)
            series_json_path = join(working_dir, series_json_filename)
            series_settings = self.merge_settings(common_settings, settings)
            self.series_pipeline(series_json_path, series_settings)

    def global_pipeline(self, series_to_process):
        self._x = []
        self._y = []
        global_tsne_input_XY_path = join(self.subjects_dir, 'global_tsne_input_XY.pickle')

        @timeit
        @check_if_already_done(global_tsne_input_XY_path, self.verbose)
        def prepare_global_tsne_input(global_tsne_input_XY_path, series_to_process):
            for subject, series in series_to_process:
                working_dir = join(self.subjects_dir, subject)
                common_settings = self.load_settings(working_dir, 'common.json', self.verbose)
                self.subject_pipeline(working_dir, series, common_settings)
            with open(global_tsne_input_XY_path, 'wb') as result_f:
                pickled = dumps([self._x, self._y])
                result_f.write(pickled)

        prepare_global_tsne_input(global_tsne_input_XY_path, series_to_process)
        with open(global_tsne_input_XY_path, 'rb') as f:
            [self._x, self._y] = loads(f.read())


        working_dir = self.subjects_dir
        filter = []
        only_vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
        only_EH_and_NG = ['EH', 'NG']
        only_AH_EH = ['AH', 'EH']
        only_SZ = ['S', 'Z']
        only_plus = ['R','L']
        filter = only_AH_EH + only_SZ  # []  # only_AH_EH  # []  # ['B','P']  # ['AH']  # 'AH'] #'AH','IY'] #['AH', 'IY'] # ['AH'] # only_vowels # [] # ['B','P']

        if filter and len(filter):
            for i in range(len(self._y)):
                subset = [any(substring in phoneme for substring in filter) for phoneme in self._y[i]]
                self._x[i] = self._x[i][subset, :]
                self._y[i] = list(compress(self._y[i], subset))

        filter_hash = '_'.join(sorted(filter))
        n_iter = 500
        perplexity = 40
        n_components = 10
        tsne_mfcc_result_path = join(working_dir, f'tsne_mfcc_result_{filter_hash}_{n_iter}_{perplexity}.pickle')
        pca_mfcc_result_path = join(working_dir, f'pca_mfcc_result_{filter_hash}.pickle')
        pca_tsne_mfcc_result_path = join(working_dir, f'pca_tsne_mfcc_result_{filter_hash}.pickle')

        @timeit
        @check_if_already_done(tsne_mfcc_result_path, self.verbose)  # , ignore_already_done=True)
        def cache_fit_tsne(tsne_mfcc_result_path, X, n_iter, perplexity, **kwargs):
            tsne_X = fit_tsne(X, n_iter=n_iter, perplexity=perplexity)
            # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            pickle_dumped = dumps(tsne_X)
            with open(tsne_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)

        @timeit
        @check_if_already_done(pca_mfcc_result_path, self.verbose)  # , ignore_already_done=True)
        def cache_pca(pca_mfcc_result_path, X, n_components, **kwargs):
            pca = fit_pca(X, verbose=self.verbose, n_components=n_components)
            x_pca = []
            for x in X:
                x_pca.append(pca.transform(x).tolist())
            pickle_dumped = dumps(x_pca)
            with open(pca_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)

        @timeit
        @check_if_already_done(pca_tsne_mfcc_result_path, self.verbose)
        def cache_pca_tsne(pca_tsne_mfcc_result_path, X, n_iter, perplexity, n_components, **kwargs):
            pca = fit_pca(X, verbose=self.verbose, n_components=n_components)
            x_pca = []
            for x in X:
                x_pca.append(pca.transform(x).tolist())
            tsne_X = fit_tsne(x_pca, n_iter=n_iter, perplexity=perplexity)
            # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            pickle_dumped = dumps(tsne_X)
            with open(pca_tsne_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)

        validation = None
        method = 'pca_tsne'
        cache_fun = None
        result_path = None
        if method == 'tsne':
            cache_fun = cache_fit_tsne
            result_path = tsne_mfcc_result_path
        elif method == 'pca':
            cache_fun = cache_pca
            result_path = pca_mfcc_result_path
        elif method == 'pca_tsne':
            cache_fun = cache_pca_tsne
            result_path = pca_tsne_mfcc_result_path
        assert cache_fun
        assert result_path
        cache_fun(result_path, self._x, n_iter=n_iter, perplexity=perplexity, n_components=n_components)
        with open(result_path, 'rb') as f:
            x_to_be_drawn = loads(f.read())
            '''
            phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                            'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
                            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
            for phoneme in phoneme_list:
                filter_local = [phoneme]
                '''
            filter_local = ['AH']  # only_AH_EH  # only_SZ  # only_AH_EH  # []
            y= self._y.copy()
            y_list = y.copy()
            if filter_local:
                for i, z in enumerate(x_to_be_drawn):
                    x_to_be_drawn[i] = [x for e, x in enumerate(z) if y[i][e] in filter_local]
                    y_list[i] = [a for e, a in enumerate(y[i]) if a in filter_local]
            else:
                for i, yi in enumerate(y):
                    if not isinstance(yi, list):
                        y_list[i] = yi.tolist()
            out_shape = (5, 5)
            '''
            ranges = [
                (49, 51), (52, 54), (55, 57),
                (58, 60), (61, 63), (64, 65),
                (66, 71), (72, 76), (77, 81),
                (11, 14), (15, 18), (19, 21),
                (0, 3), (4, 7), (8, 10),
                (35, 39), (40, 44), (45, 48),
                (22, 26), (27, 30), (31, 34)
            ]
            agg_lists = [list(range(a, b + 1)) for (a, b) in ranges]
            '''
            shape_hash = str(out_shape[0]) + '-' + str(out_shape[1])
            filter_local_hash = '_'.join(sorted(filter_local))
            draw_result_path = join(working_dir,
                                    f'draw_{method}_mfcc_result_{filter_hash}_{shape_hash}_{filter_local_hash}.png')
            # f'draw_{method}_mfcc_global_result_{shape_hash}.png')
            draw_X2(x_to_be_drawn, draw_result_path, verbose=self.verbose,
                    out_shape=out_shape,
                    # drawing_subsamples=True,
                    # agg_lists=agg_lists,
                    sup_title=f'{method}_mfcc trained on: {filter} -> visualization filter: {filter_local}',
                    validation=validation)
            points_in_single_pointcloud = 1000
            offset_points_between_frames = 500
            reverted = True
            fps = 20
            out_video_format = 'mp4'
            dpi = 100
            for (meshgrid_alpha, scatter_alpha) in [(1.0, 0.0), (0.0, 1.0)]:
                animated_result_path = join(working_dir, f'animated_{method}_mfcc_result'
                f'_{filter_hash}_{filter_local_hash}'
                f'_{points_in_single_pointcloud}'
                f'_{offset_points_between_frames}'
                f'_reverted{reverted}'
                f'_mesh{meshgrid_alpha}'
                f'_scatter{scatter_alpha}'
                f'_fps{fps}_dpi{dpi}.{out_video_format}')
                animate_language_change(x_to_be_drawn, animated_result_path, verbose=self.verbose,
                                        points=points_in_single_pointcloud,
                                        jump=offset_points_between_frames,
                                        reverted=reverted, fps=fps, save_pngs=False, format=out_video_format,
                                        scatter_alpha=scatter_alpha, meshgrid_alpha=meshgrid_alpha, dpi=dpi)
