import json
import logging
from os.path import join
from pickle import (dumps, loads)

import pandas as pd

from decorators import (check_if_already_done, timeit)
from dimensionality_reduction import (animate_language_change, draw_X2, fit_tsne, fit_pca)
from schemas import *
from chain import Pipeline
from chains.mfcc_global_pipeline import MfccGlobalPipeline
logger = logging.getLogger()


class TsnePipelineMfccGlobal(Pipeline):
    def __init__(self):
        super(TsnePipelineMfccGlobal, self).__init__()
        self._x = []  # TODO(mbodych): let's improve code readability by changing name of this variable

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_tsne_mfcc_global_result.json'

    @staticmethod
    def filename_prerequisites():
        return [MfccGlobalPipeline.result_filename]

    @staticmethod
    def filename_prerequisites_postprocessed():
        return [MfccGlobalPipeline.result_filename_postprocessed]

    def series_pipeline(self, series_json_path, series_settings):
        schema = MfccGlobalSchema()
        mfcc_global_result_path = self.filename_prerequisites()[0](series_json_path)
        mfcc_global_csv_result_path = f'{mfcc_global_result_path[:-5]}.csv'

        @timeit
        @check_if_already_done(mfcc_global_csv_result_path)  # , ignore_already_done=True)
        def store_mfcc_global_result_as_csv(mfcc_global_result_path, mfcc_global_csv_result_path):
            logger.info(f'result_path: {mfcc_global_csv_result_path}')
            with open(mfcc_global_result_path, 'r') as f_csv:
                logger.info(f'mfcc_global_result_path: {mfcc_global_result_path}')
                json_file = json.load(f_csv)
                result = schema.load(json_file)
                logger.debug(json.dumps(result, indent=4))
                df = pd.DataFrame.from_dict(result['mfcc_global_info'])
                df = df.mfcc.apply(pd.Series).add_suffix('_mfcc_features').merge(df, left_index=True,
                                                                                 right_index=True).drop(['mfcc'],
                                                                                                        axis=1)
                # filter nans:
                for c in df.filter(regex='features$', axis=1).columns:
                    df = df[df[c].notnull()]
                df.to_csv(mfcc_global_csv_result_path)

        store_mfcc_global_result_as_csv(mfcc_global_result_path, mfcc_global_csv_result_path)

        df = pd.read_csv(mfcc_global_csv_result_path)
        ## df = df[df.i == 0]

        df = df[[col for col in df.columns if col.endswith('_mfcc_features')]]
        self._x.append(df.values)

    def global_pipeline(self, series_to_process):
        self._x = []  # reset on every call to global_pipeline
        global_tsne_input_path = join(self.subjects_dir, 'global_tsne_input.pickle')

        @timeit
        @check_if_already_done(global_tsne_input_path)
        def prepare_global_tsne_input(global_tsne_input_path, series_to_process):
            for subject, series in series_to_process:
                working_dir = join(self.subjects_dir, subject)
                common_settings = self.load_settings(working_dir, 'common.json')
                self.subject_pipeline(working_dir, series, common_settings)
            with open(global_tsne_input_path, 'wb') as result_f:
                pickled = dumps(self._x)
                result_f.write(pickled)

        prepare_global_tsne_input(global_tsne_input_path, series_to_process)
        with open(global_tsne_input_path, 'rb') as f:
            self._x = loads(f.read())

        working_dir = self.subjects_dir
        tsne_mfcc_global_result_path = join(working_dir, 'tsne_mfcc_global_result.pickle')
        pca_mfcc_global_result_path = join(working_dir, 'pca_mfcc_global_result.pickle')

        @timeit
        @check_if_already_done(tsne_mfcc_global_result_path)  # , ignore_already_done=True)
        def cache_fit_tsne(tsne_mfcc_global_result_path, X):  # , y):
            tsne_X = fit_tsne(X, n_iter=250)
            # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            pickle_dumped = dumps(tsne_X)
            with open(tsne_mfcc_global_result_path, 'wb') as f:
                f.write(pickle_dumped)

        @timeit
        @check_if_already_done(pca_mfcc_global_result_path, ignore_already_done=True)
        def cache_pca(pca_mfcc_result_path, X):
            pca = fit_pca(X)
            x_pca = []
            for x in X:
                x_pca.append(pca.transform(x).tolist())
            pickle_dumped = dumps(x_pca)
            with open(pca_mfcc_result_path, 'wb') as f:
                f.write(pickle_dumped)
        validation = None
        method = 'pca'
        cache_fun = None
        result_path = None
        if method == 'tsne':
            cache_fun = cache_fit_tsne
            result_path = tsne_mfcc_global_result_path
        elif method == 'pca':
            cache_fun = cache_pca
            result_path = pca_mfcc_global_result_path
        assert cache_fun
        assert result_path
        cache_fun(result_path, self._x)
        with open(result_path, 'rb') as f:
            x_to_be_drawn = loads(f.read())
            out_shape=(7, 3)
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
            shape_hash = str(out_shape[0])+'-'+str(out_shape[1])
            draw_result_path = join(working_dir, f'draw_{method}_mfcc_global_result_{shape_hash}.png')
            draw_X2(x_to_be_drawn, draw_result_path,
                   out_shape=out_shape, drawing_subsamples=True, agg_lists=agg_lists,
                   sup_title=f'{method}_mfcc_global', validation=validation)
            points_in_single_pointcloud = 10000
            offset_points_between_frames = 5000
            reverted = True
            fps = 20
            out_video_format = 'mp4'
            dpi = 100
            for (meshgrid_alpha, scatter_alpha) in [(1.0, 0.0), (0.0, 1.0)]:
                animated_result_path = join(working_dir, f'animated_{method}_mfcc_global_result'
                                                         f'_{points_in_single_pointcloud}'
                                                         f'_{offset_points_between_frames}'
                                                         f'_reverted{reverted}'
                                                         f'_mesh{meshgrid_alpha}'
                                                         f'_scatter{scatter_alpha}'
                                                         f'_fps{fps}_dpi{dpi}.{out_video_format}')
                animate_language_change(x_to_be_drawn, animated_result_path,
                                        reverted=reverted, fps=fps, save_pngs=False,
                                        scatter_alpha=scatter_alpha, meshgrid_alpha=meshgrid_alpha, dpi=dpi)
