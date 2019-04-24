from itertools import compress
import json
from os.path import join
from pickle import (dumps, loads)

import pandas as pd

from decorators import (check_if_already_done, timeit)
from dimensionality_reduction import (draw_composition, fit_tsne, fit_pca)
from marshmallow import pprint
from schemas import *
from pipeline import Pipeline
from pipelines.spectrogram_pipeline import SpectrogramPipeline

class TsnePipeline2(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_tsne2_result.json'

    @staticmethod
    def filename_prerequisites():
        return [SpectrogramPipeline.result_filename]

    @staticmethod
    def filename_prerequisites_postprocessed():
        return [SpectrogramPipeline.result_filename_postprocessed]

    def subject_pipeline(self, working_dir, series, common_settings):
        X = []
        y = []
        series = self.sort_series(series)  # , '{}_{}.json')
        for series_json_filename in series:
            series_json_path = join(working_dir, series_json_filename)
            schema = PhonemesSpectrogramsSchema()
            spectrograms_result_path = self.filename_prerequisites()[0](series_json_path)
            spectrograms_csv_result_path = f'{spectrograms_result_path[:-5]}.csv'

            @timeit
            @check_if_already_done(spectrograms_csv_result_path, self.verbose) #, ignore_already_done=True)
            def store_spectrograms_result_as_csv(spectrograms_result_path, spectrograms_csv_result_path):
                print(f'Why I am here? result_path: {spectrograms_csv_result_path}')
                with open(spectrograms_result_path, 'r') as f_csv:
                    print(f' phonemes_spectrograms_result_path: {spectrograms_result_path}')
                    json_file = json.load(f_csv)
                    result = schema.load(json_file)
                    if self.verbose > 1:
                        pprint(result, indent=4)
                    df = pd.DataFrame.from_dict(result['spectrograms_info'])
                    df = df.signal.apply(pd.Series).add_suffix('_raw_signal').merge(df, left_index=True,
                                                                                    right_index=True).drop(['signal'],
                                                                                                           axis=1)
                    # filter nans:
                    for c in df.filter(regex='signal$', axis=1).columns:
                        df = df[df[c].notnull()]
                    df.to_csv(spectrograms_csv_result_path)

            store_spectrograms_result_as_csv(spectrograms_result_path, spectrograms_csv_result_path)

            df = pd.read_csv(spectrograms_csv_result_path)
            df = df[df.i == 0]
            y.append(df.word.values)

            df = df[[col for col in df.columns if col.endswith('_signal')]]
            X.append(df.values)
        filter = []
        only_vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
        only_EH_and_NG = ['EH', 'NG']
        only_AH_EH = ['AH', 'EH']
        only_SZ = ['S', 'Z']
        only_plus = ['R','L']
        filter = []  # ['B','P']  # ['AH']  # 'AH'] #'AH','IY'] #['AH', 'IY'] # ['AH'] # only_vowels # [] # ['B','P']

        if filter and len(filter):
            for i in range(len(y)):
                subset = [any(substring in phoneme for substring in filter) for phoneme in y[i]]
                X[i] = X[i][subset, :]
                y[i] = list(compress(y[i], subset))

        filter_hash = '_'.join(sorted(filter))
        tsne2_result_path = join(working_dir, f'tsne2_result_{filter_hash}.pickle')
        pca_result_path = join(working_dir, f'pca_result_{filter_hash}.pickle')

        @timeit
        @check_if_already_done(tsne2_result_path, self.verbose)  # , ignore_already_done=True)
        def cache_fit_tsne(tsne2_result_path, X, y):
            tsne_X = fit_tsne(X, n_iter=1000)
            # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            pickle_dumped = dumps(tsne_X)
            with open(tsne2_result_path, 'wb') as f:
                f.write(pickle_dumped)

        @timeit
        @check_if_already_done(pca_result_path, self.verbose, ignore_already_done=True)
        def cache_pca(pca_result_path, X):
            pca = fit_pca(X, verbose=self.verbose)
            x_pca = []
            for x in X:
                x_pca.append(pca.transform(x).tolist())
            pickle_dumped = dumps(x_pca)
            with open(pca_result_path, 'wb') as f:
                f.write(pickle_dumped)
        validation = None
        method = 'tsne'
        if method == 'tsne':
            cache_fit_tsne(tsne2_result_path, X, y)
            '''
            phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                            'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
                            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
            for phoneme in phoneme_list:
                filter_local = [phoneme]
                '''
            if True:
                filter_local = []
                with open(tsne2_result_path, 'rb') as f:
                    tsne_X = loads(f.read())

                    # filter_local = [] #'['AH', 'IY'] #['IY']
                    y_list = y.copy()
                    if filter_local:
                        print('i am here')
                        for i, z in enumerate(tsne_X):
                            tsne_X[i] = [x for e, x in enumerate(z) if y[i][e] in filter_local]
                            y_list[i] = [a for e, a in enumerate(y[i]) if a in filter_local]
                    else:
                        for i, yi in enumerate(y):
                            if not isinstance(yi, list):
                                y_list[i] = yi.tolist()
                    #tsne_X = [x for x, e in enumerate(z) for z, f in enumerate(tsne_X) if y[f][e] in filter]
                    #tsne_X = [x for x, e in enumerate(z) for z, f in enumerate(tsne_X) if y[f][e] in filter]
                    agg_shape=(1, 3)
                    out_shape=(3, 6)
                    shape_hash = str(agg_shape[0])+'-'+str(agg_shape[1])+'_'+str(out_shape[0])+'-'+str(out_shape[1])
                    draw_result_path = join(working_dir,
                                            f'draw_tsne2_result_{filter_hash}_{shape_hash}_{filter_local}.png')
                    draw_composition(tsne_X, y_list, draw_result_path, verbose=self.verbose,
                                     out_shape=out_shape, agg_shape=agg_shape, drawing_subsamples=True,
                                     sup_title=f'tsne2 trained on: {filter} -> visualization filter: {filter_local}',
                                     validation=validation)
        else:
            raise NotImplementedError()
