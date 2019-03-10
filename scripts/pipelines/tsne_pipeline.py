from itertools import compress
import json
from os.path import join
from pickle import (dumps, loads)

import numpy as np
import pandas as pd
import parse

from decorators import (check_if_already_done, timeit)
from dimensionality_reduction import (draw_composition, fit_tsne, fit_pca)
from marshmallow import pprint
from schemas import *
from pipeline import Pipeline
from pipelines.formants_pipeline import FormantsPipeline
# from pipelines.spectrogram_pipeline import SpectrogramPipeline

class TsnePipeline(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_tsne_result.json'

    @staticmethod
    def filename_prerequisites():
        return [FormantsPipeline.result_filename]

    def subject_pipeline(self, working_dir, series, common_settings):
        X = []
        y = []
        format = 'season{s}_part{p}.json'
        series = sorted(series, key=lambda x: (int(parse.parse(format, x)['s']),
                                               int(parse.parse(format, x)['p'])))
        for series_json_filename in series:
            settings = self.load_settings(working_dir, series_json_filename, self.verbose)
            series_json_path = join(working_dir, series_json_filename)
            series_schema = SeriesSchema()
            series_settings = series_schema.dump(self.merge_settings(common_settings, settings))
            schema = PhonemesFormantsSchema()
            formants_result_path = self.filename_prerequisites()[0](series_json_path)
            formants_csv_result_path = f'{formants_result_path[:-5]}.csv'

            @timeit
            @check_if_already_done(formants_csv_result_path, self.verbose) #, ignore_already_done=True)
            def store_formants_result_as_csv(formants_result_path, formants_csv_result_path):
                with open(formants_result_path, 'r') as f:
                    print(f' phonemes_formants_result_path: {formants_result_path}')
                    json_file = json.load(f)
                    result = schema.load(json_file)
                    if self.verbose > 1:
                        pprint(result, indent=4)
                    df = pd.DataFrame.from_dict(result['formants_info'])
                    df = df.N_largest_local_max_f.apply(pd.Series).merge(df, left_index=True,
                                                                         right_index=True).drop(['N_largest_local_max_f'],
                                                                                                axis=1)
                    df = df.N_largest_local_max_s.apply(pd.Series).merge(df, left_index=True,
                                                                         right_index=True).drop(['N_largest_local_max_s'],
                                                                                                axis=1)
                    # df.rename(columns=lambda x: x[:-1] + 'signal' if x.endswith('_x') else x, inplace=True)
                    df.rename(columns=lambda x: x[:-1] + 'formant' if x.endswith('_y') else x, inplace=True)
                    # filter nans:
                    for c in df.filter(regex='formant$', axis=1).columns:
                        df = df[df[c].notnull()]
                    df.to_csv(formants_csv_result_path)

            store_formants_result_as_csv(formants_result_path, formants_csv_result_path)

            df = pd.read_csv(formants_csv_result_path)
            y.append(df.word.values)

            df = df[[col for col in df.columns if col.endswith('_formant') or col.endswith('_signal')]]
            X.append(df.values)
        filter = []
        only_vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
        only_EH_and_NG = ['EH', 'NG']
        only_AH_EH = ['AH', 'EH']
        only_SZ = ['S', 'Z']
        only_plus = ['R','L']
        filter = [] #'AH','IY'] #['AH', 'IY'] # ['AH'] # only_vowels # [] # ['B','P']

        if filter and len(filter):
            for i in range(len(y)):
                subset = [any(substring in phoneme for substring in filter) for phoneme in y[i]]
                X[i] = X[i][subset, :]
                y[i] = list(compress(y[i], subset))

        filter_hash = '_'.join(sorted(filter))
        tsne_result_path = join(working_dir, f'tsne_result_{filter_hash}.pickle')
        pca_result_path = join(working_dir, f'pca_result_{filter_hash}.pickle')

        @timeit
        @check_if_already_done(tsne_result_path, self.verbose)  # , ignore_already_done=True)
        def cache_fit_tsne(tsne_result_path, X, y):
            tsne_X = fit_tsne(X, n_iter=250)  # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            pickle_dumped = dumps(tsne_X)
            with open(tsne_result_path, 'wb') as f:
                f.write(pickle_dumped)


        @timeit
        @check_if_already_done(pca_result_path, self.verbose, ignore_already_done=True)
        def cache_pca(pca_result_path, X):
            pca = fit_pca(X, verbose=self.verbose)  # , n_iter=3000) #, n_iter=10000, n_iter_without_progress=100, perplexity=200)
            x_pca = []
            for x in X:
                x_pca.append(pca.transform(x).tolist())
            pickle_dumped = dumps(x_pca)
            with open(pca_result_path, 'wb') as f:
                f.write(pickle_dumped)

        method = 'tsne'
        if method == 'tsne':
            cache_fit_tsne(tsne_result_path, X, y)
            phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                            'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
                            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
            if True:
            # for phoneme in phoneme_list:
                filter_local = []
                with open(tsne_result_path, 'rb') as f:
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
                            y_list[i] = yi.tolist()
                    #tsne_X = [x for x, e in enumerate(z) for z, f in enumerate(tsne_X) if y[f][e] in filter]
                    #tsne_X = [x for x, e in enumerate(z) for z, f in enumerate(tsne_X) if y[f][e] in filter]
                    agg_shape=(1, 1)
                    out_shape=(3, 15)
                    shape_hash = str(agg_shape[0])+'-'+str(agg_shape[1])+'_'+str(out_shape[0])+'-'+str(out_shape[1])
                    draw_result_path = join(working_dir, f'draw_tsne_result_{filter_hash}_{shape_hash}_{filter_local}.png')
                    draw_composition(tsne_X, y_list, series, draw_result_path, verbose=self.verbose,
                              out_shape=out_shape, agg_shape=agg_shape, drawing_subsamples=True, suptitle=f'tsne trained on: {filter} -> visualization filter: {filter_local}')
        else:

            cache_pca(pca_result_path, X)
            with open(pca_result_path, 'rb') as f:
                pca_X = loads(f.read())
                agg_shape = (1, 5)
                out_shape = (3, 3)
                shape_hash = str(agg_shape[0]) + '-' + str(agg_shape[1]) + '_' + str(out_shape[0]) + '-' + str(
                    out_shape[1])
                draw_result_path = join(working_dir, f'draw_pca_result_{filter_hash}_{shape_hash}.png')
                draw_composition(pca_X, y, series, draw_result_path, verbose=self.verbose,
                                 out_shape=out_shape, agg_shape=agg_shape)

'''
TODO: 
1. decorator does not need an extra parameter - 
they can read the result_path from *args, also all check_if_already_done decorated methods 
can be converted into members of a class that has automatically inserted check_if_already_
done + require result_path param 
2. those decorated methods should have better implementation of the what-if-already-done
meaning that they should already return required value if any
'''