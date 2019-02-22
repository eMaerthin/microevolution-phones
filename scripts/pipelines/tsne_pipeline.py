import json
from os.path import join
from pickle import (dumps, loads)

import numpy as np
import pandas as pd

from audio_processors import prepare_wav_input
from decorators import check_if_already_done
from dimensionality_reduction import (draw_tsne, fit_tsne)
from marshmallow import pprint
from schemas import *
from pipeline import Pipeline
from pipelines.formants_pipeline import FormantsPipeline


class TsnePipeline(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_tsne_result.json'

    @staticmethod
    def filename_prerequisites():
        return [FormantsPipeline.result_filename]

    def compute_tsne_and_draw(self, formants_result_path):
        pass

    def subject_pipeline(self, working_dir, series, common_settings):
        X = []
        y = []
        for series_json_filename in series:
            settings = self.load_settings(working_dir, series_json_filename, self.verbose)
            series_json_path = join(working_dir, series_json_filename)
            series_schema = SeriesSchema()
            series_settings = series_schema.dump(self.merge_settings(common_settings, settings))
            schema = PhonemesFormantsSchema()
            formants_result_path = self.filename_prerequisites()[0](series_json_path)
            formants_csv_result_path = f'{formants_result_path[:-5]}.csv'

            @check_if_already_done(formants_csv_result_path, self.verbose)
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
                    df.rename(columns=lambda x: x[:-1] + 'signal' if x.endswith('_x') else x, inplace=True)
                    df.rename(columns=lambda x: x[:-1] + 'formant' if x.endswith('_y') else x, inplace=True)
                    df.to_csv(formants_csv_result_path)

            store_formants_result_as_csv(formants_result_path, formants_csv_result_path)

            df = pd.read_csv(formants_csv_result_path)
            y.append(df.word.values)

            df = df[[col for col in df.columns if col.endswith('_formant')]]
            X.append(df.values)

        tsne_result_path = join(working_dir, 'tsne_result.pickle')

        @check_if_already_done(tsne_result_path, self.verbose)
        def cache_fit_tsne(tsne_result_path):
            tsne_X = fit_tsne(X)
            pickle_dumped = dumps(tsne_X)
            with open(tsne_result_path, 'wb') as f:
                f.write(pickle_dumped)

        cache_fit_tsne(tsne_result_path)

        with open(tsne_result_path, 'rb') as f:
            tsne_X = loads(f.read())
            only_vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
            only_SZ = ['S', 'Z']
            draw_tsne(tsne_X, y, filter_phonemes=only_SZ)

        # get_formants
        # join(working_dir
        # pipeline(self, series_json_path, series_settings):
        exit(1)
