from functools import reduce
import json
import os
from os import walk
from os.path import (isfile, join)
import ssl
import sys

import fire

from decorators import check_if_already_done
from formants_pipeline import (formants_pipeline, result_formants)
from phoneme_pipeline import (phoneme_pipeline, result_phonemes)
from schemas import SeriesSchema

ssl._create_default_https_context = ssl._create_stdlib_context

def process_phonemes_pipeline(subjects_homedir = '../subjects/', verbose = 1):
    process_pipeline(phoneme_pipeline, result_phonemes, subjects_homedir,
                     verbose)

def process_formants_pipeline(subjects_homedir = '../subjects/', verbose = 1):
    process_pipeline(formants_pipeline, result_formants, subjects_homedir,
                     verbose)

def run_all_pipelines(subjects_homedir = '../subjects/', verbose = 1):
    process_phonemes_pipeline(subjects_homedir, verbose)
    process_formants_pipeline(subjects_homedir, verbose)

def sum_series(series):
    return reduce(lambda x, y: x + len(y[1]), series, 0)

def list_subjects(dir, verbose, result_name_fun):
    series = []
    for (dirpath, dirnames, filenames) in walk(dir):
        jsons = list(filter(lambda x: all([x.endswith("json"),
                                           not(x.endswith("result.json"))]),
                            filenames))
        if len(jsons) > 0:
            dir_series = [(dirpath[len(dir):], jsons)]
            if verbose > 1:
                print(f'Added subject-series pair: {dir_series}')
            series.extend(dir_series)
    series_sum = sum_series(series)
    unprocessed_series = list(filter(lambda s:
                                     any([not(isfile(join(dir, s[0],
                                                          result_name_fun(item))))
                                          for item in s[1]]), series))
    unprocessed_series_sum = sum_series(unprocessed_series)
    if verbose > 0:
        print(f'{len(series)} subject(s) ({series_sum} series) \
              in the database in total: {series}')
        print(f'{len(unprocessed_series)} subject(s) unprocessed \
              ({unprocessed_series_sum} series): {unprocessed_series}')
        return series, unprocessed_series


def process_pipeline(pipeline_fun, result_name_fun,
                     subjects_homedir = '../subjects/', verbose = 1):
    _, series_to_process = list_subjects(subjects_homedir, verbose,
                                         result_name_fun)
    for subject, series in series_to_process:
        working_dir = join(subjects_homedir, subject)
        for (series_path, series_json_filename) in ((join(working_dir, s), s)
                                                    for s in series):
            with open(series_path) as f:
                if verbose > 0:
                    print(f' series_path: {series_path}')
                d = json.load(f)
                series_schema = SeriesSchema()
                data = series_schema.dump(d)
                pipeline_fun(series_path, series_json_filename, working_dir,
                             data, verbose)


if __name__ == '__main__':
    fire.Fire({
              'run': run_all_pipelines,
              'phonemes': process_phonemes_pipeline,
              'formants': process_formants_pipeline,
              })
