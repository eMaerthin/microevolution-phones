from functools import reduce
import json
from os import walk
from os.path import (isfile, join)
import ssl

import fire

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
        jsons = list(filter(lambda x: all([x.endswith('json'),
                                           not(x.endswith('result.json')),
                                           not(x.endswith('common.json'))]),
                            filenames))
        if len(jsons) > 0:
            dir_series = [(dirpath[len(dir):], jsons)]
            if verbose > 1:
                print(f'Added subject-series pair: {dir_series}')
            series.extend(dir_series)
    series_sum = sum_series(series)
    todo = list(filter(lambda s: any([not(isfile(join(dir, s[0],
                                                      result_name_fun(item))))
                                      for item in s[1]]), series))
    unprocessed_series_sum = sum_series(todo)
    if verbose > 0:
        print(f'{len(series)} subject(s) ({series_sum} series) \
              in the database: {series}')
        print(f'{len(todo)} not yet processed subject(s) \
              ({unprocessed_series_sum} series): {todo}')
    return series, todo

def load_settings(working_dir, json_filename, verbose):
    json_path = join(working_dir, json_filename)
    with open(json_path) as f:
        if verbose > 0:
            print(f' importing settings: {json_path}')
        print(json_path)
        d = json.load(f)
        return d

def merge_settings(setting1, setting2, raise_error_on_conflict_values = False):
    settings = {**setting1, **setting2}
    l=[setting1, setting2]
    settings['metadata'] = reduce(lambda x, y: {**x, **y},
                                  [s.get('metadata') for s in l
                                   if s.get('metadata') is not None],
                                  {})
    if raise_error_on_conflict_values:
        # TODO
        pass
    return settings

def process_pipeline(pipeline_fun, result_name_fun,
                     subjects_homedir = '../subjects/', verbose = 1):
    _, series_to_process = list_subjects(subjects_homedir, verbose,
                                         result_name_fun)
    for subject, series in series_to_process:
        working_dir = join(subjects_homedir, subject)
        common_settings = load_settings(working_dir, 'common.json', verbose)
        for series_json_filename in series:
            settings = load_settings(working_dir, series_json_filename, verbose)
            series_schema = SeriesSchema()
            data = series_schema.dump(merge_settings(common_settings, settings))
            pipeline_fun(series_json_filename, working_dir, data, verbose)


if __name__ == '__main__':
    fire.Fire({
              'run': run_all_pipelines,
              'phonemes': process_phonemes_pipeline,
              'formants': process_formants_pipeline,
              })
