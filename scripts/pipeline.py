from functools import reduce
import json
from os import walk
from os.path import (isfile, join)

from schemas import SeriesSchema


class Pipeline(object):
    def __init__(self, verbose, subjects_dir):
        self._verbose = verbose
        self._subjects_dir = subjects_dir

    @property
    def verbose(self):
        return self._verbose

    @property
    def subjects_dir(self):
        return self._subjects_dir

    ''' this should be implemented by derived classes'''
    def pipeline(self, *args):
        raise NotImplementedError

    ''' this should be implemented by derived classes'''
    @staticmethod
    def result_filename(*args):
        raise NotImplementedError

    def process_pipeline(self):
        def list_subjects(dir, verbose, result_name_fun):

            def sum_series(_series):
                return reduce(lambda x, y: x + len(y[1]), _series, 0)

            series = []
            for (dirpath, dirnames, filenames) in walk(dir):
                jsons = list(filter(lambda x: all([x.endswith('json'),
                                                   not (x.endswith('result.json')),
                                                   not (x.endswith('common.json'))]),
                                    filenames))
                if len(jsons) > 0:
                    dir_series = [(dirpath[len(dir):], jsons)]
                    if verbose > 1:
                        print(f'Added subject-series pair: {dir_series}')
                    series.extend(dir_series)
            series_sum = sum_series(series)
            todo = list(filter(lambda s: any([not (isfile(join(dir, s[0],
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
            try:
                with open(json_path) as f:
                    if verbose > 0:
                        print(f' importing settings: {json_path}')
                    print(json_path)
                    d = json.load(f)
                    return d
            except (EnvironmentError, json.decoder.JSONDecodeError) as e:
                if verbose > 0:
                    print(f' An error occurred ({e}) - defaulting to empty dict')
                return {}


        def merge_settings(setting1, setting2, raise_error_on_conflict_values=False):
            _settings = {**setting1, **setting2}
            _settings['metadata'] = reduce(lambda x, y: {**x, **y},
                                           [s.get('metadata') for s in [setting1, setting2]
                                            if s.get('metadata') is not None],
                                           {})
            if raise_error_on_conflict_values:
                '''TODO (add proper handling this) 
                raise ValueError()
                '''
                pass

            return _settings

        _, series_to_process = list_subjects(self.subjects_dir, self.verbose,
                                             self.result_filename)

        for subject, series in series_to_process:
            working_dir = join(self.subjects_dir, subject)
            common_settings = load_settings(working_dir, 'common.json', self.verbose)
            for series_json_filename in series:
                settings = load_settings(working_dir, series_json_filename, self.verbose)
                series_schema = SeriesSchema()
                data = series_schema.dump(merge_settings(common_settings, settings))
                self.pipeline(series_json_filename, working_dir, data)
