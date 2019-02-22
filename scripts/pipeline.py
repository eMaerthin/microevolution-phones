from functools import reduce
import json
from os import walk
from os.path import (isfile, join)

from schemas import SeriesSchema


class Pipeline(object):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, verbose, subjects_dir):
        self._verbose = verbose
        self._subjects_dir = subjects_dir

    @property
    def verbose(self):
        return self._verbose

    @property
    def subjects_dir(self):
        return self._subjects_dir

    def series_pipeline(self, series_json_path, series_settings):
        """
        each pipeline has a specific inner part which is called from process_pipeline()
        """
        raise NotImplementedError

    @staticmethod
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

    @staticmethod
    def merge_settings(setting1, setting2, raise_error_on_conflict_values=False):
        def check_for_conflicts(d1, k, v):
            if isinstance(v, dict):
                for _k, _v in v.items():
                    check_for_conflicts(d1, _k, _v)
            elif v != d1[k]:
                raise ValueError(f'Conflicting values {v} vs {d1[k]} for key {k}')

        settings = {**setting1, **setting2}
        settings['metadata'] = reduce(lambda x, y: {**x, **y},
                                       (s['metadata'] for s in (setting1, setting2)
                                        if s.get('metadata')),
                                       {})
        if raise_error_on_conflict_values:
            for k, v in setting1.items():
                check_for_conflicts(settings, k, v)

        return settings

    def subject_pipeline(self, working_dir, series, common_settings):
        for series_json_filename in series:
            settings = self.load_settings(working_dir, series_json_filename, self.verbose)
            series_json_path = join(working_dir, series_json_filename)
            series_schema = SeriesSchema()
            series_settings = series_schema.dump(self.merge_settings(common_settings, settings))
            self.series_pipeline(series_json_path, series_settings)

    @staticmethod
    def result_filename(*args, **kwargs):
        """
        pipeline should have a fixed filename pattern for storing final results.
        """
        raise NotImplementedError

    @staticmethod
    def filename_prerequisites(*args, **kwargs):
        """
        pipeline can have a list of filename patterns - prerequisites to be able to run the pipeline on.
        """
        raise NotImplementedError

    def process_pipeline(self):
        def subjects_and_series(dir, verbose, result_name_fun, filename_prerequisites):
            def only_not_processed(subject, jsons, dir=dir):
                return subject, [item for item in jsons if not isfile(join(dir, subject, result_name_fun(item)))]

            def sum_series(series):
                return reduce(lambda x, y: x + len(y[1]), series, 0)

            def only_todo(subject, jsons):
                if isinstance(filename_prerequisites(), list):
                    return subject, [item for item in jsons if all(
                        isfile(join(dir, subject, fn_prerequisite(item))) for fn_prerequisite in filename_prerequisites()
                    )]
                else:
                    return subject, jsons

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

            not_processed = [only_not_processed(s[0], s[1]) for s in series if
                             len(only_not_processed(s[0], s[1])[1]) > 0]

            todo = [only_todo(s[0], s[1]) for s in not_processed if
                    len(only_todo(s[0], s[1])[1]) > 0 ]

            if verbose > 0:
                print(f'{len(series)} subject(s) ({sum_series(series)} series) in the database: {series}')
                sum_not_processed = sum_series(not_processed)
                print(f'Not processed: {len(not_processed)} subjects \t {sum_not_processed} series: \t {not_processed}')
                print(f'To be processed: {len(todo)} subjects \t  {sum_series(todo)} series: \t  {todo}')
            return todo

        series_to_process = subjects_and_series(self.subjects_dir, self.verbose,
                                                self.result_filename, self.filename_prerequisites)

        for subject, series in series_to_process:
            working_dir = join(self.subjects_dir, subject)
            common_settings = self.load_settings(working_dir, 'common.json', self.verbose)
            self.subject_pipeline(working_dir, series, common_settings)
