from abc import ABC, abstractmethod
from functools import reduce
import json
from os import walk
from os.path import (isfile, join)
import parse
import string

from natsort import natsorted

from schemas import SeriesSchema


def protect(*protected):
    """
    Returns a metaclass that protects all attributes given as strings
    """
    class Protect(type):
        has_base = False

        def __new__(mcs, name, bases, attrs):
            if mcs.has_base:
                for attribute in attrs:
                    if attribute in protected:
                        raise AttributeError(f'Overriding of attribute "{attribute}" not allowed.')
            mcs.has_base = True
            class_ = super().__new__(mcs, name, bases, attrs)
            return class_
    return Protect


class Pipeline(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    _verbose = 0
    _subjects_dir = ''

    @property
    def verbose(self):
        """
        Currently there are few supported verbosity levels:
        0 - silent
        1 - verbose
        2 - extra verbose
        :return: A verbosity level
        """
        return self._verbose

    @verbose.setter
    def verbose(self, new_value):
        self._verbose = new_value

    @property
    def subjects_dir(self):
        return self._subjects_dir

    @subjects_dir.setter
    def subjects_dir(self, new_value):
        self._subjects_dir = new_value

    @abstractmethod
    def series_pipeline(self, series_json_path, series_settings):
        """
        Each pipeline has a specific inner part which is called from process_pipeline()

        :param series_json_path:
        :param series_settings:
        :return:
        """

    def load_settings(self, working_dir, settings_filename):
        """

        :param working_dir:
        :param settings_filename:
        :return:
        """
        settings_path = join(working_dir, settings_filename)
        try:
            with open(settings_path) as f:
                if self.verbose > 0:
                    print(f' importing settings: {settings_path}')
                print(settings_path)
                d = json.load(f)
                return d
        except (EnvironmentError, json.decoder.JSONDecodeError) as e:
            if self.verbose > 0:
                print(f' An error occurred ({e}) - defaulting to empty dict')
            return {}

    @staticmethod
    def merge_settings(setting1, setting2, raise_error_on_conflict_values=True):
        """

        :param setting1:
        :param setting2:
        :param raise_error_on_conflict_values:
        :return:
        """
        def check_for_conflicts(merged_value, key, proposed_value):
            if isinstance(proposed_value, dict):
                for inner_key, inner_value in proposed_value.items():
                    check_for_conflicts(merged_value[inner_key],
                                        inner_key, inner_value)
            elif proposed_value != merged_value:
                raise ValueError(f'Conflicting values {proposed_value} vs {merged_value} for key {key}')

        settings = {**setting1, **setting2}
        for key, value in settings.items():
            if isinstance(value, dict):
                settings[key] = reduce(lambda x, y: {**x, **y},
                                       (s[key] for s in (setting1, setting2) if s.get(key)),
                                       dict())

        if raise_error_on_conflict_values:
            for key, value in setting1.items():
                check_for_conflicts(settings[key], key, value)

        return SeriesSchema().dump(settings)

    @staticmethod
    def deprecated_sort_series(series, series_format='{}.json'):  # other popular format is '{}_{}.json'
        """

        :param series:
        :param series_format:
        :return:
        """
        def sorting_rules(text):
            def conditional_cast(expression):
                """ this will attempt to sort numbers using natural order algorithm """
                strip_letters = expression.strip(string.ascii_letters)
                if len(strip_letters) > 0:
                    return int(strip_letters)
                return expression

            return tuple(conditional_cast(p) for p in parse.parse(series_format, text))

        series = sorted(series, key=sorting_rules)
        return series

    @staticmethod
    def sort_series(series):
        """

        :param series:
        :return:
        """
        return natsorted(series)

    @abstractmethod
    def subject_pipeline(self, working_dir, series, common_settings):
        """

        :param working_dir:
        :param series:
        :param common_settings:
        :return:
        """
        series = self.sort_series(series)
        for series_json_filename in series:
            settings = self.load_settings(working_dir, series_json_filename, self.verbose)
            series_json_path = join(working_dir, series_json_filename)
            series_settings = self.merge_settings(common_settings, settings)
            self.series_pipeline(series_json_path, series_settings)

    @abstractmethod
    def global_pipeline(self, subject_and_series_to_process):
        """
        A standard implementation of the global pipeline is to
        run the subject pipeline on each subject' series
        but this can be overridden for instance if the purpose of the pipeline
        is to collect globally all pieces of information together.
        :param subject_and_series_to_process: a
        :return:
        """
        for subject, series in subject_and_series_to_process:
            working_dir = join(self.subjects_dir, subject)
            common_settings = self.load_settings(working_dir, 'common.json', self.verbose)
            self.subject_pipeline(working_dir, series, common_settings)

    @staticmethod
    @abstractmethod
    def result_filename(*args, **kwargs):
        """
        pipeline should have a fixed filename pattern for storing final results.
        :param args:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def result_filename_postprocessed(*args, **kwargs):
        """
        emergency pattern that, if found, will skip processing the pipeline
        :param args:
        :param kwargs:
        :return:
        """

    @staticmethod
    @abstractmethod
    def filename_prerequisites(*args, **kwargs):
        """
        A pipeline can have a list of filename patterns - prerequisites to be able to run the pipeline on.

        :param args:
        :param kwargs:
        :return:
        """

    @staticmethod
    @abstractmethod
    def filename_prerequisites_postprocessed(*args, **kwargs):
        """
        An emergency pattern that, if found, will skip processing the pipeline
        """

    def process_pipeline(self):
        """
        This is the central method of the Pipeline class and an entry point to each of the subclasses.
        This method is not abstract on purpose and is indented to be not modified in subclasses.
        """

        def subjects_and_series(dir, verbose, result_name_fun, filename_prerequisites, postprocessed,
                                prerequisites_postprocessed):
            def only_not_processed(subject, jsons, dir=dir):
                items = [item for item in jsons if not isfile(join(dir, subject, result_name_fun(item)))]
                try:
                    items = [item for item in jsons if not isfile(join(dir, subject, postprocessed(item)))]
                except NotImplementedError:
                    pass
                return subject, items

            def sum_series(series):
                return reduce(lambda x, y: x + len(y[1]), series, 0)

            def only_todo(subject, jsons):
                if isinstance(filename_prerequisites(), list):
                    items = [item for item in jsons if all(isfile(join(dir, subject, fn_prerequisite(item)))
                                                           for fn_prerequisite in filename_prerequisites())]
                    try:
                        items2 = [item for item in jsons if all(isfile(join(dir, subject, fn_prerequisite(item)))
                                                           for fn_prerequisite in prerequisites_postprocessed())]
                        items = list(set(items) | set(items2))
                    except NotImplementedError:
                        pass

                    return subject, items
                else:
                    return subject, jsons

            series = []
            for (dirpath, dirnames, filenames) in walk(dir):
                jsons = list(filter(lambda x: all([x.endswith('json'),
                                                   not (x.startswith('.')),
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
                    len(only_todo(s[0], s[1])[1]) > 0]

            if verbose > 0:
                print(f'{len(series)} subject(s) ({sum_series(series)} series) in the database: {series}')
                sum_not_processed = sum_series(not_processed)
                print(f'Not processed: {len(not_processed)} subjects \t {sum_not_processed} series: \t {not_processed}')
                print(f'To be processed: {len(todo)} subjects \t  {sum_series(todo)} series: \t  {todo}')
            return todo

        subject_and_series_to_process = subjects_and_series(self.subjects_dir, self.verbose,
                                                            self.result_filename, self.filename_prerequisites,
                                                            self.result_filename_postprocessed,
                                                            self.filename_prerequisites_postprocessed)

        self.global_pipeline(subject_and_series_to_process)
