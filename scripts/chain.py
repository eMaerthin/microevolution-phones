from abc import ABCMeta, abstractmethod
from functools import reduce
import json
from os import (makedirs, walk)
from os.path import (exists, isfile, join)

from natsort import natsorted
from toposort import toposort_flatten

from schemas import SeriesSchema


def protect_abc(*protected):
    """
    Returns a metaclass that protects all attributes given as strings
    """
    class Protect(ABCMeta):
        has_base = False

        def __new__(mcs, name, bases, attrs):
            if mcs.has_base:
                for attribute in attrs:
                    if attribute in protected:
                        raise AttributeError(f'Overriding attribute '
                                             f'"{attribute}" is not allowed.')
            mcs.has_base = True
            class_ = super().__new__(mcs, name, bases, attrs)
            return class_
    return Protect


class Chain(metaclass=protect_abc("load_settings", "merge_data_settings")):
    """
    Base class for concrete chain processes like Phoneme or Spectrogram.
    It stores a dictionary of subclasses class names and class references
    and in addition it stores a topologically ordered list of subclasses
    sorted with their dependencies, i.e. if for instance subclass A depends
    on nothing, subclass B depends on A and subclass C depends on both A and B,
    then ordered list will be [A,B,C]

    Chain has few read-write properties:
     - verbose (sets verbosity)
     - base_dir (sets parent directory where subjects from the dataset are placed)
     - process_settings (this adjust how to process the chain)

    In addition there is one factory classmethod called from_verbose_and_base_dir
    that initializes Chain with verbose and base_dir.

    Subclass has to provide an implementation of following abstract methods:
    - sample_result_filename
    - sample_layer

    chain entry point:
    - process() method that 1) populates a dataset to be traversed
        and then 2) process the chain of layers in fixed order:
         * dataset_layer that can be equipped with pre- and post- processing logic,
         and it iterates over subjects and runs subject_layer on each of them
         * subject_layer that can be equipped with pre- and post- processing logic,
         and it iterates over samples and runs sample_layer on each of them
         * sample_layer that has an implementation specific for each of the subclass
    """

    subclasses = {}
    s = {}
    ordered_subclasses = []
    requirements = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
        chains_dependencies = {value: set(value.requirements)
                               for value in cls.subclasses.values()}
        cls.ordered_subclasses[:] = toposort_flatten(chains_dependencies,
                                                     sort=False)

    def __init__(self):
        self._verbose = 0
        self._base_dir = ''
        self._results_dir = ''
        self._process_settings = {'raise_error_on_conflict_values': True}

    @property
    def process_settings(self):
        return self._process_settings

    @process_settings.setter
    def process_settings(self, new_value):
        if not isinstance(new_value, dict):
            raise TypeError('expected argument of type dict')
        self._process_settings = new_value

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
        new_value = int(new_value)
        if new_value < 0 or new_value > 2:
            raise ValueError('expected number between 0 and 2')
        self._verbose = new_value

    @property
    def base_dir(self):
        """
        An absolute path to base directory (dataset level)
        :return:
        """
        return self._base_dir

    @base_dir.setter
    def base_dir(self, new_value):
        if not exists(new_value):
            raise ValueError(f'base_dir {new_value} does not exist!')
        self._base_dir = str(new_value)
        self._results_dir = join(self._base_dir, 'results')

    @property
    def results_dir(self):
        return self._results_dir

    @classmethod
    def from_verbose_and_base_dir(cls, verbose, base_dir):
        """
        Initializes class with given verbose and base_dir values
        :param verbose: verbosity level
        :param base_dir: An absolute path to base directory
        :return:
        """
        pipeline = cls()
        pipeline.verbose = verbose
        pipeline.base_dir = base_dir
        return pipeline

    @staticmethod
    def merge_data_settings(setting1, setting2,
                            raise_error_on_conflict_values=True):
        """
        This function merges two settings represented as python
        dictionaries into single one.
        By default, this will raise an error if any conflict
        is spotted, but this may be turned off by setting
        "raise_error_on_conflict_values" to False
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
                raise ValueError(f'Conflicting values {proposed_value}'
                                 f' vs {merged_value} for key {key}')

        settings = setting1.copy()
        settings.update(setting2)
        for key, value in settings.items():
            if isinstance(value, dict):
                settings[key] = reduce(lambda x, y: {**x, **y},
                                       (s[key] for s in (setting1, setting2)
                                        if s.get(key)),
                                       dict())

        if raise_error_on_conflict_values:
            for key, value in setting1.items():
                check_for_conflicts(settings[key], key, value)

        return SeriesSchema().dump(settings)

    def subject_preprocess(self, subject, samples,
                           common_subject_settings):
        pass

    def subject_postprocess(self, subject, samples,
                            common_subject_settings):
        pass

    def dataset_preprocess(self, dataset):
        pass

    def dataset_postprocess(self, dataset):
        pass

    @staticmethod
    @abstractmethod
    def sample_result_filename(sample):
        """
        A fixed filename relative to the output directory of
        the current subject for storing results of processing each sample
        generated from f-string pattern with {sample} identifier
        :param sample: filename (most probably sample_json_path)
        relative to the output directory of the current subject
        :return: filename
        """

    @staticmethod
    def filenames_to_skip_sample(sample):
        """
        emergency patterns of the filenames that, if found, will skip
        processing given sample
        :return: list of f-string patterns for filenames relative to
        the output directory of the current subject
        (identifier {sample} will be replaced by the current sample)
        """
        return []

    @staticmethod
    def filenames_to_skip_subject(subject):
        """
        emergency patterns of the filenames that, if found, will skip
        processing given subject
        :return: list of f-string patterns for filenames relative to
        the output directory of the current dataset
        (identifier {subject} will be replaced by the current subject)
        """
        return []

    @staticmethod
    def filenames_to_skip_dataset(dataset):
        """
        emergency patterns of the filenames that, if found, will skip
        processing given dataset
        :return: list of f-string patterns for filenames relative to
        the base output directory.
        (identifier {dataset} will be replaced by the current dataset)
        """
        return []

    @staticmethod
    def sample_filename_prerequisites():
        """
        A pipeline can have a list of filename patterns
        - prerequisites to be able to run the pipeline on.
        :return:
        """
        return []

    @staticmethod
    def dataset_filename_prerequisites():
        """
        A pipeline can have a list of filename patterns
        - prerequisites to be able to run the pipeline on.
        :return:
        """
        return []

    @staticmethod
    def subject_filename_prerequisites():
        """
        A pipeline can have a list of filename patterns
        - prerequisites to be able to run the pipeline on.
        :return:
        """
        return []

    def load_settings(self, working_dir, settings_filename):
        """
        Attempt to load json with settings
        :param working_dir: root directory of the settings file
        :param settings_filename: filename of the settings
        :return: json settings decoded to python dict
        """
        settings_path = join(working_dir, settings_filename)
        try:
            with open(settings_path) as f:
                if self.verbose > 0:
                    print(f'[INFO] importing settings: {settings_path}')
                d = json.load(f)
                return d
        except (EnvironmentError, json.decoder.JSONDecodeError) as e:
            if self.verbose > 0:
                print(f'[ERROR] {e} - defaulting to empty dict')
            return {}

    @abstractmethod
    def sample_layer(self, subject, sample_json_filename, sample_settings):
        """
        The bottom layer of the chain that is called for each sample
        of the given subject.
        Each chain has a specific inner part which is called from process()
        :param subject: name of the subject
        :param sample_json_filename: filename of the sample json (used mostly to
        construct sample-specific paths)
        :param sample_settings: dictionary of settings parsed from merge
        of common subject-specific settings
        and settings read from sample_json_path
        :return:
        """

    def _check_skip_conditions(self, level, input_data, pattern):
        """
        This is an internal method to check if conditions to skip
        the current object of interest are met.

        :param level: only three values are allowed: 'subject', 'sample' and 'dataset'
        :param input_data: refers to input paths/hints
        :param pattern: refers to output paths/result filenames
        :return: True/False whether to skip the current object of interest.
        """
        if level == 'subject':
            skip_candidates = self.filenames_to_skip_subject(pattern)
            prerequisites = self.subject_filename_prerequisites()
        elif level == 'sample':
            skip_candidates = self.filenames_to_skip_sample(pattern)
            prerequisites = self.sample_filename_prerequisites()
        elif level == 'dataset':
            skip_candidates = self.filenames_to_skip_dataset(pattern)
            prerequisites = self.dataset_filename_prerequisites()
        else:
            raise ValueError(f'Invalid level: {level}')

        if any((isfile(filename_to_skip) for filename_to_skip
                in skip_candidates)):
            if self.verbose > 0:
                print(f'[INFO] detected filenames_to_skip_{level}'
                      f'for {level} {input_data} - hence processing '
                      f'the {level} is skipped')
            return True
        if any((not isfile(prerequisite(pattern)) for prerequisite
                in prerequisites)):
            if self.verbose > 0:
                print(f'[WARNING] some prerequisites are not found '
                      f'for the {level} {input_data} - hence '
                      f'processing the {level} is skipped')
                not_found = [p(pattern) for p in prerequisites
                             if not isfile(p(pattern))]
                print(f'[WARNING] not found prerequisites: {not_found}')
            return True
        return False

    def subject_layer(self, subject, samples, subject_settings):
        """
        The middle layer of the chain that is called for each subject.
        First it runs applies abstractmethod self.subject_preprocess (by default no-op),
        then it iterates over naturally sorted samples
        and runs the abstractmethod sample_layer for each of them.
        In the end method runs abstractmethod self.subject_postprocess (by default no-op).

        :param subject: name of the subject (also directory name comprising the subject data)
        :param samples: a list of all subject' inputs (relative paths of jsons)
        :param subject_settings: a dictionary describing the current subject
        based on the information from common.json
        :return: None
        """
        self.subject_preprocess(subject, samples, subject_settings)
        raise_error = self.process_settings.get("raise_error_on_conflict_values",
                                                False)
        for sample_json_filename in natsorted(samples):
            sample_settings = self.load_settings(join(self.base_dir, subject), sample_json_filename)
            output_path_pattern = join(self.results_dir, subject, sample_json_filename)
            if self._check_skip_conditions('sample', sample_json_filename, output_path_pattern):
                continue
            sample_settings = self.merge_data_settings(subject_settings, sample_settings,
                                                       raise_error)
            self.sample_layer(subject, sample_json_filename, sample_settings)
        self.subject_postprocess(subject, samples, subject_settings)

    def dataset_layer(self, dataset):
        """
        The top layer of the chain that is executed for the whole dataset once.
        First it applies abstractmethod self.dataset_preprocess (by default no-op),
        then it iterates over a dataset and for each subject runs the subject_layer
        on each subject' samples followed by the abstractmethod
        self.dataset_postprocess (by default no-op)
        :param dataset: a list of pairs: (subject, samples)
        :return: None
        """

        self.dataset_preprocess(dataset)
        for subject, samples in dataset:
            input_dir = join(self.base_dir, subject)
            output_dir = join(self.results_dir, subject)
            makedirs(output_dir, exist_ok=True)
            if self._check_skip_conditions('subject', subject, output_dir):
                continue
            subject_settings = self.load_settings(input_dir, 'common.json')
            self.subject_layer(subject, samples, subject_settings)
        self.dataset_postprocess(dataset)

    def process(self):
        """
        Central method of the Chain and an entry point to each of the subclasses.
        Not abstract on purpose and is indented to be not modified in subclasses.

        Process consists of two steps:
        1) populates a dataset to be traversed
        2) process the chain of layers in fixed order:
         * dataset_layer
         * subject_layer
         * sample_layer
        :return: None
        """
        dataset = []
        for (dir_path, _, filenames) in walk(self.base_dir):
            jsons = list(filter(lambda x: all([x.endswith('json'),
                                               not (x.startswith('.')),
                                               not (x.endswith('result.json')),
                                               not (x.endswith('common.json'))]),
                                filenames))
            if len(jsons) > 0:
                subject = dir_path[len(self.base_dir):]
                subject_samples = (subject, jsons)
                if self.verbose > 1:
                    print(f'[DETAILS] Added subject-samples pair: {subject_samples}')
                dataset.append(subject_samples)

        if not self._check_skip_conditions('dataset', dataset, self.results_dir):
            self.dataset_layer(dataset)
