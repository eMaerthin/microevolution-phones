from abc import ABCMeta, abstractmethod
from functools import reduce
import json
import logging
from multiprocessing import cpu_count, Pool
from os import (makedirs, walk)
from os.path import (exists, isfile, join)

from natsort import natsorted
from toposort import toposort_flatten

from schemas import SampleSchema
logger = logging.getLogger()


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


class Chain(metaclass=protect_abc("load_settings", "merge_data_settings",
                                  "dataset_layer", "subject_layer")):
    """
    Base class for concrete chain processes like Phoneme or Spectrogram.
    It stores a dictionary of subclasses class names and class references
    and in addition it stores a topologically ordered list of subclasses
    sorted with their dependencies, i.e. if for instance subclass A depends
    on nothing, subclass B depends on A and subclass C depends on both A and B,
    then ordered list will be [A,B,C]

    Chain has few read-write properties:
     - base_dir (sets parent directory where subjects from the dataset are placed)
     - process_settings (this adjust how to process the chain)
     - results_identifier (stores name of the directory comprising experiment's results)
     - subjects_pattern (if set, store patterns used to filter subjects)

     and a read-only property results_dir that holds absolute path of the experiment's results


    In addition there is one factory classmethod called initialize_from_parameters
    that initializes Chain with base_dir and results_identifier.

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

    allow_sample_layer_concurrency = False
    abstract_class = True
    ordered_subclasses = []
    requirements = []
    s = {}
    subclasses = {}
    subjects_pattern = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.abstract_class:
            cls.subclasses[cls.__name__] = cls
            chains_dependencies = {value: set(value.requirements)
                                   for value in cls.subclasses.values()}
            cls.ordered_subclasses[:] = list(toposort_flatten(chains_dependencies,
                                                              sort=False))

    def __init__(self):
        self._base_dir = ''
        self._process_settings = {'raise_error_on_conflict_values': True}
        self._results_identifier = ''

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

    @property
    def process_settings(self):
        return self._process_settings

    @process_settings.setter
    def process_settings(self, new_value):
        if new_value is None:
            new_value = dict()
        if not isinstance(new_value, dict):
            raise TypeError('expected argument of type dict')
        self._process_settings = new_value

    @property
    def results_dir(self):
        return join(self.base_dir, '_results', self.results_identifier)

    @property
    def results_identifier(self):
        return self._results_identifier

    @results_identifier.setter
    def results_identifier(self, new_value):
        self._results_identifier = str(new_value)

    @classmethod
    def initialize_from_parameters(cls, base_dir, process_settings,
                                   results_identifier, subjects_pattern):
        """
        Initializes class with given base_dir value
        :param base_dir: An absolute path to base directory
        :param process_settings: dictionary with settings
        :param results_identifier: name for intermediate folder
        comprising experiment' results
        :param subjects_pattern: if set, will be used to filter subjects
        :return:
        """
        pipeline = cls()
        pipeline.base_dir = base_dir
        pipeline.process_settings = process_settings
        pipeline.results_identifier = results_identifier
        pipeline.subjects_pattern = subjects_pattern
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

        return SampleSchema().dump(settings)

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
    def sample_result_filename(out_sample_path):
        """
        A fixed filename relative to the output directory of
        the current subject for storing results of processing each sample
        generated from f-string pattern with {sample} identifier
        :param out_sample_path: filename (most probably sample_json_path)
        relative to the output directory of the current subject
        :return: filename
        """

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        """
        emergency patterns of the filenames that, if found, will skip
        processing given sample
        :return: list of f-string patterns for filenames relative to
        the output directory of the current subject
        (identifier {sample} will be replaced by the current sample)
        """
        return []

    @staticmethod
    def filenames_to_skip_subject(out_subject_dir):
        """
        emergency patterns of the filenames that, if found, will skip
        processing given subject
        :return: list of f-string patterns for filenames relative to
        the output directory of the current dataset
        (identifier {subject} will be replaced by the current subject)
        """
        return []

    @staticmethod
    def filenames_to_skip_dataset(out_results_dir):
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
                logger.info(f'Importing settings: {settings_path}')
                d = json.load(f)
                return d
        except (EnvironmentError, json.decoder.JSONDecodeError) as e:
            logger.error(f'{e} - defaulting to empty dict')
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

    def _check_skip_conditions(self, level, pattern):
        """
        This is an internal method to check if conditions to skip
        the current object of interest are met.

        :param level: only three values are allowed: 'subject', 'sample' and 'dataset'
        :param pattern: refers to output paths/result filenames
        :return: True/False whether to skip the current object of interest.
        """
        skip_candidates = {'sample': self.filenames_to_skip_sample,
                           'subject': self.filenames_to_skip_subject,
                           'dataset': self.filenames_to_skip_dataset}

        prerequisites = {'sample': self.sample_filename_prerequisites,
                         'subject': self.subject_filename_prerequisites,
                         'dataset': self.dataset_filename_prerequisites}

        if any((isfile(filename_to_skip) for filename_to_skip
                in skip_candidates[level](pattern))):
            logger.info(f'Detected filenames_to_skip_{level} '
                        f'for {pattern} - hence processing '
                        f'the {level} is skipped')
            found = [skipper for skipper in skip_candidates[level](pattern)
                     if isfile(skipper)]
            logger.info(f'Found skippers: {found}')
            return True

        if any((not isfile(prerequisite(pattern)) for prerequisite
                in prerequisites[level]())):
            logger.warning(f'Some prerequisites are not found '
                           f'for the {pattern} - hence '
                           f'processing the {level} is skipped')
            not_found = [p(pattern) for p in prerequisites[level]()
                         if not isfile(p(pattern))]
            logger.warning(f'Not found prerequisites: {not_found}')
            return True

        return False

    def wrap_sample_layer(self, args):
        (sample_json_filename, subject, subject_settings) = args
        sample_settings = self.load_settings(join(self.base_dir, subject), sample_json_filename)
        output_path_pattern = join(self.results_dir, subject, sample_json_filename)
        if self._check_skip_conditions('sample', output_path_pattern):
            return
        raise_error = self.process_settings.get("raise_error_on_conflict_values",
                                                False)
        sample_settings = self.merge_data_settings(subject_settings, sample_settings,
                                                   raise_error)
        self.sample_layer(subject, sample_json_filename, sample_settings)

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
        output_subject_dir = join(self.results_dir, subject)
        makedirs(output_subject_dir, exist_ok=True)

        if not self._check_skip_conditions('subject', output_subject_dir):
            if self.allow_sample_layer_concurrency:
                p = Pool(cpu_count())
                p.map(self.wrap_sample_layer, [(sample_json_filename, subject, subject_settings) for sample_json_filename in samples])
                p.close()
                p.join()
            else:
                [self.wrap_sample_layer((sample_json_filename, subject, subject_settings)) for sample_json_filename in samples]
                #
                # sample_settings = self.load_settings(join(self.base_dir, subject), sample_json_filename)
                # output_path_pattern = join(self.results_dir, subject, sample_json_filename)
                # if self._check_skip_conditions('sample', output_path_pattern):
                #     return
                # raise_error = self.process_settings.get("raise_error_on_conflict_values",
                #                                         False)
                # sample_settings = self.merge_data_settings(subject_settings, sample_settings,
                #                                            raise_error)
                # self.sample_layer(subject, sample_json_filename, sample_settings)

            #    Process(target=self.wrap_sample_layer, args=(sample_json_filename, subject, subject_settings, lock)).start()
            # with Executor(max_workers=4) as exe:
            #     jobs = [exe.submit(self.wrap_sample_layer, sample_json_filename, subject, subject_settings, lock)
            #             ]
            #     [job.result() for job in jobs]
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
        if not self._check_skip_conditions('dataset', self.results_dir):
            for subject, samples in dataset:
                if not self.subjects_pattern or any(pattern in subject for pattern in self.subjects_pattern):
                    input_dir = join(self.base_dir, subject)
                    subject_settings = self.load_settings(input_dir, 'common.json')
                    self.subject_layer(subject, samples, subject_settings)
                logger.info(f'Subject {subject} discarded due to subjects_pattern {self.subjects_pattern}')
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
        excluded = ['_results', '_configs']
        for (dir_path, dirs, filenames) in walk(self.base_dir,
                                               topdown=True):
            dirs[:] = [d for d in dirs if d not in excluded]
            jsons = list(filter(lambda x: all([x.endswith('json'),
                                               not (x.startswith('.')),
                                               not (x.endswith('result.json')),
                                               not (x.endswith('common.json'))]),
                                filenames))
            if len(jsons) > 0:
                subject = dir_path[len(self.base_dir):]
                samples = natsorted(jsons)
                if 'limit_recordings_per_subject' in self._process_settings:
                    samples = samples[:self._process_settings['limit_recordings_per_subject']]
                subject_samples = (subject, samples)
                logger.debug(f'Added subject-samples pair: {subject_samples}')
                dataset.append(subject_samples)

        if self.sample_result_filename('').endswith('.json'):
            assert self.sample_result_filename('').endswith('result.json'),\
                'sample_result_filename should end with result.json. '\
                f'is: {self.sample_result_filename("")}'

        self.dataset_layer(dataset)
