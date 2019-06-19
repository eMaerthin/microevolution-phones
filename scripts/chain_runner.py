import json
from os import makedirs
from os.path import join
from pathlib import Path
from shutil import copy

from chain import Chain
import chains as this_line_is_necessary_to_register_chains
from schemas import ChainRunnerSettingsSchema

# TODO(marcin): replace "verbose" with https://realpython.com/python-logging/

# TODO(marcin): add a chain that translate the sound to words and selects segments of given words ONLY
# TODO(marcin): implement visualization of Paths (connected points representing tsne on the pointclouds
#  within given timeframe (for instance dense pointclouds from each minute of the same sample)

class ChainRunner(object):

    @classmethod
    def from_experiment_config(cls, experiment_config_path):
        schema = ChainRunnerSettingsSchema()
        with open(experiment_config_path, 'r') as f_csv:
            json_file = json.load(f_csv)
            result = schema.load(json_file)
            result_dir = join(result['dataset_home_dir'], '_results',
                              result['results_identifier'])
            makedirs(result_dir, exist_ok=True)
            copy(experiment_config_path,
                 join(result_dir, Path(experiment_config_path).name))
            return cls(**result)

    def __init__(self, dataset_home_dir='../subjects/',
                 process_settings='', results_identifier='',
                 subjects_pattern=None, verbose=1):
        self._dataset_home_dir = dataset_home_dir
        self._process_settings = process_settings
        self._results_identifier = results_identifier
        self._subjects_pattern = subjects_pattern
        self._verbose = verbose

    @property
    def dataset_home_dir(self):
        """
        read-only property that points to the dataset_home_dir
        :return:
        """
        return self._dataset_home_dir

    @property
    def process_settings(self):
        """
        read-only property that returns dictionary with all chain-custom properties
        :return:
        """
        return self._process_settings

    @property
    def results_identifier(self):
        """
        read-only property that points to the results directory name
        :return:
        """
        return self._results_identifier

    @property
    def subjects_pattern(self):
        """
        read-only property that can be used for filtering subjects.
        Inactive by default (set to None)
        :return:
        """
        return self._subjects_pattern

    @property
    def verbose(self):
        """
        read-only property that controls verbosity
        :return:
        """
        return self._verbose

    def process_chain(self, chain_name='Phoneme'):
        """
        One of two possible way of processing the data.
        Here just a single chain (chain_name) is launched (if chain_name is registered).
        :param chain_name: a chain name to be processed
        :return: None
        """
        chain = Chain.subclasses.get(chain_name, None)
        try:
            if self.verbose > 0:
                print(f'running chain {chain.__name__}')
            chain.initialize_from_parameters(self.dataset_home_dir, self.process_settings,
                                             self.results_identifier, self.subjects_pattern,
                                             self.verbose).process()
        except AttributeError as e:
            if self.verbose > 0:
                print(f'[ERROR] chain_name {chain_name} not found. {e}')

    @staticmethod
    def filter_chains(chains, filter_set, filter_function):
        """
        Filters chains by applying include/exclude filter
        :param chains: input chains to be filtered
        :param filter_set: a set of pairs chain patterns (so incomplete names are allowed).
        If set is empty, process all available chains.
        :param filter_function: two options are allowed.
        'include' -> process only mentioned chains,
        'exclude' -> process all but mentioned chains
        :return:
        """
        if not isinstance(filter_set, set):
            return chains
        if len(filter_set) == 0:
            return chains

        if filter_function == 'include':
            return [chain for chain in chains if
                    any({element in chain.__name__ for element in filter_set})]
        elif filter_function == 'exclude':
            return [chain for chain in chains if
                    all({element not in chain.__name__ for element in filter_set})]
        else:
            raise ValueError(f'filter_function should be either "include" or "exclude", '
                             f'was {filter_function}')

    def process_chains(self, filter_set=None, filter_function='include'):
        """
        One of two possible way of processing the data.
        Here all registered chains are processed that are not filtered out
        by filter initialized with filter_set and filter_function parameters.
        :param filter_set: a set of pairs chain patterns (so incomplete names are allowed).
        If set is empty or None, process all registered Chain subclasses (chains).
        :param filter_function: two options are allowed.
        'include' -> process only mentioned chains,
        'exclude' -> process all but mentioned chains
        :return: None
        """
        try:
            chains = self.filter_chains(Chain.ordered_subclasses, filter_set,
                                        filter_function)
            for chain in chains:
                self.process_chain(chain.__name__)
        except ValueError as e:
            if self.verbose > 0:
                print(f'[ERROR] {e}')

    def process_mfcc_chains(self):
        self.process_chains({'Mfcc'})
