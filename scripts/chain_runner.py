from chain import Chain
import pipelines # required to register chains


class ChainRunner(object):

    def __init__(self, dataset_home_dir='../subjects/', verbose=1):
        self._dataset_home_dir = dataset_home_dir
        self._verbose = verbose

    @property
    def verbose(self):
        """
        read-only property that controls verbosity
        :return:
        """
        return self._verbose

    @property
    def dataset_home_dir(self):
        """
        read-only property that points to the dataset_home_dir
        :return:
        """
        return self._dataset_home_dir

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
            chain.from_verbose_and_base_dir(self.verbose, self.dataset_home_dir).process()
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

    def process_chains(self, filter_set={}, filter_function='include'):
        """
        One of two possible way of processing the data.
        Here all registered chains are processed that are not filtered out
        by filter initialized with filter_set and filter_function parameters.
        :param filter_set: a set of pairs chain patterns (so incomplete names are allowed).
        If set is empty, process all available chains.
        :param filter_function: two options are allowed.
        'include' -> process only mentioned chains,
        'exclude' -> process all but mentioned chains
        :return: None
        """
        try:
            chains = self.filter_chains(Chain.ordered_subclasses, filter_set, filter_function)
            for chain in chains:
                if self.verbose > 0:
                    print(f'[INFO] running chain {chain.__name__}')
                chain.from_verbose_and_base_dir(self.verbose, self.dataset_home_dir).process()
        except ValueError as e:
            if self.verbose > 0:
                print(f'[ERROR] {e}')

    def process_mfcc_chains(self):
        self.process_chains({'Mfcc'})
