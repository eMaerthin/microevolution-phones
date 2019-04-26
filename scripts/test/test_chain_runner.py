import pytest

import chain_runner


class TestChainRunner(object):
    class Axyz(object):
        pass

    class Bxyz(object):
        pass

    class Cde(object):
        pass

    class Xy(object):
        pass

    class Xyz(object):
        pass

    chains = [Axyz, Bxyz, Cde, Xy, Xyz]
    filter_set = {'yz', 'C'}
    filter_function = 'include'
    invalid_filter_function = 'foobar'
    empty_filter_set = {}
    empty_chains = []
    filter_exclude = 'exclude'

    def test_invalid_filter_function_in_filter_chains(self):
        with pytest.raises(ValueError):
            chain_runner.ChainRunner.filter_chains(self.chains,
                                                   self.filter_set,
                                                   self.invalid_filter_function)

    def test_include_in_filter_chains(self):
        expected_chains = self.chains.copy()
        expected_chains.remove(self.Xy)
        chains = chain_runner.ChainRunner.filter_chains(self.chains,
                                                        self.filter_set,
                                                        self.filter_function)
        assert expected_chains == chains

    def test_exclude_in_filter_chains(self):
        expected_chains = [self.Xy]
        chains = chain_runner.ChainRunner.filter_chains(self.chains,
                                                        self.filter_set,
                                                        self.filter_exclude)
        assert expected_chains == chains

    def test_empty_filter_set(self):
        expected_chains = self.chains.copy()
        chains = chain_runner.ChainRunner.filter_chains(self.chains,
                                                        self.empty_filter_set,
                                                        self.filter_function)
        assert expected_chains == chains

    def test_empty_chains(self):
        expected_chains = self.empty_chains.copy()
        chains = chain_runner.ChainRunner.filter_chains(self.empty_chains,
                                                        self.filter_set,
                                                        self.filter_function)
        assert chains == expected_chains
