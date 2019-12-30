from abc import abstractmethod
import logging
from os.path import (dirname, join)

from audio_processors import audio_and_segment_paths
from chain import Chain
from chains.phoneme import Phoneme
from chains.words import Words
from schemas import *

logger = logging.getLogger()

DEFAULT_BLACKLISTED_LABELS = ['SIL', '+SPN+', '+NSN+']

class LabeledBase(Chain):

    allow_sample_layer_concurrency = True
    abstract_class = True
    requirements = []
    _blacklisted = []

    def sample_filename_prerequisites(self):
        if self.process_settings.get("use_words_instead_of_phonemes", False):
            return [Words.sample_result_filename]
        return [Phoneme.sample_result_filename]

    @property
    def blacklisted_labels(self):
        return self._blacklisted

    @blacklisted_labels.setter
    def blacklisted_labels(self, new_value):
        if isinstance(new_value, list):
            self._blacklisted = new_value
        else:
            raise TypeError(f'blacklisted_labels setter expected list,'
                            f' got {type(new_value)}')

    def sample_layer(self, subject, sample_json_filename, sample_settings):

        def mp4_path(sample):
            return f'{sample[:-5]}_audio.mp4'
        url = sample_settings.get('url')
        datatype = sample_settings.get('datatype')
        assert(datatype is not None)

        output_path_pattern = join(self.results_dir, subject, sample_json_filename)

        if url.startswith('http'):
            assert datatype == 'mp4', 'Currently that\'s the only audio extension ' \
                                      'we support for http-based url ' \
                                      '- better check now than later!'
            audio_path = mp4_path(output_path_pattern)
        elif url.endswith(datatype):
            audio_path = join(dirname(output_path_pattern), url)
        else:
            raise ValueError(f'unhandled url: {url} (settings: {sample_settings})')
        if not exists(audio_path):
            raise FileNotFoundError(f'File not found {audio_path}')

        prerequisites = self.sample_filename_prerequisites()

        labels_path = prerequisites[0](output_path_pattern)
        logger.info(f'audio_path: {audio_path}, labels_path: {labels_path}')
        original_freq = self.process_settings.get('original_freq', True)
        _, segments_path = audio_and_segment_paths(audio_path, original_freq)
        self.compute_target(segments_path, labels_path, output_path_pattern)

    @abstractmethod
    def compute_target(self, segments_path, labels_path, output_path_pattern):
        """
        Convenient method to reuse sample_layer with other similar chains
        i.e. only compute_target can be overridden.
        :param segments_path: absolute path to segments wav (as pointed by the sample json)
        :param labels_path: json with labels details (either coming from chain Phoneme or Words or anything else)
        :param output_path_pattern: pattern to produce output path
        :return: None
        """
        pass