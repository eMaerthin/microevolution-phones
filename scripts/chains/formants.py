import json
import logging
import os

import numpy as np
from scipy.signal import (argrelmax, spectrogram)

from chains.labeled_base import DEFAULT_BLACKLISTED_LABELS, LabeledBase
from chains.phoneme import Phoneme
from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *

logger = logging.getLogger()


class Formants(LabeledBase):

    allow_sample_layer_concurrency = True
    abstract_class = False
    requirements = [Phoneme]

    @staticmethod
    def sample_result_filename(out_sample_path):
        filename, _ = os.path.splitext(out_sample_path)
        return f'{filename}_formants_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        filename, _ = os.path.splitext(out_sample_path)
        return [f'{filename}_formants_result.csv']

    @staticmethod
    def sample_filename_prerequisites():
        return [Phoneme.sample_result_filename]

    _blacklisted = DEFAULT_BLACKLISTED_LABELS

    def _compute_formants(self, segments_path, phonemes_result_path,
                          formants_result_path):
        formant_maximum_len = self.process_settings.get('formant_maximum_len',
                                                        4096)
        spectrogram_window = self.process_settings.get('spectrogram_window',
                                                       ('kaiser', 4.0))
        spectrogram_n = self.process_settings.get('spectrogram_n', 5)

        @check_if_already_done(formants_result_path,
                               validator=lambda bool_val: bool_val)
        def recognize_formants(segments_path, phonemes_result_path,
                               formants_result_path):
            logger.info(f'segments_path: {segments_path}')
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            schema = DecoderOutputSchema()
            with open(phonemes_result_path, 'r') as f:
                json_file = json.load(f)
                labels_result = schema.load(json_file)
                labels_info = [info for info in labels_result['segment_info']
                               if info['word'] not in self.blacklisted_labels]
                formants_result = []
                for info in labels_info:
                    start, stop = (1000 * info['start'],
                                   1000 * info['end'])
                    segment = np.array(wav[start:stop].get_array_of_samples())
                    freq, t, Sxx = spectrogram(segment, frequency,
                                               window=spectrogram_window,
                                               nperseg=min(formant_maximum_len,
                                                           len(segment)),
                                               noverlap=1)
                    for i in range(len(t)):
                        n = spectrogram_n
                        ith_spectrogram = Sxx[:, i]
                        spectrogram_normalized = ith_spectrogram/sum(ith_spectrogram)
                        local_maxima = argrelmax(ith_spectrogram)[0]
                        n_largest_local_max_f_idx = local_maxima[ith_spectrogram[local_maxima].argsort()[-n:][::-1]]
                        n_largest_local_max_f = freq[n_largest_local_max_f_idx]
                        n_largest_local_max_s = spectrogram_normalized[n_largest_local_max_f_idx]
                        formant_result = {'t': t[i], 'i': i,
                                          'len_t': len(t), 'len_freq': len(freq),
                                          'freq_delta': freq[1] - freq[0], **info,
                                          'max_f': freq[np.argmax(Sxx[:, i])], 'N': n,
                                          'N_largest_local_max_f': n_largest_local_max_f,
                                          'N_largest_local_max_s': n_largest_local_max_s}
                        formants_result.append(formant_result)
                formants = PhonemesFormantsSchema()
                formants_dict = {'formants_info': formants_result}
                result = formants.dumps(formants_dict)
                with open(formants_result_path, 'w') as f:
                    f.write(result)
                    return True
        recognize_formants(segments_path, phonemes_result_path, formants_result_path)

    def compute_target(self, segments_path, phonemes_path, output_path_pattern):
        """
        Convenient method to reuse sample_layer with other similar chains
        i.e. only compute_target can be overridden.
        :param segments_path: absolute path to segments wav (as pointed by the sample json)
        :param phonemes_path: json with phonemes details (from chain Phoneme)
        :param output_path_pattern: pattern to produce output path
        :return: None
        """
        formants_result_path = self.sample_result_filename(output_path_pattern)
        self._compute_formants(segments_path, phonemes_path, formants_result_path)
        logger.info(f'formants result path: {formants_result_path}')
