import json
from os.path import join

import numpy as np
from scipy.signal import (argrelmax, spectrogram)

from audio_processors import prepare_wav_input
from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from chain import Chain, protect_abc
from pipelines.phoneme import Phoneme


class Formants(Chain):

    requirements = [Phoneme]

    @staticmethod
    def sample_result_filename(sample):
        return f'{sample[:-5]}_formants_result.json'

    @staticmethod
    def filenames_to_skip_sample(sample):
        return [f'{sample[:-5]}_formants_result.csv']

    @staticmethod
    def sample_filename_prerequisites():
        def audio_path(sample):
            return f'{sample[:-5]}_audio.mp4'
        return [audio_path, Phoneme.sample_result_filename]

    _blacklisted_phonemes = ['SIL', '+SPN+', '+NSN+']

    @property
    def blacklisted_phonemes(self):
        return self._blacklisted_phonemes

    @blacklisted_phonemes.setter
    def blacklisted_phonemes(self, new_value):
        if isinstance(new_value, list):
            self._blacklisted_phonemes = new_value
        else:
            raise TypeError(f'blacklisted_phonemes setter expected list,'
                            f' got {type(new_value)}')

    def _compute_formants(self, segments_path, phonemes_result_path,
                          formants_result_path):
        @check_if_already_done(formants_result_path, self.verbose,
                               lambda bool_val: bool_val)
        def recognize_formants(segments_path, phonemes_result_path,
                               formants_result_path):
            if self.verbose > 0:
                print(f'[INFO] segments_path: {segments_path}')
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            schema = PhonemesSchema()
            with open(phonemes_result_path, 'r') as f:
                print(f' phonemes_result_path: {phonemes_result_path}')
                json_file = json.load(f)
                phonemes_result = schema.load(json_file)
                phonemes_info = [info for info in phonemes_result['info']
                                 if info['word'] not in self.blacklisted_phonemes]
                maximum_len = 4096
                formants_result = []
                for info in phonemes_info:
                    start, stop = (1000 * info['start'],
                                   1000 * info['end'])
                    segment = np.array(wav[start:stop].get_array_of_samples())
                    freq, t, Sxx = spectrogram(segment, frequency,
                                               window=('kaiser', 4.0),
                                               nperseg=min(maximum_len,
                                                           len(segment)),
                                               noverlap=1)
                    for i in range(len(t)):
                        n = 5
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

    def sample_layer(self, subject, sample_json_filename, settings):
        datatype = settings.get('datatype')
        assert(datatype is not None)
        segments = settings.get('segments', [{'start': 'begin', 'stop': 'end'}])

        # TODO Currently, that's only audio extension we support - better check now than later:
        assert datatype == 'mp4'
        prerequisites = self.sample_filename_prerequisites()
        output_path_pattern = join(self.results_dir, subject, sample_json_filename)

        audio_path, phonemes_path = (prerequisites[i](output_path_pattern) for i in range(2))
        if self.verbose > 0:
            print(f'[INFO] audio_path: {audio_path}, phonemes_path: {phonemes_path}')
        _, segments_path = prepare_wav_input(audio_path, datatype, segments,
                                             self.verbose, use_original_frequency=True)
        self._compute_target(segments_path, phonemes_path, output_path_pattern)

    def _compute_target(self, segments_path, phonemes_path, series_json_path):
        formants_result_path = self.sample_result_filename(series_json_path)
        self._compute_formants(segments_path, phonemes_path, formants_result_path)
        if self.verbose > 0:
            print(f'[INFO] formants result path: {formants_result_path}')
