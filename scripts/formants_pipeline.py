import json

import numpy as np
from scipy.signal import (argrelmax, spectrogram)

from audio_processors import download_youtube_url, prepare_wav_input
from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from pipeline import Pipeline
from phoneme_pipeline import PhonemePipeline


class FormantsPipeline(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_formants_result.json'

    @staticmethod
    def filename_prerequisites():
        def audio_path(json_path):
            return f'{json_path[:-5]}_audio.mp4'
        return [audio_path, PhonemePipeline.result_filename]

    _blacklisted_phonemes = ['SIL']

    @property
    def blacklisted_phonemes(self):
        return self._blacklisted_phonemes

    @blacklisted_phonemes.setter
    def blacklisted_phonemes(self, new_value):
        if isinstance(new_value, list):
            self._blacklisted_phonemes = new_value
        else:
            raise ValueError('Wrong parameter for blacklisted_phonemes setter')

    def compute_formants(self, segments_path, phonemes_result_path, formants_result_path):
        @check_if_already_done(formants_result_path, self.verbose, lambda x: x)
        def recognize_formants(segments_path, phonemes_result_path, formants_result_path):
            print(segments_path)
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            schema = PhonemesSchema()
            with open(phonemes_result_path, 'r') as f:
                print(f' phonemes_result_path: {phonemes_result_path}')
                json_file = json.load(f)
                phonemes_result = schema.load(json_file)
                phonemes_info = [info for info in phonemes_result['info']
                                 if info['word'] not in self.blacklisted_phonemes]
                # phonemes_info = phonemes_info[:100]
                ms_markers = [(1000 * p['start'], 1000 * p['end']) for p in phonemes_info]
                scattered_segments = [np.array(wav[start:stop].get_array_of_samples()) for (start,stop) in ms_markers]
                maximum_len = 8192
                spectrograms = [spectrogram(segment, frequency, window=('kaiser', 4.0),
                                            nperseg=min(maximum_len, len(segment)), noverlap=1)
                                for segment in scattered_segments]
                formants_result = []
                for (freq, t, Sxx), info in zip(spectrograms, phonemes_info):
                    for i in range(len(t)):
                        n = 5
                        ith_spectrogram = Sxx[:, i]
                        local_maxima = argrelmax(ith_spectrogram)[0]
                        n_largest_local_max_f_idx = local_maxima[ith_spectrogram[local_maxima].argsort()[-n:][::-1]]
                        n_largest_local_max_f = freq[n_largest_local_max_f_idx]
                        n_largest_local_max_s = ith_spectrogram[n_largest_local_max_f_idx]
                        formant_result = {'t': t[i], 'i': i, 'len_t': len(t), 'len_freq': len(freq),
                                          'freq_delta': freq[1] - freq[0], 'info': info,
                                          'max_f': freq[np.argmax(Sxx[:, i])], 'N': n,
                                          'N_largest_local_max_f': n_largest_local_max_f,
                                          'N_largest_local_max_s': n_largest_local_max_s}
                        formants_result.append(formant_result)
                formants = PhonesFormantsSchema()
                formants_dict = {'formants_info': formants_result}
                result = formants.dumps(formants_dict)
                with open(formants_result_path, 'w') as f:
                    f.write(result)
                    return True
            return False
        recognize_formants(segments_path, phonemes_result_path, formants_result_path)

    def pipeline(self, series_json_path, series_settings):
        datatype = series_settings.get('datatype')
        assert(datatype is not None)
        segments = series_settings.get('segments', [{'start': 'begin', 'stop': 'end'}])

        # TODO Currently, that's only audio extension we support - better check now than later:
        assert datatype == 'mp4'
        prerequisites = self.filename_prerequisites()
        audio_path, phonemes_path = (prerequisites[i](series_json_path) for i in range(2))
        if self.verbose > 0:
            print(f'audio_path: {audio_path}, phonemes_path: {phonemes_path}')
        wav_path, segments_path = prepare_wav_input(audio_path, datatype, segments,
                                                    self.verbose, use_original_frequency=True)
        formants_result_path = self.result_filename(series_json_path)
        self.compute_formants(segments_path, phonemes_path, formants_result_path)
        if self.verbose > 0:
            print(f'formants result path: {formants_result_path}')
