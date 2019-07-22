import json
import logging

import numpy as np
from python_speech_features import mfcc

from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from chains.formants import Formants
logger = logging.getLogger()


class MfccLocal(Formants):
    """
    MfccLocal computes Mfcc features for each phoneme from the sample
    that are not blacklisted based on phoneme label that is
    received from Phoneme chain.

    It subclasses Formants to not repeat the sample_layer logic
    which is valid also in this context
    """

    allow_sample_layer_concurrency = True

    @staticmethod
    def sample_result_filename(out_sample_path):
        return f'{out_sample_path[:-5]}_mfcc_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        return [f'{out_sample_path[:-5]}_mfcc_result.csv']

    def _compute_mfcc(self, segments_path, phonemes_result_path, mfcc_result_path):
        phoneme_len = self.process_settings.get("phoneme_len", 2048)
        ignore_shorter_phonemes = self.process_settings.get("ignore_shorter_phonemes", True)
        mfcc_nfft = self.process_settings.get("mfcc_nfft", 2048)
        mfcc_winstep = self.process_settings.get("mfcc_winstep", 0.1)

        @check_if_already_done(mfcc_result_path, lambda x: x)
        def store_mfcc(segments_path, phonemes_result_path, mfcc_result_path):
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            schema = DecoderOutputSchema()
            with open(phonemes_result_path, 'r') as f:
                json_file = json.load(f)
                phonemes_result = schema.load(json_file)
                phonemes_info = [info for info in phonemes_result['segment_info']
                                 if info['word'] not in self.blacklisted_phonemes]
                mfcc_result = []
                for info in phonemes_info:
                    start, stop = (1000 * info['start'], 1000 * info['end'])
                    segment = np.array(wav[start:stop].get_array_of_samples())
                    if ignore_shorter_phonemes and segment.size < phoneme_len:
                        continue
                    mfcc_features = mfcc(segment, samplerate=frequency,
                                         nfft=mfcc_nfft, winstep=mfcc_winstep)
                    for i in range(len(mfcc_features)):
                        ith_mfcc = np.array(mfcc_features[i, :])
                        ith_mfcc_result_row = {'i': i, 'length': len(mfcc_features),
                                               'mfcc': ith_mfcc, **info}
                        mfcc_result.append(ith_mfcc_result_row)
                mfcc_schema = MfccLocalSchema()
                mfcc_dict = {'mfcc_info': mfcc_result}
                result = mfcc_schema.dumps(mfcc_dict)
                with open(mfcc_result_path, 'w') as result_f:
                    result_f.write(result)
                    return True
        store_mfcc(segments_path, phonemes_result_path, mfcc_result_path)

    def compute_target(self, segments_path, phonemes_path, output_path_pattern):
        mfcc_result_path = self.sample_result_filename(output_path_pattern)
        self._compute_mfcc(segments_path, phonemes_path, mfcc_result_path)
        logger.info(f'mfcc result path: {mfcc_result_path}')
