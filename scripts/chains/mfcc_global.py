import logging

import numpy as np
from python_speech_features import mfcc

from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from chains.formants import Formants
logger = logging.getLogger()


class MfccGlobal(Formants):
    """
    MfccGlobal computes Mfcc features for the whole sample
    (based on segments wav) - i.e. without using information
    about phoneme label that can be received from Phoneme chain

    It subclasses Formants to not repeat the sample_layer logic
    which is valid also in this context
    """

    allow_sample_layer_concurrency = True

    @staticmethod
    def sample_result_filename(out_sample_path):
        return f'{out_sample_path[:-5]}_mfcc_global_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        return [f'{out_sample_path[:-5]}_mfcc_global_result.csv']

    def _compute_global_mfcc(self, segments_path, mfcc_global_result_path):
        mfcc_nfft = self.process_settings.get("mfcc_nfft", 2048)
        mfcc_winstep = self.process_settings.get("mfcc_winstep", 0.1)

        @check_if_already_done(mfcc_global_result_path, lambda x: x)
        def store_global_mfcc(segments_path, mfcc_global_result_path):
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            mfcc_result = []
            array_of_samples = np.array(wav.get_array_of_samples())
            mfcc_features = mfcc(array_of_samples, samplerate=frequency,
                                 nfft=mfcc_nfft, winstep=mfcc_winstep)
            for i in range(len(mfcc_features)):
                ith_mfcc = np.array(mfcc_features[i, :])
                ith_mfcc_result_row = {'i': i, 'mfcc': ith_mfcc}
                mfcc_result.append(ith_mfcc_result_row)
            mfcc_global_schema = MfccGlobalSchema()
            mfcc_dict = {'mfcc_global_info': mfcc_result}
            result = mfcc_global_schema.dumps(mfcc_dict)
            with open(mfcc_global_result_path, 'w') as result_f:
                result_f.write(result)
                return True

        store_global_mfcc(segments_path, mfcc_global_result_path)

    def compute_target(self, segments_path, _, output_path_pattern):
        mfcc_global_result_path = self.sample_result_filename(output_path_pattern)
        self._compute_global_mfcc(segments_path, mfcc_global_result_path)
        logger.info(f'mfcc_global result path: {mfcc_global_result_path}')
