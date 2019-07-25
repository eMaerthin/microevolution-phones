import logging

import numpy as np
from python_speech_features import mfcc

from format_converters import get_segment
from schemas import *
from chains.mfcc import Mfcc
logger = logging.getLogger()


class MfccGlobal(Mfcc):
    """
    MfccGlobal computes Mfcc features for the whole sample
    (based on segments wav) - i.e. without using information
    about phoneme label that can be received from Phoneme chain

    It subclasses Formants to not repeat the sample_layer logic
    which is valid also in this context
    """

    abstract_class = False

    @staticmethod
    def sample_result_filename(out_sample_path):
        return f'{out_sample_path[:-5]}_mfcc_global_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        return [f'{out_sample_path[:-5]}_mfcc_global_result.csv']

    @staticmethod
    def serialize_to_json(mfcc_result):
        """
        :param mfcc_result: list of mfcc measurements with
        necessary metadata
        :return: serialized object of proper schema
        """
        mfcc_global_schema = MfccGlobalSchema()
        mfcc_dict = {'mfcc_global_info': mfcc_result}
        return mfcc_global_schema.dumps(mfcc_dict)

    def compute_mfcc(self, segments_path, phonemes_result_path):
        """

        :param segments_path: path to the input wav
        :param phonemes_result_path: path to phonemes results
        that is required by the Local version of the Mfcc
        :return: computed list of mfcc features with all required metadata
        """
        mfcc_nfft = self.process_settings.get("mfcc_nfft", 2048)
        mfcc_winstep = self.process_settings.get("mfcc_winstep", 0.1)
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
        return mfcc_result