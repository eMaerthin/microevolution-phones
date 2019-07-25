import json
import logging

import numpy as np
from python_speech_features import mfcc

from format_converters import get_segment
from schemas import *
from chains.mfcc import Mfcc
logger = logging.getLogger()


class MfccLocal(Mfcc):
    """
    MfccLocal computes Mfcc features for each phoneme from the sample
    that are not blacklisted based on phoneme label that is
    received from Phoneme chain.

    It subclasses Formants to not repeat the sample_layer logic
    which is valid also in this context
    """

    abstract_class = False

    @staticmethod
    def sample_result_filename(out_sample_path):
        return f'{out_sample_path[:-5]}_mfcc_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        return [f'{out_sample_path[:-5]}_mfcc_result.csv']

    @staticmethod
    def serialize_to_json(mfcc_result):
        """
        :param mfcc_result: list of mfcc measurements with
        necessary metadata
        :return: serialized object of proper schema
        """
        mfcc_schema = MfccLocalSchema()
        mfcc_dict = {'mfcc_info': mfcc_result}
        return mfcc_schema.dumps(mfcc_dict)

    def compute_mfcc(self, segments_path, phonemes_result_path):
        """

        :param segments_path: path to the input wav
        :param phonemes_result_path: path to phonemes results
        that is required by the Local version of the Mfcc
        :return: computed list of mfcc features with all required metadata
        """
        wav = get_segment(segments_path, 'wav')
        frequency = wav.frame_rate
        phoneme_len = self.process_settings.get("phoneme_len", 2048)
        ignore_shorter_phonemes = self.process_settings.get("ignore_shorter_phonemes", True)
        mfcc_nfft = self.process_settings.get("mfcc_nfft", 2048)
        mfcc_winstep = self.process_settings.get("mfcc_winstep", 0.1)

        with open(phonemes_result_path, 'r') as f:
            schema = DecoderOutputSchema()
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
        return mfcc_result