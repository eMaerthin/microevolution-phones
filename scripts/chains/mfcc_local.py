import json
import logging
import os

import numpy as np
from python_speech_features import mfcc

from format_converters import get_segment
from schemas import *
from chains.labeled_base import DEFAULT_BLACKLISTED_LABELS
from chains.mfcc import Mfcc
from chains.phoneme import Phoneme
from chains.words import Words
logger = logging.getLogger()


class MfccLocal(Mfcc):
    """
    MfccLocal computes Mfcc features for each label (phoneme or word)
    from the sample that are not blacklisted based on label that is
    received from Phoneme/Words chain.

    It subclasses abstract chain class Mfcc
    """

    abstract_class = False
    requirements = [Phoneme, Words]

    @staticmethod
    def sample_result_filename(out_sample_path):
        filename, _ = os.path.splitext(out_sample_path)
        return f'{filename}_mfcc_result.json'

    @staticmethod
    def filenames_to_skip_sample(out_sample_path):
        filename, _ = os.path.splitext(out_sample_path)
        return [f'{filename}_mfcc_result.csv']

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

    def compute_mfcc(self, segments_path, labels_result_path):
        """

        :param segments_path: path to the input wav
        :param labels_result_path: path to labels results (phonemes or words)
        that is required by the Local version of the Mfcc
        :return: computed list of mfcc features with all required metadata
        """
        wav = get_segment(segments_path, 'wav')
        frequency = wav.frame_rate
        chunk_len = self.process_settings.get("chunk_len",
                                              self.process_settings.get("phoneme_len", 2048))
        ignore_short_chunks = self.process_settings.get("ignore_short_chunks",
                                                        self.process_settings.get("ignore_shorter_phonemes", True))
        mfcc_nfft = self.process_settings.get("mfcc_nfft", 2048)
        mfcc_winstep = self.process_settings.get("mfcc_winstep", 0.01)

        self.blacklisted_labels = self.process_settings.get("blacklisted", DEFAULT_BLACKLISTED_LABELS)
        with open(labels_result_path, 'r') as f:
            schema = DecoderOutputSchema()
            json_file = json.load(f)
            labels_result = schema.load(json_file)
            labels_info = [info for info in labels_result['segment_info']
                           if info['word'] not in self.blacklisted_labels]


        mfcc_result = []
        for info in labels_info:
            start, stop = (1000 * info['start'], 1000 * info['end'])
            segment = np.array(wav[start:stop].get_array_of_samples())
            if ignore_short_chunks and segment.size < chunk_len:
                continue
            mfcc_features = mfcc(segment, samplerate=frequency,
                                 nfft=mfcc_nfft, winstep=mfcc_winstep)
            for i in range(len(mfcc_features)):
                ith_mfcc = np.array(mfcc_features[i, :])
                ith_mfcc_result_row = {'i': i, 'length': len(mfcc_features),
                                       'mfcc': ith_mfcc, **info}
                mfcc_result.append(ith_mfcc_result_row)
        return mfcc_result