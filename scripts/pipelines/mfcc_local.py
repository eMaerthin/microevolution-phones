import json

import numpy as np
from python_speech_features import mfcc
from scipy.signal import spectrogram

from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from pipelines.formants_pipeline import FormantsPipeline
from pipelines.phoneme_pipeline import PhonemePipeline


class MfccPipeline(FormantsPipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_mfcc_result.json'

    @staticmethod
    def result_filename_postprocessed(json_path):
        return f'{json_path[:-5]}_mfcc_result.csv'

    def compute_mfcc(self, segments_path, phonemes_result_path, mfcc_result_path,
                             phoneme_len=2048, ignore_shorter_phonemes=True):
        @check_if_already_done(mfcc_result_path, self.verbose, lambda x: x)
        def store_mfcc(segments_path, phonemes_result_path, mfcc_result_path):
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
                mfcc_result = []
                for info in phonemes_info:
                    start, stop = (1000 * info['start'], 1000 * info['end'])
                    segment = np.array(wav[start:stop].get_array_of_samples())
                    if ignore_shorter_phonemes and segment.size < phoneme_len:
                        continue
                    mfcc_features = mfcc(segment, frequency, nfft=phoneme_len)
                    for i in range(len(mfcc_features)):
                        ith_mfcc = np.array(mfcc_features[i, :])
                        ith_mfcc_result_row = {'i': i, 'length': len(mfcc_features),
                                               'mfcc': ith_mfcc, **info}
                        mfcc_result.append(ith_mfcc_result_row)
                mfcc_schema = MfccSchema()
                mfcc_dict = {'mfcc_info': mfcc_result}
                result = mfcc_schema.dumps(mfcc_dict)
                with open(mfcc_result_path, 'w') as result_f:
                    result_f.write(result)
                    return True
        store_mfcc(segments_path, phonemes_result_path, mfcc_result_path)

    def compute_target(self, segments_path, phonemes_path, series_json_path):
        mfcc_result_path = self.result_filename(series_json_path)
        self.compute_mfcc(segments_path, phonemes_path, mfcc_result_path)
        if self.verbose > 0:
            print(f'mfcc result path: {mfcc_result_path}')
