import json

import numpy as np
from python_speech_features import mfcc

from decorators import check_if_already_done
from format_converters import get_segment
from schemas import *
from pipelines.formants_pipeline import FormantsPipeline


class MfccGlobalPipeline(FormantsPipeline):
    '''
    'FormantsPipeline' prepares segments wav in original frequency and delivers it as 'segments_path'
    '''
    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_mfcc_global_result.json'

    @staticmethod
    def result_filename_postprocessed(json_path):
        return MfccGlobalPipeline.result_filename(json_path)
        # return f'{json_path[:-5]}_mfcc_global_result.csv'

    def compute_global_mfcc(self, segments_path, mfcc_global_result_path, mfcc_nfft=2048, mfcc_winstep=0.1):
        @check_if_already_done(mfcc_global_result_path, self.verbose, lambda x: x)
        def store_global_mfcc(segments_path, mfcc_global_result_path):
            print(segments_path)
            wav = get_segment(segments_path, 'wav')
            frequency = wav.frame_rate
            mfcc_result = []
            array_of_samples = np.array(wav.get_array_of_samples())
            mfcc_features = mfcc(array_of_samples, samplerate=frequency, nfft=mfcc_nfft, winstep=mfcc_winstep)
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

    def compute_target(self, segments_path, _, series_json_path):
        mfcc_global_result_path = self.result_filename(series_json_path)
        self.compute_global_mfcc(segments_path, mfcc_global_result_path)
        if self.verbose > 0:
            print(f'mfcc_global result path: {mfcc_global_result_path}')
