import json
from os.path import join

from marshmallow import pprint
from pocketsphinx.pocketsphinx import Decoder

from audio_processors import download_youtube_url, prepare_wav_input
from decorators import check_if_already_done
from schemas import *

from pipeline import Pipeline


class FormantsPipeline(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_formants_result.json'

    @staticmethod
    def filename_prerequisites():
        def wav_path(json_path):
            return f'{json_path[:-5]}_audio_segments.wav'
        return [wav_path]

    def pipeline(self, series_json_filename, working_dir, data):
        # TODO
        '''
        url = data.get('url')
        datatype = data.get('datatype')
        assert(datatype is not None)
        segments = data.get('segments')
        if segments is None:
            segments = [{'start':'begin', 'stop':'end'}]
        metadata = data.get('metadata')
        lang_code = None
        if metadata is not None:
            lang_code = metadata.get('language')
        '''
        series_path = join(working_dir, series_json_filename)
        input_audio_path = f'{series_path[:-5]}_audio_segments.wav'
        formants_result_file = self.result_filename(series_path)
        if self.verbose > 0:
            print(f'formants result file: {formants_result_file}')
