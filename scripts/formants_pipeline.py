import json
from os.path import join

from marshmallow import pprint
from pocketsphinx.pocketsphinx import Decoder

from audio_processors import download_youtube_url, prepare_wav_input
from decorators import check_if_already_done
from schemas import *

def result_formants(json_path):
    return f'{json_path[:-5]}_formants_result.json'

def formants_pipeline(series_json_filename, working_dir, data,
                     verbose):
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
    input_audio_path=f'{series_path[:-5]}_audio_segments.wav'
    formants_result_file = result_formants(series_path)
    if verbose>0:
        print(f'formants result file: {formants_result_file}')
