import logging
from os.path import (exists, join)
from shutil import copy
from audio_processors import (download_youtube_url,
                              prepare_wav_input,
                              resolve_audio_path)
from chain import Chain
logger = logging.getLogger()


class Preprocess(Chain):
    """
    At the moment aim of this chain is to download youtube input for other chains
    """
    allow_sample_layer_concurrency = False
    requirements = []

    @staticmethod
    def sample_result_filename(out_sample_path):
        return f'{out_sample_path[:-5]}_preprocess_result.json'

    def sample_layer(self, subject, sample_json_filename, sample_settings):
        url = sample_settings.get('url')
        datatype = sample_settings.get('datatype')
        assert(datatype is not None)
        segments = sample_settings.get('segments', [{'start': 'begin', 'stop': 'end'}])
        metadata = sample_settings.get('metadata')
        need_full_length = self.process_settings.get('need_full_length_wav', False)
        intro_duration = metadata.get('intro_duration', 0.0)
        outro_duration = metadata.get('outro_duration', 0.0)
        lang_code = None
        if isinstance(metadata, dict):
            lang_code = metadata.get('language')
        output_path_pattern = join(self.results_dir, subject, sample_json_filename)
        if url.startswith('http'):
            download_youtube_url(url, datatype,
                                 output_path_pattern,
                                 lang_code)
            audio_path = resolve_audio_path(url, datatype, output_path_pattern)
            prepare_wav_input(audio_path, datatype, segments, intro_duration,
                              outro_duration, need_full_length)

        elif url.endswith(datatype):  # assuming that url is local filename
            input_audio_path = join(self.base_dir, subject, url)
            if not exists(input_audio_path):
                raise FileNotFoundError(f'File not found {input_audio_path}')
            audio_path = resolve_audio_path(url, datatype, output_path_pattern)
            copy(input_audio_path, audio_path)
            prepare_wav_input(audio_path, datatype, segments, intro_duration,
                              outro_duration, need_full_length)
        else:
            raise ValueError(f'unhandled url: {url} (settings: {sample_settings})')

        preprocess_result_file = self.sample_result_filename(output_path_pattern)
        logger.info(f'preprocess result file: {preprocess_result_file}')
