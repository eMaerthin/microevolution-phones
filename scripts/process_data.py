from functools import (reduce, wraps)
import json
from os import walk
from os.path import (isfile, join)
import ssl

import fire
from marshmallow import Schema, fields
from pocketsphinx.pocketsphinx import Decoder
from pydub import AudioSegment
from pytube import YouTube

from format_converters import to_16k_mono_wav

ssl._create_default_https_context = ssl._create_stdlib_context

class SegmentSchema(Schema):
    start = fields.Str()
    stop = fields.Str()

class MetadataSchema(Schema):
    subject = fields.Str()
    language = fields.Str()
    country = fields.Str()
    profession = fields.Str()
    gender = fields.Str()
    age = fields.Integer()

class SeriesSchema(Schema):
    url = fields.Str()
    datatype = fields.Str()
    record_date = fields.Str()
    segments = fields.Nested(SegmentSchema, many=True)
    metadata = fields.Nested(MetadataSchema)

def sum_series(series):
    return reduce(lambda x, y: x + len(y[1]), series, 0)

def done(path):
    return f'{path}.done'

def result(json_path):
    return f'{json_path[:-5]}_result.json'

def list_subjects(dir, verbose):
    series = []
    for (dirpath, dirnames, filenames) in walk(dir):
        jsons = list(filter(lambda x: all([x.endswith("json"),
                                           not(x.endswith("result.json"))]),
                            filenames))
        if len(jsons) > 0:
            dir_series = [(dirpath[len(dir):], jsons)]
            if verbose > 1:
                print(f'Added subject-series pair: {dir_series}')
            series.extend(dir_series)
    series_sum = sum_series(series)
    unprocessed_series = list(filter(lambda s:
                                     any([not(isfile(join(dir, s[0],
                                                          result(item))))
                                          for item in s[1]]), series))
    unprocessed_series_sum = sum_series(unprocessed_series)
    if verbose > 0:
        print(f'{len(series)} subject(s) ({series_sum} series) \
              in the database in total: {series}')
        print(f'{len(unprocessed_series)} subject(s) unprocessed \
              ({unprocessed_series_sum} series): {unprocessed_series}')
    return series, unprocessed_series


def check_if_already_done(check_path, verbose=0):
    def decorator_if_already_done(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isfile(done(check_path)):
                func(*args, **kwargs)
                with open(done(check_path), 'w') as f:
                    f.write('OK')
            elif verbose > 0:
                print(f'skipping evaluating this method because \
                      {done(check_path)} already exists')
        return wrapper
    return decorator_if_already_done
                       
def download_youtube_url(url, datatype, output_path, filename, lang_code,
                         verbose):
    audio_path = join(output_path, f'{filename}.{datatype}')
    @check_if_already_done(audio_path, verbose)
    def download_audio(url, datatype, output_path, filename, verbose):
        if verbose > 0:
            print(f' trying to download youtube movie {url}')
        yt = YouTube(url)
        audio_stream = yt.streams.filter(subtype=datatype,
                                         only_audio=True).order_by('abr').asc().first()
        assert(audio_stream is not None)
        if verbose > 0:
            print(f'Found valid audio stream: {audio_stream}')
        audio_stream.download(output_path=output_path, filename=filename)
        return yt

    caption_path = join(output_path, f'{filename}.captions')
    @check_if_already_done(caption_path, verbose)
    def download_caption(yt, caption_path, lang_code, verbose):
        if lang_code is None:
            lang_code = 'en' # by default
        caption = yt.captions.get_by_language_code(lang_code)
        if caption is not None:
            if verbose > 0:
                print(f'Found valid caption for lang_code: {lang_code}')
            with open(caption_path, 'w') as f:
                f.write(caption.xml_captions)
        elif verbose > 0:
            print(f'Warning, a caption for lang_code {lang_code} not found')

    yt = download_audio(url, datatype, output_path, filename, verbose)
    download_caption(yt, caption_path, lang_code, verbose)
    return audio_path, caption_path

def time_string_to_sec(str):
    time_parts = str.split(':') # we expect to see something like HH:MM:SS.XYZ
    assert(0 <= len(time_parts) <= 3)
    sec_time = sum(pow(60,i) * float(t)
                   for i,t in enumerate(reversed(time_parts)))
    return sec_time

def map_time_string(str, rec_time):
    if str is 'begin':
        return 0.0
    elif str is 'stop':
        return rec_time
    else:
        sec_time = time_string_to_sec(str)
        assert(0 <= sec_time <= rec_time)
        return sec_time

def parse_segment(segment, rec_time):
    assert 'start' in segment
    assert 'stop' in segment
    assert len(segment) is 2
    # pydub does things in milliseconds
    mapped_segment = {k: 1000 * map_time_string(v, rec_time) for k, v in segment.items()}
    return mapped_segment

def prepare_wav_input(audio_path, datatype, segments, verbose):
    wav_path = f'{audio_path[:-3]}wav'
    @check_if_already_done(wav_path, verbose)
    def convert_to_wav(audio_path, wav_path, datatype):
        to_16k_mono_wav(input=audio_path, output=wav_path,
                        input_format=datatype)

    segments_path = f'{audio_path[:-4]}_segments.wav'
    @check_if_already_done(segments_path, verbose)
    def filter_segments(wav_path, segments_path, segments):
        recording = AudioSegment.from_wav(wav_path)
        recording_time = recording.duration_seconds
        parsed_segments = [parse_segment(segment, recording_time)
                           for segment in segments]
        empty = AudioSegment.empty()
        filtered_rec = reduce(lambda x, y: x + recording[y['start']:y['stop']],
                              parsed_segments, empty)
        filtered_rec.export(segments_path, format='wav')
    convert_to_wav(audio_path, wav_path, datatype)
    filter_segments(wav_path, segments_path, segments)
    return wav_path, segments_path

def compute_phonemes(segments_path):
    # todo
    pass

def process_data(subjects_homedir = '../subjects/', verbose = 1):
    _, series_to_process = list_subjects(subjects_homedir, verbose)
    for subject, series in series_to_process:
        working_dir = join(subjects_homedir, subject)
        for (series_path, series_json_filename) in ((join(working_dir, s), s)
                                                    for s in series):
            with open(series_path) as f:
                if verbose > 0:
                    print(f' series_path: {series_path}')
                d = json.load(f)
                series = SeriesSchema()
                data = series.dump(d)
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
                audio_file_name=f'{series_json_filename[:-5]}_audio'
                audio_path, caption_path = download_youtube_url(url,
                                                                datatype,
                                                                working_dir,
                                                                audio_file_name,
                                                                lang_code,
                                                                verbose)
                wav_path, segments_path = prepare_wav_input(audio_path,
                                                            datatype, segments,
                                                            verbose)
                compute_phonemes(segments_path)

if __name__ == '__main__':
    fire.Fire({
              'run': process_data,
              })
