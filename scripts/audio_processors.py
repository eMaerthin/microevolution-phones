
import datetime as dt
from functools import reduce
import os
from os.path import (basename, dirname, exists, join, split)

import json
from pydub import AudioSegment
from pytube import (compat, Playlist, YouTube)
from retry import retry

from decorators import check_if_already_done
from format_converters import convert_to_16k_mono_wav, convert_to_mono_wav_original_frame_rate
from tkinter import *
from tqdm import tqdm
import easygui as g
import logging
logger = logging.getLogger()

class PytubePublishedDateRetrieval(compat.HTMLParser):
    vid_published_date = None
    vid_published_timestamp = None

    def handle_starttag(self, tag, attrs):
        if tag == 'meta':
            if attrs[0] == ('itemprop', 'datePublished'):
                self.vid_published_date = dt.datetime.strptime(attrs[1][1], '%Y-%m-%d').date()
                self.vid_published_timestamp = dt.datetime.strptime(attrs[1][1], '%Y-%m-%d').timestamp()


def retrieve_datetime_info_from_yt(yt):
    html_parser = PytubePublishedDateRetrieval()
    html_parser.feed(yt.watch_html)
    published_date = html_parser.vid_published_date
    published_timestamp = html_parser.vid_published_timestamp
    html_parser.close()
    return [published_date, published_timestamp, yt.length]


@retry(KeyError, tries=5, delay=1, backoff=2, logger=logger)
def retrieve_datetime_info(url):
    yt = YouTube(url)
    return retrieve_datetime_info_from_yt(yt)


def process_playlist_url(**kwargs):
    root = Tk()
    url = g.enterbox(msg='Paste url containing playlist link',
                     title='Playlist url', root=root)
    assert url
    output_path = g.diropenbox(msg='Select directory to store list with links',
                               title='Select directory',
                               default='/Volumes/Transcend/phd/subjects/microevolution-lang-phones-data/subjects')
    assert output_path
    entered = g.multenterbox(msg='Enter subject metadata', title='Metadata',
                             fields=['subject', 'language', 'country', 'state',
                                     'profession', 'birth year',
                                     'intro duration [s]', 'outro duration [s]'],
                             values=['', 'en', 'us', '', 'vlogger', '', '0.0', '0.0'])
    assert entered
    [subject, language, country, state, profession, birth_year, intro, outro] = entered
    gender = g.choicebox(msg='Pick a gender of the subject', title='Metadata::Gender', choices=['male', 'female', 'unknown'])
    assert gender
    root.destroy()
    common = {'datatype': 'mp4',
              'metadata': {'subject': subject, 'language': language,
                           'gender': gender, 'country': country,
                           'state': state, 'profession': profession,
                           'birth_year': birth_year,
                           'list_filename': 'list.txt',
                           'playlist_url': url,
                           'intro_duration': intro,
                           'outro_duration': outro}}
    download_youtube_playlist(url, output_path, common)
    success = prepare_jsons(output_path, common)
    logger.info(f'success status: {success}')


def prepare_jsons(output_path, common, metadata_needed=True):
    if not common:
        logger.warning(f'invalid common: {common}')
        return False
    if 'metadata' not in common.keys():
        logger.warning(f'metadata not in common.keys: {common.keys()}')
        return False
    if 'list_filename' not in common['metadata'].keys():
        logger.warning(f'list_filename not in common[metadata].keys: {common["metadata"].keys()}')
        return False
    urls = join(output_path, common['metadata']['list_filename'])
    if not exists(urls):
        logger.warning(f'not existing urls: {urls}')
        return False

    with open(urls, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            url = re.search("(?P<url>https?://[^\s]+)", line).group("url")
            new_json = join(output_path, f'{i}.json')
            if metadata_needed:
                metadata = {}
                args = retrieve_datetime_info(url)
                (published_date, published_timestamp, length_in_seconds) = args
                metadata['published_date'] = str(published_date)
                metadata['published_timestamp'] = float(published_timestamp)
                metadata['length_in_seconds'] = int(length_in_seconds)
                if 'birth_year' in common['metadata'].keys():
                    if common['metadata']['birth_year'].isdecimal():
                        age = published_date.year - int(common['metadata']['birth_year'])
                        metadata['subject_age'] = age
                new_settings = {'url': url, 'metadata': metadata}
            else:
                new_settings = {'url': url}
            with open(new_json, 'w') as fw:
                fw.write(json.dumps(new_settings))
    return True


def download_youtube_playlist(url, output_path, common):
    list_path = join(output_path, 'list.txt')

    with open(join(output_path, 'common.json'), 'w') as file:
        file.write(json.dumps(common))

    @check_if_already_done(list_path)
    @retry(KeyError, tries=3, delay=1, backoff=2, logger=logger)
    def download_playlist(url, list_path):
        logger.info(f'trying to download youtube playlist {url}')
        pl = Playlist(url)
        pl.populate_video_urls()
        with open(list_path, 'w') as f:
            for video_url in pl.video_urls:
                f.write(video_url + '\n')

    download_playlist(url, list_path)


def resolve_caption_path(output_path_pattern, lang_code):
    filename = basename(f'{output_path_pattern[:-5]}_audio')
    output_path = dirname(output_path_pattern)
    return join(output_path, f'{filename}_{lang_code}.captions')


def download_youtube_url(url, datatype, output_path_pattern, lang_code):
    audio_path = resolve_audio_path(url, datatype, output_path_pattern)
    caption_path = resolve_caption_path(output_path_pattern, lang_code)
    ignore_already_done = False

    @check_if_already_done(audio_path, ignore_done=ignore_already_done)
    @retry(KeyError, tries=7, delay=1, backoff=2, logger=logger)
    def download_audio(yt, datatype, audio_path):
        logger.info(f'trying to download youtube movie {url}')
        stream = yt.streams.filter(subtype=datatype).order_by('itag').asc().first()
        assert(stream is not None)
        logger.info(f'Found valid stream: {stream}')
        output_path, filename = split(f'{audio_path[:-4]}')
        stream.download(output_path=output_path, filename=filename)

    @check_if_already_done(caption_path, ignore_done=ignore_already_done,
                           validator=lambda v: v)
    @retry(KeyError, tries=7, delay=1, backoff=2, logger=logger)
    def download_caption(yt, caption_path, lang_code):
        if lang_code is None:
            lang_code = 'en' # by default
        caption = yt.captions.get_by_language_code(lang_code)
        if caption is not None:
            logger.info(f'Found valid caption for lang_code: {lang_code}')
            with open(caption_path, 'w') as f:
                f.write(caption.xml_captions)
                return True
        logger.warning(f'A caption for lang_code {lang_code} is not found')
        return False

    @retry(KeyError, tries=7, delay=1, backoff=1, logger=logger)
    def get_yt_handler(url):
        return YouTube(url)

    if ignore_already_done or not exists(audio_path):  # (download caption is not mandatory)
        yt = get_yt_handler(url)
        download_audio(yt, datatype, audio_path)
        try:
            download_caption(yt, caption_path, lang_code)
        except AttributeError as e:
            logger.error(f'Attribute error: {e} - Will abandon downloading captions')


def time_token_to_sec(time_token):
    if isinstance(time_token, float):
        return time_token
    elif not isinstance(time_token, str):
        raise ValueError("Don't know how to interpret provided time_token")
    time_parts = time_token.split(':') # we expect to see something like HH:MM:SS.XYZ
    assert(0 <= len(time_parts) <= 3)
    sec_time = sum(pow(60,i) * float(t)
                   for i,t in enumerate(reversed(time_parts)))
    return sec_time


def map_time(time_token, rec_time, intro, outro):
    begin = intro
    end = rec_time - outro
    if time_token is 'begin':
        return begin
    elif time_token is 'end':
        return end
    else:
        sec_time = time_token_to_sec(time_token)
        assert begin <= sec_time, f'{sec_time} should be larger than {begin}'
        assert sec_time <= end, f'{sec_time} should be smaller than {end}'
        return sec_time


def parse_segment(segment, rec_time, intro, outro):
    assert 'start' in segment
    assert 'stop' in segment
    assert len(segment) is 2
    # pydub does things in milliseconds
    mapped_segment = {k: 1000 * map_time(v, rec_time, intro, outro)
                      for k, v in segment.items()}
    return mapped_segment


def resolve_audio_path(url_or_local, datatype, output_path_pattern):
    output_path = dirname(output_path_pattern)
    if url_or_local.startswith('http'):
        filename = basename(f'{output_path_pattern[:-5]}_audio')
        audio_path = join(output_path, f'{filename}.{datatype}')
    elif url_or_local.endswith(datatype):
        audio_path = join(output_path, url_or_local)
    else:
        raise ValueError(f'unhandled url_or_local: {url_or_local}')
    return audio_path


def audio_and_segment_paths(in_audio_path, use_original_frequency, audio_format='wav'):
    dot_in_extension_len = len(in_audio_path.split('.')[-1]) + 1
    if use_original_frequency:
        audio_path = f'{in_audio_path[:-dot_in_extension_len]}_orig_freq.{audio_format}'
    else:
        audio_path = f'{in_audio_path[:-dot_in_extension_len]}_16khz.{audio_format}'
    dot_extension_len = len(audio_path.split('.')[-1]) + 1
    segments_path = f'{audio_path[:-dot_extension_len]}_segments.{audio_format}'
    return audio_path, segments_path


def load_recording(audio_path):
    audio_format = audio_path.split('.')[-1]
    if audio_format == 'wav':
        return AudioSegment.from_wav(audio_path)
    return AudioSegment.from_file(audio_path)


def filter_segments(audio_path, segments_path, segments, intro_duration, outro_duration, silence_duration=0):
    recording = load_recording(audio_path)
    recording_time = recording.duration_seconds
    parsed_segments = [parse_segment(segment, recording_time,
                                     intro_duration, outro_duration)
                       for segment in segments]
    silence = AudioSegment.empty()
    if silence_duration > 0:
        silence = AudioSegment.silent(duration=silence_duration)
    filtered_rec = reduce(lambda x, y: x + recording[y['start']:y['stop']] + silence,
                          parsed_segments, AudioSegment.empty())
    filtered_rec.export(segments_path, format='wav')


def prepare_wav_input(audio_path, datatype, segments, intro_duration,
                      outro_duration, need_full_length=False):

    def convert_to_wav(audio_path, datatype, segments, original_freq, need_full_length):
        wav_path, segments_path = audio_and_segment_paths(audio_path,
                                                          original_freq)
        convert_check_path = wav_path
        if not need_full_length:
            convert_check_path = segments_path

        @check_if_already_done(segments_path)
        def wrapped_filter_segments(audio_path, segments_path, segments):
            filter_segments(audio_path, segments_path, segments, intro_duration, outro_duration)


        @check_if_already_done(convert_check_path)
        def convert(audio_path, wav_path, segments_path, segments, original_freq):
            if original_freq:
                convert_to_mono_wav_original_frame_rate(input=audio_path,
                                                        output=wav_path,
                                                        input_format=datatype)
            else:
                convert_to_16k_mono_wav(input=audio_path, output=wav_path,
                                        input_format=datatype)
            filter_segments(wav_path, segments_path, segments)
        convert(audio_path, wav_path, segments_path, segments, original_freq)
        if not need_full_length and exists(wav_path):
            os.remove(wav_path)

    for original_freq in (True, False):
        convert_to_wav(audio_path, datatype, segments, original_freq, need_full_length)
