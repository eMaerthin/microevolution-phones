
import datetime as dt
from functools import reduce
from os.path import (isfile, join)
from urllib.error import URLError

import json
from pydub import AudioSegment
from pytube import (compat, extract, Playlist, YouTube)
from retry import retry

from decorators import check_if_already_done
from format_converters import get_frame_rate, convert_to_16k_mono_wav, convert_to_mono_wav_original_frame_rate
from tkinter import *
from tqdm import tqdm
import easygui as g


def process_playlist_url(verbose=1, **kwargs):
    root = Tk()
    url = g.enterbox(msg='Paste url containing playlist link',
                     title='Playlist url', root=root)
    assert url
    output_path = g.diropenbox(msg='Select directory to store list with links',
                               title='Select directory',
                               default='/Volumes/Transcend/phd/subjects/microevolution-lang-phones-data/subjects')
    assert output_path
    entered = g.multenterbox(msg='Enter subject id', title='Metadata details',
                             fields=['subject', 'language', 'country', 'state', 'profession', 'birth year'],
                             values=['', 'en', 'us', '', 'vlogger', ''])
    assert entered
    [subject, language, country, state, profession, birth_year] = entered
    gender = g.choicebox(msg='Pick a gender', title='Metadata::Gender', choices=['male', 'female', 'unknown'])
    assert gender
    root.destroy()
    common = {'datatype':'mp4', 'metadata': {'subject': subject, 'language': language, 'gender': gender,
                                             'country': country, 'state': state, 'profession': profession,
                                             'birth_year': birth_year, 'list': 'list.txt'}}
    download_youtube_playlist(url, output_path, common, verbose)
    success = prepare_jsons(output_path, common)
    print(success)


def prepare_jsons(output_path, common, metadata_needed=True):
    if not common:
        return False
    if 'metadata' not in common.keys():
        return False
    if 'list' not in common['metadata'].keys():
        return False
    urls = join(output_path, common['metadata']['list'])
    if not isfile(urls):
        return False

    class PytubePublishedDateRetrieval(compat.HTMLParser):
        vid_published_date = None

        def handle_starttag(self, tag, attrs):
            if tag == 'meta':
                if attrs[0] == ('itemprop', 'datePublished'):
                    self.vid_published_date = dt.datetime.strptime(attrs[1][1], '%Y-%m-%d').date()

    @retry(KeyError, tries=5, delay=1, backoff=2)
    def retrieve_published_date_and_recorded_length(url):
        yt = YouTube(url)
        html_parser = PytubePublishedDateRetrieval()
        html_parser.feed(yt.watch_html)
        published_date = html_parser.vid_published_date
        html_parser.close()
        return [published_date, yt.length]

    with open(urls, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            url = re.search("(?P<url>https?://[^\s]+)", line).group("url")
            new_json = join(output_path, f'{i}.json')
            if metadata_needed:
                metadata = {}
                [published_date, length_in_seconds] = retrieve_published_date_and_recorded_length(url)
                metadata['published_date'] = str(published_date)
                metadata['length_in_seconds'] = length_in_seconds
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


def download_youtube_playlist(url, output_path, common, verbose):
    list_path = join(output_path, 'list.txt')

    with open(join(output_path, 'common.json'), 'w') as file:
        file.write(json.dumps(common))

    @check_if_already_done(list_path, verbose)
    @retry(KeyError, tries=3, delay=1, backoff=2)
    def download_playlist(url, list_path, verbose):
        if verbose > 0:
            print(f' trying to download youtube playlist {url}')
        pl = Playlist(url)
        pl.populate_video_urls()
        with open(list_path, 'w') as f:
            for video_url in pl.video_urls:
                f.write(video_url + '\n')

    download_playlist(url, list_path, verbose)


def download_youtube_url(url, datatype, output_path, filename, lang_code,
                         verbose):
    audio_path = join(output_path, f'{filename}.{datatype}')
    caption_path = join(output_path, f'{filename}.captions')

    @check_if_already_done(audio_path, verbose)
    @retry(KeyError, tries=7, delay=1, backoff=2)
    def download_audio(url, datatype, output_path, filename, verbose):
        if verbose > 0:
            print(f' trying to download youtube movie {url}')
        yt = YouTube(url)
        audio_stream = yt.streams.filter(subtype=datatype).order_by('itag').asc().first()
        assert(audio_stream is not None)
        if verbose > 0:
            print(f'Found valid audio stream: {audio_stream}')
        audio_stream.download(output_path=output_path, filename=filename)

    @check_if_already_done(caption_path, verbose)
    @retry(KeyError, tries=3, delay=1, backoff=2)
    def download_caption(url, caption_path, lang_code, verbose):
        yt = YouTube(url)
        if lang_code is None:
            lang_code = 'en' # by default
            try:
                caption = yt.captions.get_by_language_code(lang_code)
                if caption is not None:
                    if verbose > 0:
                        print(f'Found valid caption for lang_code: {lang_code}')
                    with open(caption_path, 'w') as f:
                        f.write(caption.xml_captions)
                elif verbose > 0:
                    print(f'Warning, a caption for lang_code {lang_code} not found')
            except AttributeError:
                print(f'Attribute error:(')

    download_audio(url, datatype, output_path, filename, verbose)
    try:
        download_caption(url, caption_path, lang_code, verbose)
    except (KeyError, URLError):
        pass
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
    elif str is 'end':
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


def prepare_wav_input(audio_path, datatype, segments, verbose, use_original_frequency=False):
    if use_original_frequency:
        wav_path = f'{audio_path[:-4]}_orig_freq.wav'
    else:
        wav_path = f'{audio_path[:-3]}wav'

    segments_path = f'{wav_path[:-4]}_segments.wav'

    @check_if_already_done(wav_path, verbose)
    def convert_to_16k_freq_mono_wav(audio_path, wav_path, datatype):
        convert_to_16k_mono_wav(input=audio_path, output=wav_path,
                                input_format=datatype)

    @check_if_already_done(wav_path, verbose)
    def convert_to_original_freq_mono_wav(audio_path, wav_path, datatype):
        convert_to_mono_wav_original_frame_rate(input=audio_path, output=wav_path,
                                                input_format=datatype)

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

    if use_original_frequency:
        convert_to_original_freq_mono_wav(audio_path, wav_path, datatype)
    else:
        convert_to_16k_freq_mono_wav(audio_path, wav_path, datatype)
    filter_segments(wav_path, segments_path, segments)
    return wav_path, segments_path
