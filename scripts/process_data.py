from functools import reduce
import json
from os import walk
from os.path import (isfile, join)
import ssl

import fire
from pytube import YouTube

ssl._create_default_https_context = ssl._create_stdlib_context

def sum_series(series):
    return reduce(lambda x, y: x + len(y[1]), series, 0)

def list_subjects(dir, verbose):
    series = []
    for (dirpath, dirnames, filenames) in walk(dir):
        jsons = list(filter(lambda x: all([x.endswith("json"), not(x.endswith("result.json"))]), filenames))
        if len(jsons) > 0:
            dir_series = [(dirpath[len(dir):], jsons)]
            if verbose > 1:
                print(f'Added subject-series pair: {dir_series}')
            series.extend(dir_series)
    series_sum = sum_series(series)
    unprocessed_series = list(filter(lambda s: any([not(isfile(join(dir, s[0], f'{item[:-5]}_result.json'))) for item in s[1]]), series))
    unprocessed_series_sum = sum_series(unprocessed_series)
    if verbose > 0:
        print(f'{len(series)} subject(s) ({series_sum} series) in the database in total: {series}')
        print(f'{len(unprocessed_series)} subject(s) unprocessed ({unprocessed_series_sum} series): {unprocessed_series}')
    return series, unprocessed_series

def download_youtube_url(url, datatype, output_path, filename, lang_code, verbose):
    # TODO: check if .done files exist. If yes, do not repeat downloading
    audio_done_path = join(output_path, f'{filename}.{datatype}.done')
    caption_path = join(output_path, f'{filename}.captions')
    caption_done_path = f'{caption_path}.done'
    if not isfile(audio_done_path):
        if verbose > 0:
            print(f' trying to download youtube movie {url}')
        yt = YouTube(url)
        audio_stream = yt.streams.filter(subtype=datatype, only_audio=True).order_by('abr').asc().first()
        assert(audio_stream is not None)
        if verbose > 0:
            print(f'Found valid audio stream: {audio_stream}')
        audio_stream.download(output_path=output_path, filename=filename)
        with open(audio_done_path, 'w') as f:
            f.write('OK')
    elif verbose > 0:
        print(f' skipping downloading youtube movie {url} because {audio_done_path} already exists')

    if not isfile(caption_done_path):
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
        with open(caption_done_path, 'w') as f:
            f.write('OK')
    elif verbose > 0:
        print(f' skipping downloading captions for {url} because {caption_done_path} already exists')

def process_data(subjects_homedir = '../subjects/', verbose = 1):
    _, series_to_process = list_subjects(subjects_homedir, verbose)
    for subject, series in series_to_process:
        working_dir = join(subjects_homedir, subject)
        for (series_path, file_name) in ((join(working_dir, s), s) for s in series):
            with open(series_path) as f:
                if verbose > 0:
                    print(f' series_path: {series_path}')
                d = json.load(f)
                url = d.get('url')
                assert(url is not None)
                datatype = d.get('datatype')
                assert(datatype is not None)
                metadata = d.get('metadata')
                lang_code = None
                if metadata is not None:
                    lang_code = metadata.get('language')
                audio_file_name=f'{file_name[:-5]}_audio'
                download_youtube_url(url, datatype, working_dir, audio_file_name, lang_code, verbose)

if __name__ == '__main__':
    fire.Fire({
              'run': process_data,
              })

