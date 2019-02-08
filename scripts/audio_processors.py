from functools import reduce
from os.path import join

from pydub import AudioSegment
from pytube import YouTube

from decorators import check_if_already_done
from format_converters import to_16k_mono_wav

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
    caption_path = join(output_path, f'{filename}.captions')
    @check_if_already_done(caption_path, verbose)
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
    download_caption(url, caption_path, lang_code, verbose)
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
