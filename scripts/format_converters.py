from functools import partial
import fire
import logging
from pydub import AudioSegment
logger = logging.getLogger()


def get_segment(input_audio, input_format):
    return AudioSegment.from_file(input_audio, format=input_format)


def get_frame_rate(input_audio, input_format):
    return get_segment(input_audio, input_format).frame_rate


def audio_converter(input_audio, output_audio, input_format='mp3',
                    output_format='wav', frame_rate=None, channels=None):
    sound = get_segment(input_audio, input_format)
    if channels:
        sound = sound.set_channels(channels)
    if frame_rate:
        sound = sound.set_frame_rate(frame_rate)
    try:
        sound.export(output_audio, format=output_format)
    except IOError as e:
        logger.error(f'Error while converting {input_audio} to {output_audio}: {e}')
    return sound.channels, sound.frame_rate


_to_16k_mono_wav = partial(audio_converter, output_format='wav',
                          frame_rate=16000, channels=1)


def convert_to_16k_mono_wav(input, output, input_format='mp3'):
    _to_16k_mono_wav(input, output, input_format)


def convert_to_mono_wav_original_frame_rate(input, output, input_format='mp3'):
    audio_converter(input, output, input_format, output_format='wav', channels=1)


if __name__ == '__main__':
    from logging.config import fileConfig
    fileConfig('logging.conf')

    fire.Fire({
              'audio': audio_converter,
              'to_16k_mono_wav': convert_to_16k_mono_wav,
              })
