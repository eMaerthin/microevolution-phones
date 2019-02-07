from functools import partial
import fire
import numpy as np
from pydub import AudioSegment

def audio_converter(input, output, input_format='mp3',
                    output_format='wav', frame_rate=16000, channels=1):
    sound = AudioSegment.from_file(input, format=input_format)
    sound = sound.set_channels(channels)
    sound = sound.set_frame_rate(frame_rate)
    try:
        sound.export(output, format=output_format)
    except IOError:
        print(f'Error while converting {input} to {output}')

to_16k_mono_wav = partial(audio_converter, output_format='wav',
                                  frame_rate=16000, channels=1)

def convert_to_16k_mono_wav(input, output, input_format='mp3'):
    to_16k_mono_wav(input, output, input_format)

if __name__ == '__main__':
    fire.Fire({
              'audio': audio_converter,
              'to_16k_mono_wav': convert_to_16k_mono_wav,
              })
