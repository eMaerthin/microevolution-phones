import fire
import numpy as np
from pydub import AudioSegment

def converter(input_mp3, output_wav, frame_rate=16000, channels=1):
    sound = AudioSegment.from_file(input_mp3, format="mp3")
    sound = sound.set_channels(channels)
    sound = sound.set_frame_rate(frame_rate)
    try:
        sound.export(output_wav,
                     format="wav")
    except IOError:
        print(f'Error while exporting to {output_wav}')
                        

if __name__ == '__main__':
    fire.Fire({
              'converter': converter,
              })
