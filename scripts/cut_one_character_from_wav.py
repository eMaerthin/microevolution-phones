import fire
from functools import reduce
import json
from os.path import join
from pickle import loads

# import unittest

import numpy as np
import pandas as pd
from scipy.io.wavfile import read as read_wav, write as write_wav

from audio_processors import filter_segments
last_ok = None

class WavProcessor(object):
    """A simple WavProcessor class."""

    def __init__(self, segments_path, wav_path=None, timestamp_offset=0,
                 root_dir='.'):
        self.path = segments_path
        self.root_dir = root_dir
        if not wav_path:
            wav_path = segments_path
        self.wav_path = wav_path
        self.timestamp_offset = timestamp_offset
        self.rate = None
        self.data = None
        self.read_path()

    def read_path(self):
        rate, data = read_wav(self.path)

        self.rate = rate
        self.data = data
        print(f' sampling rate: {self.rate} data shape: {self.data.shape}')


    def save_wav(self, out_path, data):
        print(f' sampling rate: {self.rate} data shape: {data.shape}')
        write_wav(out_path, self.rate, data)

    def cut_one_chunk(self, start_sec, end_sec):
        start, end = map(lambda x: int(x * self.rate), [start_sec, end_sec])
        return self.data[start:end]

    def cut_multiple_chunks(self, starts, stops, wait_ms=0, min_len_s=0):
        assert len(starts) == len(stops)
        chunks = [self.cut_one_chunk(start, max(start + min_len_s, stop))
                  for (start, stop) in zip(starts, stops)]
        filter_segments()
        wait_chunk = np.zeros(int(wait_ms * self.rate / 1000.0))
        return reduce(lambda x, y: np.concatenate((x, wait_chunk, y)), chunks)

    def save_specific_character(self, character, annotation_file=None,
                                save_path=None, wait_ms=0, min_len_s=0):
        if annotation_file is None:
            annotation_file = self.path + "_out_log.txt"
        audio_output_format = 'wav'
        if save_path is None:
            save_path = self.path + f"_test.{audio_output_format}"
        annotations = Annotations(anno_path=annotation_file,
                                  wav_path=self.wav_path,
                                  root_dir='.',
                                  timestamp_offset=self.timestamp_offset)
        view_df = annotations.view_specific_character(character)
        starts, stops = map(lambda x: view_df[x].values.tolist(), ['Start', 'Stop'])
        self.save_segments(starts, stops, wait_ms, min_len_s, save_path)

    def save_segments(self, starts, stops, wait_ms, min_len_s, save_path):
        assert len(starts) == len(stops)
        segments = [{'start': start, 'stop': min(len(self.data)/self.rate - 0.01,
                                                 max(start + min_len_s, stop))}
                    for (start, stop) in zip(starts, stops)]
        filter_segments(self.path, save_path, segments, 0.0, 0.0, silence_duration=wait_ms)

    def save_specific_coords(self, min_x1=-np.inf, min_x2=-np.inf,
                             max_x1=np.inf, max_x2=np.inf, annotation_file=None,
                             save_path=None, wait_ms=0, min_len_s=0):
        if annotation_file is None:
            annotation_file = self.path + "_out_log.txt"
        audio_output_format = 'wav'
        if save_path is None:
            save_path = self.path + f"_test.{audio_output_format}"
        annotations = Annotations(anno_path=annotation_file,
                                  wav_path=self.wav_path,
                                  root_dir=self.root_dir)
        view_df = annotations.view_specific_coords(min_x1=min_x1, min_x2=min_x2,
                                                   max_x1=max_x1, max_x2=max_x2)
        starts, stops = map(lambda x: view_df[x].values.tolist(), ['Start', 'Stop'])
        assert len(starts) > 0
        assert len(starts) == len(stops)
        segments = zip(starts, stops)
        global last_ok
        last_ok = None

        def filter_events(x):
            global last_ok
            if last_ok is None:
                last_ok = x[0]
                return True
            if last_ok + min_len_s <= x[0]:
                last_ok = x[0]
                return True
            return False
        filtered = filter(filter_events, sorted(segments))
        filtered_starts, filtered_stops = zip(*filtered)
        self.save_segments(filtered_starts, filtered_stops, wait_ms, min_len_s, save_path)
        #out_data = self.cut_multiple_chunks(filtered_starts, filtered_stops, wait_ms, min_len_s)
        #self.save_wav(save_path, out_data)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_valid_entry(obj):
    words = str(obj).split()
    if len(words) < 4:
        return False
    return isfloat(words[1])


class Annotations(object):
    """A simple Annotations class."""

    def __init__(self, anno_path, wav_path=None, root_dir='.',
                 mfcc_window_duration=0.025):
        self.path = anno_path
        self.wav_path = wav_path
        self.annotations_df = None
        self.root_dir = root_dir
        self.mfcc_window_duration = mfcc_window_duration
        self.read_annotations()

    def read_annotations(self, annotations_format='Events'):
        if annotations_format is 'Old':
            # read annotation files - expected format: CODE START STOP PROB
            data = pd.read_csv(self.path, sep="!", header=None, names=['raw_text'])
            df_tmp = data[data['raw_text'].map(lambda x: is_valid_entry(x))]
            cols = ['Code', 'Start', 'Stop', 'Prob']
            df = pd.DataFrame(columns=cols)
            for i in np.arange(4):
                df[cols[i]] = df_tmp['raw_text'].map(lambda x: str(x).split()[i])
                if i > 0:
                    df[cols[i]] = pd.to_numeric(df[cols[i]])
            self.annotations_df = df
        elif annotations_format is 'Events':
            with open(self.path, 'rb') as f:
                data = loads(f.read())
            assert data
            subject_samples = set([(event.subject, event.sample) for event in data])
            good_pairs = []
            for pair in subject_samples:
                settings_path = join(self.root_dir, pair[0], pair[1])
                with open(settings_path) as f:
                    d = json.load(f)
                assert d
                wav_candidate = join(self.root_dir, pair[0], d.get('url', ''))
                print('{wav_candidate} vs {self.wav_path}')
                print(f'{wav_candidate} vs {self.wav_path}')
                if wav_candidate == self.wav_path or d['url'] == self.wav_path:
                    timestamp_offset = d.get('metadata', {}).get('published_timestamp', 0.0)
                    good_pairs.append(pair)
                    print('success')
            assert len(good_pairs) == 1
            filtered_events = [event for event in data if event.subject is good_pairs[0][0]
                               and event.sample is good_pairs[0][1]]
            df = pd.DataFrame(filtered_events)
            print(f'deduced timestamp offset: {timestamp_offset}')
            df.timestamp_raw -= timestamp_offset
            df[['Start', 'Stop']] = df.apply(lambda row: [row['timestamp_raw'],
                                                                           row['timestamp_raw'] + self.mfcc_window_duration],
                                                              axis=1, result_type='expand')
            self.annotations_df = df
        else:
            raise ValueError('invalid format')

    def view_specific_character(self, character):
        df = self.annotations_df
        return df[df['Code'].map(lambda x: x == character)]

    def view_specific_coords(self, min_x1=-np.inf, min_x2=-np.inf, max_x1=np.inf, max_x2=np.inf):
        df = self.annotations_df
        assert 'x' in df.columns
        return df[df.x.map(lambda v: [max_x1, max_x2] >= list(v) >= [min_x1, min_x2])]

"""
 class TestWavProcessor(unittest.TestCase):

  def setUp(self):
    self.wav_processor = WavProcessor('dummy_path')

  def test_read_wav(self):
    self.wav_processor.path    
"""

if __name__ == '__main__':
    fire.Fire({
        'anno': Annotations,
        'wav': WavProcessor,
        # 'test': unittest.main()
    })
