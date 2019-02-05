import fire
from functools import reduce
# import unittest

import numpy as np
import pandas as pd
from scipy.io.wavfile import read as read_wav, write as write_wav

class WavProcessor(object):
  """A simple WavProcessor class."""

  def __init__(self, path):
    self.path = path
    self.read_path()

  def read_path(self):
    rate,data = read_wav(self.path)
    self.rate = rate
    self.data = data

  def save_wav(self, out_path, data):
    write_wav(out_path, self.rate, data)

  def cut_one_chunk(self, start_sec, end_sec):
    start, end = map(lambda x: int(x * self.rate), [start_sec, end_sec])
    return self.data[start:end]
  
  def cut_multiple_chunks(self, starts, stops, wait_ms=0):
    assert len(starts) == len(stops)
    chunks = [self.cut_one_chunk(start, stop) 
              for (start, stop) in zip(starts,stops)]
    wait_chunk = np.zeros(int(wait_ms * self.rate / 1000.0))
    return reduce(lambda x, y: np.concatenate((x, wait_chunk, y)), chunks)
  
  def save_specific_character(self, character, annotation_file=None, 
                              save_path=None, wait_ms=0):
    if annotation_file is None:
      annotation_file = self.path+"_out_log.txt"
    if save_path is None:
      save_path = self.path+"_test.wav"
    annotations = Annotations(annotation_file)
    view_df = annotations.view_specific_character(character)
    starts,stops = map(lambda x: view_df[x].values.tolist(), ['Start', 'Stop'])
    out_data = self.cut_multiple_chunks(starts, stops, wait_ms)
    self.save_wav(save_path, out_data)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def is_valid_entry(obj):
  words = str(obj).split()
  if len(words)<4:
    return False
  return isfloat(words[1])

class Annotations(object):
  """A simple Annotations class."""
  def __init__(self, anno_path):
    self.path = anno_path
    self.read_annotations()

  def read_annotations(self):
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

  def view_specific_character(self, character):
    df = self.annotations_df
    return df[df['Code'].map(lambda x: x==character)]

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
