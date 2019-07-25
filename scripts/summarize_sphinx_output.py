import logging
import sys
import numpy as np
import pandas as pd
logger = logging.getLogger()

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


if __name__ == '__main__':
    from logging.config import fileConfig
    fileConfig('logging.conf')

    if len(sys.argv)<2:
      raise ValueError('error!')

    # Load data
    data = pd.read_csv(sys.argv[1], sep=',', header=None, names=['text'])
    df = data[data['text'].map(lambda x: is_valid_entry(x))]
    df.index=np.arange(len(df))
    column_names = ['phones', 'start_time', 'stop_time', 'column4','duration']
    new_df = pd.DataFrame(index=df.index, columns=column_names)
    # new_df[column_names[0]] = df['text'].map(lambda x: str(x).split()[0])
    for i in np.arange(4):
      new_df[column_names[i]] = df['text'].map(lambda x: str(x).split()[i])
    new_df[column_names[4]] = new_df.apply(lambda x: float(x.stop_time)-float(x.start_time), axis=1)
    total_recording_time = new_df.stop_time.astype(float).max()
    # DONE: preprocess data now:
    # v delete long phones (duration > duration_threshold) - assuming those are fp
    # v remove special phones of the form +phone_id+ for instance +BREATH+
    # v add more meaningful statistics, for instance mean and stdev of the duration
    max_duration = 1
    new_df = new_df[new_df['duration'].map(lambda x: float(x)<max_duration)]

    new_df = new_df[new_df['start_time'].map(lambda x: float(x)>300.0)]

    new_df = new_df[new_df['stop_time'].map(lambda x: float(x)<3500.0)]

    # removing special characters +SOMETHING+ and SIL:
    new_df = new_df[new_df['phones'].map(lambda x: (x.find("+") == -1) & (x != "SIL"))]

    # only A or E or I or U or O or Y
    new_df = new_df[new_df['phones'].map(lambda x: (x.find("A") != -1) | (x.find("E") != -1) | (x.find("I") != -1) | (x.find("U") != -1) | (x.find("O") != -1) | (x.find("Y") != -1))]

    # the histogram can be built from counting unique start_time - this is unique field from the very beginning of the definition
    x = new_df.groupby('phones')['start_time'].nunique()
    duration_group = new_df.groupby('phones')['duration']
    y1 = duration_group.sum()
    y2 = duration_group.mean()
    y3 = duration_group.agg(np.std, ddof=0)
    df_results = pd.DataFrame(data={'phones_histogram': x, 'phones_histogram_freq': x/x.sum(), 'phones_duration_total': y1, 'phones_duration_mean': y2, 'phones_duration_std': y3},
                              dtype=float,
                              index=x.index,
                              columns={'phones_histogram', 'phones_histogram_freq', 'phones_duration_total', 'phones_duration_mean', 'phones_duration_std'})
    logger.info(f'Total time: {total_recording_time}s.')
    logger.info(f'Total time covered by meaningful phonemes: {df_results.phones_duration_total.sum()}s.')
    logger.info(f'Total number of phonemes: {df_results.phones_histogram.sum()}.')
    logger.info(f"Results: {df_results[['phones_histogram_freq']]}")  #.sort_values(by='phones', ascending=False))#.head(n=20))
