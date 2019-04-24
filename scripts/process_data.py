import ssl

import fire

from audio_processors import process_playlist_url
from pipeline import Pipeline

ssl._create_default_https_context = ssl._create_stdlib_context


def process_phonemes_pipeline(subjects_homedir='../subjects/', verbose=1):
    pipeline = Pipeline.subclasses['PhonemePipeline'](verbose, subjects_homedir)
    pipeline.process_pipeline()


def process_formants_pipeline(subjects_homedir='../subjects/', verbose=1):
    pipeline = Pipeline.subclasses['FormantsPipeline'](verbose, subjects_homedir)
    pipeline.process_pipeline()

# TODO Clean up needed!
def run_all_pipelines(subjects_homedir='../subjects/', verbose=1):
    print('here')
    print(Pipeline.subclasses)
    for k, v in Pipeline.subclasses.items():
        print('there')
        if verbose > 0:
            print(f'running pipeline {k}')

        v.verbose = verbose
        v.subjects_dir = subjects_homedir
        v.process_pipeline()

def run_mfcc_pipelines(subjects_homedir='../subjects/', verbose=1):
    print('here')
    print(Pipeline.subclasses)
    for k, v in Pipeline.subclasses.items():
        if 'Mfcc' in k:
            print('there')
            if verbose > 0:
                print(f'running pipeline {k}')
            v.verbose = verbose
            v.subjects_dir = subjects_homedir
            v.process_pipeline()


if __name__ == '__main__':
    fire.Fire({
              'run': run_all_pipelines,
              'mfcc': run_mfcc_pipelines,
              'playlist': process_playlist_url
              # 'phonemes': process_phonemes_pipeline,
              # 'formants': process_formants_pipeline,
              })
