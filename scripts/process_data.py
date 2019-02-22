import ssl

import fire

from pipeline import Pipeline
import pipelines

ssl._create_default_https_context = ssl._create_stdlib_context


def process_phonemes_pipeline(subjects_homedir='../subjects/', verbose=1):
    pipeline = Pipeline.subclasses['PhonemePipeline'](verbose, subjects_homedir)
    pipeline.process_pipeline()


def process_formants_pipeline(subjects_homedir='../subjects/', verbose=1):
    pipeline = Pipeline.subclasses['FormantsPipeline'](verbose, subjects_homedir)
    pipeline.process_pipeline()


def run_all_pipelines(subjects_homedir='../subjects/', verbose=1):
    print('here')
    print(Pipeline.subclasses)
    for k, v in Pipeline.subclasses.items():
        print('there')
        if verbose > 0:
            print(f'running pipeline {k}')
        v(verbose, subjects_homedir).process_pipeline()


if __name__ == '__main__':
    fire.Fire({
              'run': run_all_pipelines,
              'phonemes': process_phonemes_pipeline,
              'formants': process_formants_pipeline,
              })
