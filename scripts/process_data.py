
import ssl

import fire

from formants_pipeline import FormantsPipeline
from phoneme_pipeline import PhonemePipeline

ssl._create_default_https_context = ssl._create_stdlib_context


def process_phonemes_pipeline(subjects_homedir = '../subjects/', verbose = 1):
    pipeline = PhonemePipeline(verbose, subjects_homedir)
    pipeline.process_pipeline()


def process_formants_pipeline(subjects_homedir = '../subjects/', verbose = 1):
    pipeline = FormantsPipeline(verbose, subjects_homedir)
    pipeline.process_pipeline()


def run_all_pipelines(subjects_homedir = '../subjects/', verbose = 1):
    process_phonemes_pipeline(subjects_homedir, verbose)
    process_formants_pipeline(subjects_homedir, verbose)


if __name__ == '__main__':
    fire.Fire({
              'run': run_all_pipelines,
              'phonemes': process_phonemes_pipeline,
              'formants': process_formants_pipeline,
              })
