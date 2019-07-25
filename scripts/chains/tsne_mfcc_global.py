import json
import logging
from os.path import join

import pandas as pd

from chains.tsne_mfcc import TsneMfcc
from chains.mfcc_global import MfccGlobal
from schemas import *
logger = logging.getLogger()


class TsneMfccGlobal(TsneMfcc):
    abstract_class = False

    requirements = [MfccGlobal]

    @staticmethod
    def sample_result_filename(out_sample_path):
        return MfccGlobal.filenames_to_skip_sample(out_sample_path)[0]

    @staticmethod
    def sample_filename_prerequisites():
        return [MfccGlobal.sample_result_filename]

    @staticmethod
    def filenames_to_skip_dataset(out_results_dir):
        return [join(out_results_dir, 'global_tsne_input_XY.pickle')]

    @staticmethod
    def get_result_paths_dict(working_dir, filter_hash, n_iter, perplexity, pca_components):
        pca_tsne_pickle_filename = f'pca_tsne_global_mfcc_result_{filter_hash}_{n_iter}_{perplexity}_{pca_components}.pickle'
        ret = {'tsne': join(working_dir, f'tsne_global_mfcc_result_{filter_hash}_{n_iter}_{perplexity}.pickle'),
               'pca': join(working_dir, f'pca_global_mfcc_result_{filter_hash}_{pca_components}.pickle'),
               'pca_tsne': join(working_dir, pca_tsne_pickle_filename)}
        return ret

    def serialized_mfcc_to_csv(self, mfcc_result_path, mfcc_csv_result_path):
        logger.info(f'result_path: {mfcc_csv_result_path}')
        schema = MfccGlobalSchema()
        with open(mfcc_result_path, 'r') as f_csv:
            logger.info(f'mfcc_global_result_path: {mfcc_result_path}')
            json_file = json.load(f_csv)
            result = schema.load(json_file)
            logger.debug(json.dumps(result, indent=4))
            df = pd.DataFrame.from_dict(result['mfcc_global_info'])
            df = df.mfcc.apply(pd.Series).add_suffix('_mfcc_features').merge(df, left_index=True,
                                                                             right_index=True).drop(['mfcc'],
                                                                                                    axis=1)
            # filter nans:
            for c in df.filter(regex='features$', axis=1).columns:
                df = df[df[c].notnull()]
            df.to_csv(mfcc_csv_result_path)
            return True

    @staticmethod
    def labels_timestamps_from_df(df, published_timestamp):
        labels = [None for _ in df.i.values]
        # TODO this has to be fixed, target implementation should take into account sampling rate
        timestamps = [published_timestamp + delta for delta in df.i.values]
        return labels, timestamps