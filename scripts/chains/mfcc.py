from abc import abstractmethod
import logging

from chains.labeled_base import LabeledBase
from chains.phoneme import Phoneme
from chains.words import Words
from decorators import check_if_already_done

logger = logging.getLogger()


class Mfcc(LabeledBase):
    """
    Abstract class for sharing common logic of MfccLocal
    and MfccGlobal chains.

    """

    abstract_class = True

    @staticmethod
    @abstractmethod
    def serialize_to_json(mfcc_result):
        """
        :param mfcc_result: list of mfcc measurements with
        necessary metadata
        :return: serialized object of proper schema
        """
        pass

    @abstractmethod
    def compute_mfcc(self, segments_path, labels_result_path):
        """

        :param segments_path: path to the input wav
        :param labels_result_path: path to phonemes/words results
        that is required by the Local version of the Mfcc
        :return: computed list of mfcc features with all required metadata
        """
        pass

    def _compute_mfcc(self, segments_path, labels_result_path, mfcc_result_path):

        @check_if_already_done(mfcc_result_path, validator=lambda x: x)
        def store_mfcc(segments_path, labels_path, mfcc_result_path):
            mfcc_result = self.compute_mfcc(segments_path, labels_path)
            result = self.serialize_to_json(mfcc_result)
            with open(mfcc_result_path, 'w') as result_f:
                result_f.write(result)
                return True

        store_mfcc(segments_path, labels_result_path, mfcc_result_path)

    def compute_target(self, segments_path, labels_path, output_path_pattern):
        mfcc_result_path = self.sample_result_filename(output_path_pattern)
        self._compute_mfcc(segments_path, labels_path, mfcc_result_path)
        logger.info(f'mfcc result path: {mfcc_result_path}')
