from abc import abstractmethod
import logging


from decorators import check_if_already_done
from chains.formants import Formants
logger = logging.getLogger()


class Mfcc(Formants):
    """
    Mfcc -> Local computes Mfcc features for each phoneme from the sample
    that are not blacklisted based on phoneme label that is
    received from Phoneme chain.

    Mfcc -> Global computes Mfcc features for the whole sample
    (based on segments wav) - i.e. without using information
    about phoneme label that can be received from Phoneme chain


    It subclasses Formants to not repeat the sample_layer logic
    which is valid also in this context
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
    def compute_mfcc(self, segments_path, phonemes_result_path):
        """

        :param segments_path: path to the input wav
        :param phonemes_result_path: path to phonemes results
        that is required by the Local version of the Mfcc
        :return: computed list of mfcc features with all required metadata
        """
        pass

    def _compute_mfcc(self, segments_path, phonemes_result_path, mfcc_result_path):

        @check_if_already_done(mfcc_result_path, validator=lambda x: x)
        def store_mfcc(segments_path, phonemes_result_path, mfcc_result_path):
            mfcc_result = self.compute_mfcc(segments_path, phonemes_result_path)
            result = self.serialize_to_json(mfcc_result)
            with open(mfcc_result_path, 'w') as result_f:
                result_f.write(result)
                return True

        store_mfcc(segments_path, phonemes_result_path, mfcc_result_path)

    def compute_target(self, segments_path, phonemes_path, output_path_pattern):
        mfcc_result_path = self.sample_result_filename(output_path_pattern)
        self._compute_mfcc(segments_path, phonemes_path, mfcc_result_path)
        logger.info(f'mfcc result path: {mfcc_result_path}')
