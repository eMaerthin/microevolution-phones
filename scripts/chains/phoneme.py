import json
from os.path import (basename, dirname, join)

from marshmallow import pprint
from pocketsphinx.pocketsphinx import Decoder

from audio_processors import download_youtube_url, prepare_wav_input
from decorators import check_if_already_done
from schemas import (PhonemeInfoSchema, PhonemesHypothesisSchema, PhonemesSchema)

from chain import Chain

MODEL_DIR = "../thirdparty/pocketsphinx/model"


class Phoneme(Chain):

    requirements = []

    @staticmethod
    def sample_result_filename(sample):
        return f'{sample[:-5]}_phoneme_result.json'

    def _compute_phonemes(self, segments_path, phonemes_result_path):
        """

        :param segments_path:
        :param phonemes_result_path:
        :return:
        """
        model_dir = self.process_settings.get('model_dir', MODEL_DIR)
        decoder_hmm = self.process_settings.get('decoder_hmm', 'en-us/en-us')
        decoder_allphone = self.process_settings.get('decoder_allphone',
                                                     'en-us/en-us-phone.lm.bin')
        decoder_dict = self.process_settings.get('decoder_dict',
                                                 'en-us/cmudict-en-us.dict')
        decoder_lw = self.process_settings.get('decoder_lw', 2.0)
        decoder_pip = self.process_settings.get('decoder_pip', 0.3)
        decoder_beam = self.process_settings.get('decoder_beam', 1e-200)
        decoder_pbeam = self.process_settings.get('decoder_pbeam', 1e-20)
        decoder_mmap = self.process_settings.get('decoder_mmap', False)
        decoder_stream_buf_size = self.process_settings.get('decoder_stream_buf_size',
                                                            8192)
        pprint_indent = self.process_settings.get('pprint_indent', 4)

        @check_if_already_done(phonemes_result_path, self.verbose)
        def recognize_phonemes(segments_path, phonemes_result_path, model_dir):

            # Create a decoder with certain model
            config = Decoder.default_config()
            config.set_string('-hmm', join(model_dir, decoder_hmm))
            config.set_string('-allphone', join(model_dir, decoder_allphone))
            config.set_string('-dict', join(model_dir, decoder_dict))
            config.set_float('-lw', decoder_lw)
            config.set_float('-pip', decoder_pip)
            config.set_float('-beam', decoder_beam)
            config.set_float('-pbeam', decoder_pbeam)
            config.set_boolean('-mmap', decoder_mmap)
            hyps=[]
            segs=[]
            decoder = Decoder(config)
            with open(segments_path, 'rb') as stream:
                in_speech_buffer = False
                decoder.start_utt()
                while True:
                    ph_info = PhonemeInfoSchema()
                    phonemes = PhonemesSchema()
                    hypothesis = PhonemesHypothesisSchema()
                    buf = stream.read(decoder_stream_buf_size)
                    if buf:
                        decoder.process_raw(buf, False, False)
                        if decoder.get_in_speech() != in_speech_buffer:
                            in_speech_buffer = decoder.get_in_speech()
                            if not in_speech_buffer:
                                decoder.end_utt()
                                segs += [ph_info.dump(dict(word=seg.word,
                                                           start=seg.start_frame/100,
                                                           end=seg.end_frame/100,
                                                           prob=seg.prob))
                                         for seg in decoder.seg()]
                                hyp = decoder.hyp()
                                hyp_dict = dict(best_score=hyp.best_score,
                                                hypstr=hyp.hypstr, prob=hyp.prob)
                                hyp_result = hypothesis.dump(hyp_dict)
                                hyps.append(hyp_result)
                                decoder.start_utt()
                    else:
                        if in_speech_buffer:
                            decoder.end_utt()
                            segs += [ph_info.dump(dict(word=seg.word,
                                                       start=seg.start_frame/100,
                                                       end=seg.end_frame/100,
                                                       prob=seg.prob))
                                     for seg in decoder.seg()]
                            hyp = decoder.hyp()
                            hyp_dict = dict(best_score=hyp.best_score,
                                            hypstr=hyp.hypstr, prob=hyp.prob)
                            hyp_result = hypothesis.dump(hyp_dict)
                            hyps.append(hyp_result)
                        break
            phonemes_dict = dict(hypotheses=hyps, info=segs)
            phonemes_result = phonemes.dumps(phonemes_dict)
            with open(phonemes_result_path, 'w') as f:
                f.write(phonemes_result)

        recognize_phonemes(segments_path, phonemes_result_path)

        if self.verbose > 1:
            schema = PhonemesSchema()
            with open(phonemes_result_path, 'r') as f:
                print(f'[DETAILS] phonemes_result_path: {phonemes_result_path}')
                json_file = json.load(f)
                result = schema.load(json_file)
                pprint(result, indent=pprint_indent)

    def sample_layer(self, subject, sample_json_filename, settings):
        url = settings.get('url')
        datatype = settings.get('datatype')
        assert(datatype is not None)
        segments = settings.get('segments', [{'start': 'begin', 'stop': 'end'}])
        metadata = settings.get('metadata')
        lang_code = None
        if isinstance(metadata, dict):
            lang_code = metadata.get('language')
        output_path_pattern = join(self.results_dir, subject, sample_json_filename)
        audio_file_name = basename(f'{output_path_pattern[:-5]}_audio')
        audio_path, caption_path = download_youtube_url(url, datatype,
                                                        dirname(output_path_pattern),
                                                        audio_file_name, lang_code,
                                                        self.verbose)
        wav_path, segments_path = prepare_wav_input(audio_path, datatype, segments,
                                                    self.verbose)
        phonemes_result_file = self.sample_result_filename(output_path_pattern)
        if self.verbose > 0:
            print(f'[INFO] phonemes result file: {phonemes_result_file}')
        self._compute_phonemes(segments_path, phonemes_result_file)
