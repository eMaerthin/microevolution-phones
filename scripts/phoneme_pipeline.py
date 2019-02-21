import json
from os.path import (dirname, join)

from marshmallow import pprint
from pocketsphinx.pocketsphinx import Decoder

from audio_processors import download_youtube_url, prepare_wav_input
from decorators import check_if_already_done
from schemas import (PhonemeInfoSchema, PhonemesHypothesisSchema, PhonemesSchema)

from pipeline import Pipeline

MODELDIR = "../pocketsphinx/model"


class PhonemePipeline(Pipeline):

    @staticmethod
    def result_filename(json_path):
        return f'{json_path[:-5]}_result.json'

    @staticmethod
    def filename_prerequisites():
        return []

    def compute_phonemes(self, segments_path, phonemes_result_path,
                         modeldir=MODELDIR):
        @check_if_already_done(phonemes_result_path, self.verbose)
        def recognize_phonemes(segments_path, phonemes_result_path, modeldir):
            # Create a decoder with certain model
            config = Decoder.default_config()
            config.set_string('-hmm', join(modeldir, 'en-us/en-us'))
            config.set_string('-allphone', join(modeldir, 'en-us/en-us-phone.lm.bin'))
            config.set_string('-dict', join(modeldir, 'en-us/cmudict-en-us.dict'))
            config.set_float('-lw', 2.0)
            config.set_float('-pip', 0.3)
            config.set_float('-beam', 1e-200)
            config.set_float('-pbeam', 1e-20)
            config.set_boolean('-mmap', False)
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
                    buf = stream.read(8192)
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

        recognize_phonemes(segments_path, phonemes_result_path, modeldir)

        if self.verbose > 1:
            schema = PhonemesSchema()
            with open(phonemes_result_path, 'r') as f:
                print(f' phonemes_result_path: {phonemes_result_path}')
                json_file = json.load(f)
                result = schema.load(json_file)
                pprint(result, indent=4)

        return phonemes_result_path

    def pipeline(self, series_json_path, series_settings):
        url = series_settings.get('url')
        datatype = series_settings.get('datatype')
        assert(datatype is not None)
        segments = series_settings.get('segments', [{'start': 'begin', 'stop': 'end'}])
        metadata = series_settings.get('metadata')
        lang_code = None
        if isinstance(metadata, dict):
            lang_code = metadata.get('language')
        audio_file_name = f'{series_json_path[:-5]}_audio'
        audio_path, caption_path = download_youtube_url(url, datatype,
                                                        dirname(series_json_path),
                                                        audio_file_name, lang_code,
                                                        self.verbose)
        wav_path, segments_path = prepare_wav_input(audio_path, datatype, segments,
                                                    self.verbose)
        phonemes_result_file = self.result_filename(series_json_path)
        if self.verbose > 0:
            print(f'phonemes result file: {phonemes_result_file}')
        self.compute_phonemes(segments_path, phonemes_result_file)
