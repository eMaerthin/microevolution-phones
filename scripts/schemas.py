from marshmallow import (fields, Schema)
from os.path import exists

class SegmentSchema(Schema):
    start = fields.Str()
    stop = fields.Str()


class MetadataSchema(Schema):
    subject = fields.Str()
    language = fields.Str()
    country = fields.Str()
    profession = fields.Str()
    gender = fields.Str()
    age = fields.Integer()


class SeriesSchema(Schema):
    url = fields.Str()
    datatype = fields.Str()
    record_date = fields.Str()
    segments = fields.Nested(SegmentSchema, many=True)
    metadata = fields.Nested(MetadataSchema)


class PhonemesHypothesisSchema(Schema):
    best_score = fields.Number()
    hypstr = fields.String()
    prob = fields.Number()


class PhonemeInfoSchema(Schema):
    word = fields.String()
    start = fields.Number()
    end = fields.Number()
    prob = fields.Number()


class PhonemesSchema(Schema):
    hypotheses = fields.Nested(PhonemesHypothesisSchema, many=True)
    info = fields.Nested(PhonemeInfoSchema, many=True)


class FormantsInfoSchema(PhonemeInfoSchema):
    t = fields.Number(validate=lambda t: t >= 0)
    i = fields.Number(validate=lambda i: i >= 0)
    len_t = fields.Number()
    len_freq = fields.Number()
    freq_delta = fields.Number()
    max_f = fields.Number()
    N = fields.Number()
    N_largest_local_max_f = fields.List(fields.Number())
    N_largest_local_max_s = fields.List(fields.Number())


class PhonemesFormantsSchema(Schema):
    formants_info = fields.Nested(FormantsInfoSchema, many=True)


class SpectrogramInfoSchema(PhonemeInfoSchema):
    t = fields.Number(validate=lambda t: t >= 0)
    i = fields.Number(validate=lambda i: i >= 0)
    len_t = fields.Number()
    len_freq = fields.Number()
    freq_delta = fields.Number()
    N = fields.Number()
    # frequency = fields.List(fields.Number(validate=lambda f: f >= 0))
    signal = fields.List(fields.Number(validate=lambda s: s >= 0))
    # normalized_signal = fields.List(fields.Number(validate=lambda n: n >= 0))


class PhonemesSpectrogramsSchema(Schema):
    spectrograms_info = fields.Nested(SpectrogramInfoSchema, many=True)


class MfccInfoSchema(PhonemeInfoSchema):
    i = fields.Number(validate=lambda i: i >= 0)
    length = fields.Number(validate=lambda l: l >= 0)
    mfcc = fields.List(fields.Number())


class MfccSchema(Schema):
    mfcc_info = fields.Nested(MfccInfoSchema, many=True)


class SingleMfccGlobalSchema(Schema):
    i = fields.Number(validate=lambda i: i >= 0)
    mfcc = fields.List(fields.Number())


class MfccGlobalSchema(Schema):
    mfcc_global_info = fields.Nested(SingleMfccGlobalSchema, many=True)


class ChainRunnerSettingsSchema(Schema):
    dataset_home_dir = fields.String(validate=lambda home_dir: exists(home_dir))
    process_settings = fields.Dict()
    results_identifier = fields.String()
    verbose = fields.Number(validate=lambda i: all((i >= 0, i <= 2)))
