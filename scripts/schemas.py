from marshmallow import (fields, Schema)


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
    i = fields.Number(validate=lambda t: t >= 0)
    len_t = fields.Number()
    len_freq = fields.Number()
    freq_delta = fields.Number()
    max_f = fields.Number()
    N = fields.Number()
    N_largest_local_max_f = fields.List(fields.Number())
    N_largest_local_max_s = fields.List(fields.Number())


class PhonemesFormantsSchema(Schema):
    formants_info = fields.Nested(FormantsInfoSchema, many=True)
