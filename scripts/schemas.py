
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
