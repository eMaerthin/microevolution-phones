from collections import namedtuple
from marshmallow import (fields, Schema)
from os.path import exists


class SegmentSchema(Schema):
    start = fields.Str()
    stop = fields.Str()


class MetadataSchema(Schema):
    subject = fields.Str()
    language = fields.Str()
    country = fields.Str()
    state = fields.Str()
    profession = fields.Str()
    gender = fields.Str()
    birth_year = fields.Integer()
    list_filename = fields.Str()
    playlist_url = fields.Str()
    subject_age = fields.Integer()
    intro_duration = fields.Number()
    outro_duration = fields.Number()
    published_date = fields.Str()
    published_timestamp = fields.Float()
    recorded_length = fields.Number()


class SampleSchema(Schema):
    url = fields.Str()
    datatype = fields.Str()
    record_date = fields.Str()
    segments = fields.Nested(SegmentSchema, many=True)
    metadata = fields.Nested(MetadataSchema)


class PocketsphinxHypothesisSchema(Schema):
    best_score = fields.Number()
    hypstr = fields.String()
    prob = fields.Number()


class PocketsphinxSegmentSchema(Schema):
    word = fields.String()
    start = fields.Number()
    end = fields.Number()
    prob = fields.Number()


class DecoderOutputSchema(Schema):
    hypotheses = fields.Nested(PocketsphinxHypothesisSchema, many=True)
    segment_info = fields.Nested(PocketsphinxSegmentSchema, many=True)


class FormantsInfoSchema(PocketsphinxSegmentSchema):
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


class SpectrogramInfoSchema(PocketsphinxSegmentSchema):
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


class MfccInfoSchema(PocketsphinxSegmentSchema):
    i = fields.Number(validate=lambda i: i >= 0)
    length = fields.Number(validate=lambda length: length >= 0)
    mfcc = fields.List(fields.Number())


class MfccLocalSchema(Schema):
    mfcc_info = fields.Nested(MfccInfoSchema, many=True)


class SingleMfccGlobalSchema(Schema):
    i = fields.Number(validate=lambda i: i >= 0)
    mfcc = fields.List(fields.Number())


class MfccGlobalSchema(Schema):  # TODO(marcin) equip it with metadata, now it seems to be not useful!
    mfcc_global_info = fields.Nested(SingleMfccGlobalSchema, many=True)


class ChainRunnerSettingsSchema(Schema):
    dataset_home_dir = fields.String(validate=lambda home_dir: exists(home_dir))
    process_settings = fields.Dict()
    results_identifier = fields.String()
    subjects_pattern = fields.List(fields.String())


class EventSchema(Schema):
    """
    x - features
    label - word.value (usually phoneme e.g. 'AH' or word e.g. 'car')
    timestamp - timestamp preprocessed to [0,1] range
    timestamp_raw - raw timestamp based on date when video was published
    subject - identifier of the subject
    sample - identifier of the sample
    """
    x = fields.List(fields.Number())
    label = fields.Str()
    timestamp = fields.Float()
    timestamp_raw = fields.Float()
    subject = fields.Str()
    sample = fields.Str()


class ConverterSchema(Schema):
    """
    subject - identifier of the subject
    timestamp_min - minimal timestamp from events for a given subject
    timestamp_max - minimal timestamp from events for a given subject
    time_scale - factor used to compute preprocessed timestamps equal to (timestamp_max - timestamp_min)
    color - fixed colour based on the subject
    color_labels - fixed list of colours based on the labels
    """
    subject = fields.Str()
    timestamp_min = fields.Float()
    timestamp_max = fields.Float()
    time_scale = fields.Float()
    color = fields.Str()
    color_labels = fields.List(fields.Str())


class SubjectWordSchema(Schema):
    word = fields.Str()


def generate_namedtuple(name, schema):
    return namedtuple(name, schema.__dict__['_declared_fields'].keys())


DecoderOutput = generate_namedtuple('DecoderOutput', DecoderOutputSchema)
SubjectWord = generate_namedtuple('SubjectWord', SubjectWordSchema)
Converter = generate_namedtuple('Converter', ConverterSchema)
Event = generate_namedtuple('Event', EventSchema)

