"""Frozen public identity of REL+ v2.1."""

REL_PLUS_V2_ALPHA = 45.0
REL_PLUS_V2_LAMBDA = 0.5
REL_PLUS_V2_NORMAL_RADIUS = 2
REL_PLUS_V2_CANONICAL_SHAPE = (480, 480)
REL_PLUS_V2_CHANNEL_ORDER = ("EGVIA", "LOA", "ReD")
REL_PLUS_V2_INVALID_TRIPLET = (255, 255, 255)
REL_PLUS_V2_1_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
REL_PLUS_V2_1_INVALID_POLICY = "SOURCE_COMPAT_STORAGE_255"

# The source normal fit needs at least three valid points. This threshold is
# diagnostic only and never changes the formal depth-derived encoding mask.
REL_PLUS_V2_MIN_NORMAL_SUPPORT = 3

# The original getRMatrix has no unique yaw-preserving answer at 180 degrees.
GRAVITY_ANTIPARALLEL_THRESHOLD_DEGREES = 179.999
