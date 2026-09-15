# Vendored REL Source Notice

`third_party/rel_original/hha_util.py` is kept byte-identical to the frozen
REL+ v2.1 integration source. Its non-constant-superpixel branch is historical
code and is not used by the formal REL/REL+ path, which supplies an all-ones
superpixel array.

Do not reuse that helper for a different HHA-generation task without a separate
source and behavior audit. This integration neither repairs nor generalizes the
unused branch.

The constant-superpixel path was tested directly against the live frozen
`hha_util.py` and the current author `utils/rgbd_util.py`; the live-source test
passed without modifying the vendored file.
