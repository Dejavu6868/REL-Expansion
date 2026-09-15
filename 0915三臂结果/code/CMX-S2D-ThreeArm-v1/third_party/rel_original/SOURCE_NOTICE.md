# Original panoramic REL source notice

The numerical REL path in this directory was extracted from the public
`SrtaEstrella/REL-SF4PASS` implementation:

- `rel.py`: `getImage` and `getREL` from `getREL.py`.
- `rgbd_util.py`: the called ERP point-cloud, normal and gravity functions
  from `utils/rgbd_util.py`.
- `hha_util.py`: the compatibility helper from
  `/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py`, because the inspected public
  repository imports that module but does not contain it.

Only package-relative imports and removal of unrelated encodings or dataset
loops differ from the source files. The REL numerical statements and their
order are unchanged. The included `LICENSE` covers the copied public code.

This directory contains original ERP REL only. It does not contain REL+,
perspective intrinsics or extrinsics, Gate, SMMF, DyMM, region slicing, or a
training-time REL generator.
