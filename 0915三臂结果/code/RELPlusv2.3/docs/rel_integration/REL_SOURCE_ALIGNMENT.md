# REL source alignment

## Sources inspected

- Official CMX: `/home/zhuzhaoziao/RELPlus/CMX`, official remote
  `https://github.com/huaaaliu/RGBX_Semantic_Segmentation.git`, branch `main`,
  clean at inspection. Latest commit date/title: `2024-09-02 22:15:42 +0800` /
  `Update download link for ZJU-RGB-P`.
- Locally verified ERP-REL: `/home/zhuzhaoziao/RELPlus/REL`.
- Public REL reference: `/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference`,
  remote `https://github.com/SrtaEstrella/REL-SF4PASS.git`, branch `main`, clean
  at inspection. Latest commit date/title: `2026-06-01 09:58:29 +0800` /
  `Remove unused _gitignore template`.
- Integrated copy: `third_party/rel_original` in this repository.

The public reference imports `utils/hha_util.py` but does not contain that
file. Both the already verified local implementation and this integration use
`/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py` for the called filter, matrix and
gravity helpers. This compatibility source is not presented as an
author-tracked file.

## Executed source alignment

AST function-body comparison passed for `getImage`, `getREL`,
`processDepthImage_ERP`, `getPointCloud_ERP`,
`computeNormalsSquareSupport_ERP`, `filterItChopOff`, `mutiplyIt`, `invertIt`,
`getRMatrix`, `rotatePC`, `getGDir` and `getGDirHelper`. Three real ERP images
also produced arrays exactly equal to the public executable path with
`np.array_equal`.

## Semantic comparison

| Item | Paper definition supplied for this task | Public code behavior | Local verified and integrated behavior | Conclusion |
|---|---|---|---|---|
| Input | Complete ERP depth | `uint16` PNG is read, incremented in place, converted to `float32`, divided by 512 | Same | Executable paths agree |
| Invalid depth | Not fully specified | Raw `65535` wraps to zero during the in-place increment and becomes the missing mask | Same | Executable paths agree; retained unchanged |
| ERP point cloud | Cylindrical/spherical geometry from full panorama | Longitude/latitude rays produce `x,y,z`; no pinhole camera parameters | Same | Agree |
| ReD | `sqrt(px^2 + py^2)` | `hypot(pcRot_x, pcRot_y)`, then per-image min/max encoding | Same | Geometric quantity agrees; code adds per-image encoding |
| Height | `pz - min(Pz)` | `pcRot_z` normalized with the 1st and 99th percentiles | Same | Paper summary and code differ |
| Vertical angle | `arccos(N dot G)` | After gravity rotation, `arccos(-NRot_z)` encoded to 0–255 | Same | Equivalent under the code coordinate convention |
| EGVIA condition | Blend angle and height near horizontal surfaces | Marks near-horizontal pixels, but blends only `~is_horizontal` | Same | Paper summary and executable code differ |
| EGVIA parameters | `alpha=45`, `lambda=0.5` | Defaults 45 and 0.5 | Same | Agree |
| LOA | ERP longitude and tangent/normal relation | `phi=(u/w)2pi-pi`; clipped dot expression followed by `arccos` | Same | Executable paths agree |
| Array order | Three channels ReD/EGVIA/LOA; order not established by the prose alone | `stack([angle, HA, RD])` | `[EGVIA, LOA, ReD]` | Executable order established |
| Saved/reloaded order | Must be traced | `imwrite` then `IMREAD_UNCHANGED` preserves array indices | Same | Exact round trip passed |

## Decision

The integration deliberately preserves the already verified local and public
end-to-end behavior. It does not reverse the EGVIA condition or replace the
height percentiles. By project decision on 2026-08-16, this executable behavior
is the REL implementation standard for future generation, training and
evaluation. The paper-prose versus executable-code difference remains recorded
and the two definitions must not be called identical, but it no longer blocks
the project's representation definition. See `REL_IMPLEMENTATION_STANDARD.md`.

No perspective intrinsics, extrinsics, FoV, pose, crop-aware camera model or
REL+ floor reference is present in the called path.
