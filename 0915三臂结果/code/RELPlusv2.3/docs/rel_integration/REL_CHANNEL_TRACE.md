# REL channel trace

## End-to-end order

| Stage | Array channel 0 | Array channel 1 | Array channel 2 |
|---|---|---|---|
| REL core return | EGVIA (`angle`) | LOA (`HA`) | ReD (`RD`) |
| Before `cv2.imwrite` | EGVIA | LOA | ReD |
| `cv2.imread(..., IMREAD_UNCHANGED)` | EGVIA | LOA | ReD |
| CMX Dataset before normalize | EGVIA | LOA | ReD |
| Tensor after transpose | EGVIA | LOA | ReD |

OpenCV writes a three-channel array using BGR convention. Therefore an
RGB-oriented view of the PNG has `R=ReD`, `G=LOA`, `B=EGVIA`. Reloading with
`IMREAD_UNCHANGED` restores the original OpenCV array indices. CMX-REL does not
apply `BGR2RGB` to REL, because doing so would reverse channel 0 and channel 2
relative to the author code's actual array path.

This follows OpenCV's documented distinction between image-read flags and
color conversion: `IMREAD_UNCHANGED` loads the stored channels as-is, while
color conversion is a separate `cvtColor` operation:

- https://docs.opencv.org/4.9.0/d8/d6a/group__imgcodecs__flags.html
- https://docs.opencv.org/trunk/d8/d01/group__imgproc__color__conversions.html

The network therefore receives:

```text
tensor[:, 0] = EGVIA
tensor[:, 1] = LOA
tensor[:, 2] = ReD
```

## Three real S3D samples

All arrays below are `uint8`, shape `2048 x 4096` per channel, range 0–255,
with zero NaN, zero Inf and no constant channel.

| Sample | Channel | Meaning | Mean | Std |
|---|---:|---|---:|---:|
| s3d_01 | 0 | EGVIA | 130.165976 | 106.245808 |
| s3d_01 | 1 | LOA | 89.239364 | 20.169099 |
| s3d_01 | 2 | ReD | 36.597131 | 40.272937 |
| s3d_02 | 0 | EGVIA | 129.036670 | 105.778058 |
| s3d_02 | 1 | LOA | 89.524320 | 21.316080 |
| s3d_02 | 2 | ReD | 33.462380 | 40.708483 |
| s3d_03 | 0 | EGVIA | 131.048577 | 95.227739 |
| s3d_03 | 1 | LOA | 92.200988 | 27.203901 |
| s3d_03 | 2 | ReD | 28.163575 | 28.235621 |

For the actual model batch after official CMX normalization:

| Tensor channel | Meaning | Min | Max | Mean | Std |
|---:|---|---:|---:|---:|---:|
| 0 | EGVIA | -2.100779 | 2.231784 | 0.997876 | 1.201367 |
| 1 | LOA | -2.035714 | 1.098039 | -0.248211 | 0.431698 |
| 2 | ReD | -1.089847 | 1.716253 | -0.271253 | 0.609151 |

Detailed machine-readable results and visualizations are at:

```text
/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests/smoke_results.json
/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests/channel_visualization/
```
