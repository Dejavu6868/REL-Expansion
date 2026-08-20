# Three-arm data contract

This is a read-only readiness audit for a possible future RGBD/HHA/REL+
comparison. It does not authorize or launch that comparison and it does not
change any loader.

The audit reads every manifest row and records:

- RawDepth file count, decode failures, dtype distribution, shape
  distribution and global min/max;
- HHA file count, decode failures, uint8 480 x 480 x 3 compliance and OpenCV
  BGR channel-read behavior.

If RawDepth is uint16, the result is
`RGBD_INPUT_CONTRACT_REQUIRES_DECISION`, because the current source loader's
`cv2.IMREAD_GRAYSCALE` may silently compress the input to uint8. V2.3 does not
silently fix or reinterpret this behavior.

A RawDepth or HHA issue would not block the current CMX-REL+ single arm. It
would block a future three-arm comparison until the input contract is
explicitly resolved.

## Runtime result

The full read-only audit completed with exit code 0:

- RawDepth: 70,496/70,496 decoded, zero failures, all uint8 480 x 480,
  min 0, max 103, `RGBD_INPUT_READY`;
- HHA: 70,496/70,496 decoded, zero failures, all uint8 480 x 480 x 3,
  `HHA_INPUT_READY`;
- current REL+ arm blocked by this audit: false;
- future three-arm input contract blocked by this audit: false.

This readiness result does not authorize a three-arm training run. The runtime
report is stored at
`/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/reports/three_arm_data_contract.json`.
