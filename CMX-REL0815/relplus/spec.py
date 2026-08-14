"""Machine-readable identity for the REL-default perspective adaptation."""

import hashlib
import json


RELPLUS_SPEC = {
    "model_name": "cmx_rel+",
    "config_name": "cmx_rel+",
    "representation_semantics": "REL-default",
    "representation_version": "relplus_rel_default_v2",
    "point_frame": "camera_centered_world_axes",
    "translation_in_red_loa": False,
    "channel_order": ["ReD", "EGVIA", "LOA"],
    "depth_definition": "camera-z; metres=(uint16+1)/512; 65535 invalid",
    "pixel_origin": 1,
    "normal_estimator": "REL square-support algebraic plane fit",
    "normal_radius_native_pixels": 3,
    "alpha_degrees": 45.0,
    "lambda": 0.5,
    "red_height_normalization": "valid-image min-max to uint8",
    "angle_normalization": "zero-to-pi linearly mapped to uint8",
    "invalid_pixel": [255, 255, 255],
    "native_depth_shape": [1080, 1080],
    "cache_shape": [480, 480, 3],
    "cache_resize": "bilinear channels; nearest validity mask",
    "output_dtype": "uint8",
    "intrinsics_usage": "K backprojects camera-z depth into camera coordinates",
    "extrinsics_usage": (
        "R rotates points and normals into world axes; t/C are validated and retained "
        "for provenance but do not enter ReD, EGVIA, or LOA"
    ),
}


def canonical_spec_json():
    return json.dumps(RELPLUS_SPEC, sort_keys=True, separators=(",", ":")) + "\n"


RELPLUS_SPEC_SHA256 = hashlib.sha256(canonical_spec_json().encode("utf-8")).hexdigest()
