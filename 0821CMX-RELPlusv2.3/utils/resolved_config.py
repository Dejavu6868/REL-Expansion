"""Apply a launcher-produced, narrowly scoped V2.3 runtime config overlay."""

import json
from pathlib import Path


ALLOWED_RUNTIME_FIELDS = {
    "training_authorized",
    "source_compatible_invalid_accepted",
    "formal_cache_root",
    "x_root_folder",
    "x_valid_root_folder",
    "full_manifest",
    "cache_generation_report",
    "generation_resolved_manifest_path",
    "resolved_manifest_path",
    "train_source",
    "eval_source",
    "class_mapping",
    "cache_audit_report",
    "training_data_preflight_report",
    "ddp_smoke_report",
    "output_dir",
    "log_dir",
    "tb_dir",
    "log_dir_link",
    "checkpoint_dir",
    "log_file",
    "link_log_file",
    "val_log_file",
    "link_val_log_file",
    "launch_id",
}


def apply_resolved_config(config, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("integration_protocol_id") != config.integration_protocol_id:
        raise RuntimeError("resolved config integration protocol mismatch")
    if payload.get("representation_protocol_id") != config.representation_protocol_id:
        raise RuntimeError("resolved config representation protocol mismatch")
    overrides = payload.get("runtime_overrides")
    if not isinstance(overrides, dict):
        raise RuntimeError("resolved config runtime_overrides must be an object")
    unexpected = sorted(set(overrides) - ALLOWED_RUNTIME_FIELDS)
    if unexpected:
        raise RuntimeError("resolved config contains forbidden fields: {}".format(unexpected))
    if overrides.get("training_authorized") is not True:
        raise RuntimeError("resolved config must explicitly authorize formal training")
    if overrides.get("source_compatible_invalid_accepted") is not True:
        raise RuntimeError("resolved config must explicitly accept SOURCE_COMPAT_STORAGE_255")
    for field, value in overrides.items():
        setattr(config, field, value)
    return payload
