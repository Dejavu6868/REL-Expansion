import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools/generate_pilot_cache.py"
SPEC = importlib.util.spec_from_file_location("pilot_cache_tool", str(TOOL))
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_selection_is_exactly_six_per_area_group_and_36_total():
    rows = []
    concrete_areas = (
        "area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6"
    )
    for area_index, area in enumerate(concrete_areas):
        area_group = "area_5" if area.startswith("area_5") else area
        for index in range(10):
            rows.append(
                {
                    "sample_id": "{}/camera_{}_room_{}_frame_{}".format(
                        area, area_index, index, index
                    ),
                    "area": area,
                    "area_group": area_group,
                    "room": "room_{}".format(index),
                    "camera": "{}_{}".format(area_index, index),
                    "status": "PASS",
                    "gravity_alignment_angle_deg": str(index * 10 + area_index),
                    "depth_invalid_ratio": str(index / 100.0),
                    "normal_quality_ratio": str(1.0 - index / 100.0),
                    "floor_ratio": str(index / 20.0),
                    "ceiling_ratio": str((9 - index) / 20.0),
                }
            )
    selected = PILOT.select_pilot(rows)
    assert len(selected) == 36
    assert len({row["sample_id"] for row in selected}) == 36
    for group in PILOT.AREA_GROUPS:
        assert sum(row["selection_area_group"] == group for row in selected) == 6
    area5 = {row["area"] for row in selected if row["selection_area_group"] == "area_5"}
    assert area5 == {"area_5a", "area_5b"}
