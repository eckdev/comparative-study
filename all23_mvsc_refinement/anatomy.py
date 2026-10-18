"""Landmark types and anatomical groups for the single all-23 local refiner."""

from __future__ import annotations

from all23_rgb_geodesic_cascade.anatomy import (
    CORE20,
    HARD3,
    NUM_LANDMARKS,
    mirror_permutation,
)


ALL23_INDICES = tuple(range(NUM_LANDMARKS))

# Landmark type decides the scoring head, the search radius and the jitter used
# for training. Types follow the anatomical definition, not observed difficulty:
# contour/extremal points are defined by a broad surface outline (chin, angle),
# the hairline point by a texture boundary, all others by a local corner or tip.
LANDMARK_TYPES = {
    "corner": (1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20),
    "contour": (10, 11, 12, 21, 22),
    "texture_boundary": (0,),
}
TYPE_NAMES = tuple(LANDMARK_TYPES)
LANDMARK_TO_TYPE = {
    landmark: type_index
    for type_index, landmarks in enumerate(LANDMARK_TYPES.values())
    for landmark in landmarks
}

# Anatomical groups condition the heads through an embedding and are used for
# reporting. The six Core20 groups are unchanged from Core20-MVSC.
ANATOMICAL_GROUPS = {
    "hairline": (0,),
    "upper_midline": (1, 2),
    "nasal_oral_midline": (3, 4, 5, 6, 7, 8, 9),
    "chin_contour": (10, 11, 12),
    "ocular": (13, 14, 15, 16),
    "alar": (17, 18),
    "commissure": (19, 20),
    "mandibular_angle": (21, 22),
}
GROUP_NAMES = tuple(ANATOMICAL_GROUPS)
LANDMARK_TO_GROUP = {
    landmark: group_index
    for group_index, landmarks in enumerate(ANATOMICAL_GROUPS.values())
    for landmark in landmarks
}

# Training-only anatomical consistency anchors; the target is never its own anchor.
ANCHORS = {
    0: (1, 2, 13, 16),
    1: (2, 13, 16),
    2: (1, 14, 15),
    3: (2, 5, 17, 18),
    4: (3, 5, 17, 18),
    5: (3, 4, 6),
    6: (5, 7, 19, 20),
    7: (6, 8, 19, 20),
    8: (7, 9, 19, 20),
    9: (8, 10),
    10: (9, 11, 12),
    11: (10, 12),
    12: (10, 11),
    13: (1, 14, 16),
    14: (2, 13, 15),
    15: (2, 14, 16),
    16: (1, 13, 15),
    17: (3, 5, 18),
    18: (3, 5, 17),
    19: (6, 7, 20),
    20: (6, 7, 19),
    21: (19, 11, 12, 13),
    22: (20, 11, 12, 16),
}

LEFT_LANDMARKS = frozenset((13, 14, 17, 19, 21))
RIGHT_LANDMARKS = frozenset((15, 16, 18, 20, 22))

# Fine-view half widths (mm). Context views always cover the full candidate ROI.
FINE_VIEW_RADIUS_MM = {0: 20.0, 1: 14.0, 2: 14.0, 10: 14.0, 11: 14.0, 12: 14.0, 21: 25.0, 22: 25.0}
DEFAULT_FINE_VIEW_RADIUS_MM = 11.0


def landmark_type(landmark: int) -> str:
    return TYPE_NAMES[LANDMARK_TO_TYPE[int(landmark)]]


def landmark_type_index(landmark: int) -> int:
    return LANDMARK_TO_TYPE[int(landmark)]


def anatomical_group(landmark: int) -> str:
    return GROUP_NAMES[LANDMARK_TO_GROUP[int(landmark)]]


def anatomical_group_index(landmark: int) -> int:
    return LANDMARK_TO_GROUP[int(landmark)]


def fine_view_radius_mm(landmark: int) -> float:
    return float(FINE_VIEW_RADIUS_MM.get(int(landmark), DEFAULT_FINE_VIEW_RADIUS_MM))


def prior_context_indices(target: int) -> tuple[int, ...]:
    """Core20 context for a target; noisy Hard3 points never act as context."""
    return tuple(index for index in CORE20 if index != int(target))


def mirror_landmark_index(landmark: int) -> int:
    return int(mirror_permutation()[int(landmark)])


def validate_all23_schema() -> None:
    for mapping in (LANDMARK_TYPES, ANATOMICAL_GROUPS):
        flattened = tuple(index for values in mapping.values() for index in values)
        assert len(flattened) == len(set(flattened)) == NUM_LANDMARKS
        assert set(flattened) == set(ALL23_INDICES)
    assert set(ANCHORS) == set(ALL23_INDICES)
    assert all(landmark not in anchors for landmark, anchors in ANCHORS.items())
    assert all(len(anchors) <= 4 for anchors in ANCHORS.values())
    assert LEFT_LANDMARKS.isdisjoint(RIGHT_LANDMARKS)
    for left in LEFT_LANDMARKS:
        assert mirror_landmark_index(left) in RIGHT_LANDMARKS
    assert set(HARD3) <= set(LANDMARK_TYPES["contour"]) | set(
        LANDMARK_TYPES["texture_boundary"]
    )


validate_all23_schema()
