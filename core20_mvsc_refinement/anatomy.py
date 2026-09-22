"""Verified anatomical grouping used only by the Core20 Stage 4 model."""

from __future__ import annotations

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3
from all23_rgb_geodesic_cascade.anatomy import mirror_permutation


CORE20_INDICES = tuple(CORE20)
CORE20_GROUPS = {
    "upper_midline": (1, 2),
    "nasal_oral_midline": (3, 4, 5, 6, 7, 8, 9),
    "chin_contour": (10, 11, 12),
    "ocular": (13, 14, 15, 16),
    "alar": (17, 18),
    "commissure": (19, 20),
}
GROUP_NAMES = tuple(CORE20_GROUPS)
LANDMARK_TO_GROUP = {
    landmark: group_index
    for group_index, landmarks in enumerate(CORE20_GROUPS.values())
    for landmark in landmarks
}

# Local structure used by the training-only anatomical consistency term. The
# target itself is never present among its anchors.
ANCHORS = {
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
}

LEFT_LANDMARKS = frozenset((13, 14, 17, 19))
RIGHT_LANDMARKS = frozenset((15, 16, 18, 20))


def core20_group(landmark: int) -> str:
    return GROUP_NAMES[LANDMARK_TO_GROUP[int(landmark)]]


def core20_group_index(landmark: int) -> int:
    return LANDMARK_TO_GROUP[int(landmark)]


def mirror_landmark_index(landmark: int) -> int:
    """Return the verified bilateral partner (midline points map to themselves)."""
    return int(mirror_permutation()[int(landmark)])


def validate_core20_schema() -> None:
    flattened = tuple(
        landmark for landmarks in CORE20_GROUPS.values() for landmark in landmarks
    )
    assert len(flattened) == len(set(flattened)) == 20
    assert set(flattened) == set(CORE20_INDICES)
    assert set(flattened).isdisjoint(HARD3)
    assert all(landmark not in anchors for landmark, anchors in ANCHORS.items())


validate_core20_schema()
