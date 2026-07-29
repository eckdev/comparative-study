"""Canonical anatomical contract for the 23-point orthodontic dataset."""

NUM_LANDMARKS = 23

LANDMARK_NAMES = (
    "Trichion",
    "Glabella",
    "Nasion",
    "Pronasale",
    "Columella",
    "Subnasale",
    "Labiale superius",
    "Stomion",
    "Labiale inferius",
    "Sublabiale",
    "Pogonion",
    "Gnathion",
    "Menton",
    "Exocanthion left",
    "Endocanthion left",
    "Endocanthion right",
    "Exocanthion right",
    "Alare left",
    "Alare right",
    "Cheilion left",
    "Cheilion right",
    "Gonion left",
    "Gonion right",
)

MIDLINE = tuple(range(13))
SYMMETRY_PAIRS = ((13, 16), (14, 15), (17, 18), (19, 20), (21, 22))
HARD3 = (0, 21, 22)
CORE20 = tuple(index for index in range(NUM_LANDMARKS) if index not in HARD3)

TEXTURE_LANDMARKS = (0, 13, 14, 15, 16, 17, 18, 19, 20)
CONTOUR_LANDMARKS = (10, 11, 12, 21, 22)
GENERIC_LANDMARKS = tuple(
    index for index in range(NUM_LANDMARKS)
    if index not in set(TEXTURE_LANDMARKS) | set(CONTOUR_LANDMARKS)
)

# Sequential midline relations plus clinically meaningful bilateral/local relations.
ANATOMICAL_EDGES = tuple(
    dict.fromkeys(
        [(index, index + 1) for index in range(12)]
        + list(SYMMETRY_PAIRS)
        + [
            (1, 13), (1, 16), (2, 14), (2, 15),
            (3, 17), (3, 18), (5, 17), (5, 18),
            (7, 19), (7, 20), (10, 21), (10, 22),
            (11, 21), (11, 22), (12, 21), (12, 22),
        ]
    )
)


def landmark_group(index):
    if index in HARD3:
        return "hard3"
    if index in MIDLINE:
        return "midline"
    return "bilateral"


def head_group(index):
    if index in TEXTURE_LANDMARKS:
        return "texture"
    if index in CONTOUR_LANDMARKS:
        return "contour"
    return "generic"


def roi_radius_mm(index):
    if index == 0:
        return 35.0
    if index in (21, 22):
        return 45.0
    if index in set(range(3, 9)) | set(range(13, 21)):
        return 20.0
    return 25.0


def heatmap_sigma_mm(index):
    if index == 0:
        return 6.0
    if index in (21, 22):
        return 5.0
    if index in (10, 11, 12):
        return 3.5
    return 2.5


def mirror_permutation():
    permutation = list(range(NUM_LANDMARKS))
    for left, right in SYMMETRY_PAIRS:
        permutation[left], permutation[right] = right, left
    return tuple(permutation)


def graph_attention_mask():
    """Return True where landmark-token attention must be blocked."""
    import torch

    allowed = torch.eye(NUM_LANDMARKS, dtype=torch.bool)
    for left, right in ANATOMICAL_EDGES:
        allowed[left, right] = True
        allowed[right, left] = True
    # All midline tokens share global profile information.
    for left in MIDLINE:
        for right in MIDLINE:
            allowed[left, right] = True
    return ~allowed


def validate_schema():
    assert len(LANDMARK_NAMES) == NUM_LANDMARKS
    assert set(MIDLINE).isdisjoint({item for pair in SYMMETRY_PAIRS for item in pair})
    assert set(CORE20) | set(HARD3) == set(range(NUM_LANDMARKS))
    assert set(TEXTURE_LANDMARKS) | set(CONTOUR_LANDMARKS) | set(GENERIC_LANDMARKS) == set(range(NUM_LANDMARKS))
    assert mirror_permutation()[mirror_permutation()[21]] == 21


validate_schema()
