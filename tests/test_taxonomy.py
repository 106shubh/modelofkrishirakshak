"""Tests for the taxonomy / knowledge base."""
from cropguard.taxonomy import CROPS, REGIONS, TAXONOMY


def test_38_plantvillage_classes():
    assert len(TAXONOMY.classes) == 38


def test_pest_disease_ids_deterministic_and_unique():
    ids = [TAXONOMY.pest_disease_id(c) for c in TAXONOMY.classes]
    assert len(ids) == len(set(ids))
    assert ids == [TAXONOMY.pest_disease_id(c) for c in TAXONOMY.classes]


def test_pest_disease_id_roundtrip():
    for class_name in TAXONOMY.classes:
        pid = TAXONOMY.pest_disease_id(class_name)
        assert TAXONOMY.class_for_id(pid) == class_name


def test_crop_ids_valid():
    for class_name in TAXONOMY.classes:
        crop_id = TAXONOMY.crop_id(class_name)
        assert crop_id in CROPS


def test_is_healthy_flag():
    assert TAXONOMY.is_healthy("Tomato___healthy")
    assert not TAXONOMY.is_healthy("Tomato___Late_blight")


def test_regions_are_maharashtra_districts():
    assert len(REGIONS) >= 10
    assert "nashik" in REGIONS
