"""Conservative perceptual grouping supplements exact decoded-pixel hashes."""

import pytest

from covid_xray.data import build_manifest, perceptual_hash, validate_manifest


def test_perceptual_hash_is_reproducible_64_bit(image_index):
    path = image_index.iloc[0].path
    value = perceptual_hash(path)
    assert len(value) == 16
    assert int(value, 16) >= 0
    assert perceptual_hash(path) == value


def test_perceptual_matches_group_different_pixel_hashes(image_index):
    frame = image_index.copy()
    frame['perceptual_hash'] = frame.path.map(perceptual_hash)
    # Simulate a manually confirmed signature match with distinct real pixels.
    # The splitter conservatively joins them even with different patient IDs.
    frame.loc[1, 'perceptual_hash'] = frame.loc[0, 'perceptual_hash']
    result = build_manifest(frame)
    patients = set(frame.iloc[:2].patient_id)
    assert result[result.patient_id.isin(patients)].split.nunique() == 1


def test_cross_split_perceptual_matches_are_rejected(manifest):
    first = manifest[manifest.split.eq('train')].index[0]
    second = manifest[manifest.split.eq('test')].index[0]
    manifest.loc[second, 'perceptual_hash'] = manifest.loc[first, 'perceptual_hash']
    with pytest.raises(ValueError, match='perceptual_hash'):
        validate_manifest(manifest)


def test_official_test_boundary_cannot_be_changed_after_splitting(image_index):
    image_index['original_split'] = 'train'
    for _, group in image_index.groupby('label'):
        image_index.loc[group.index[:3], 'original_split'] = 'test'
    frame = build_manifest(image_index)
    index = frame[frame.split.eq('test')].index[0]
    frame.loc[index, 'split'] = 'train'
    with pytest.raises(ValueError, match='official test boundary'):
        validate_manifest(frame)


def test_whitespace_identity_variants_are_not_silently_split(image_index):
    image_index.loc[0, 'patient_id'] += ' '
    with pytest.raises(ValueError, match='patient_id'):
        build_manifest(image_index)


def test_mixed_string_and_numeric_patient_ids_require_reconciliation(image_index):
    image_index.loc[0, 'patient_id'] = 123
    image_index.loc[1, 'patient_id'] = '123'
    with pytest.raises(ValueError, match='patient_id'):
        build_manifest(image_index)
