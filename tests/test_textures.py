import numpy as np

from experiments.textures import TEXTURES, random_texture


def test_every_texture_has_the_requested_shape_and_dtype():
    for name, make in TEXTURES.items():
        texture = make(64, 96, np.random.default_rng(0))
        assert texture.shape == (64, 96, 3), name
        assert texture.dtype == np.uint8, name
        assert texture.std() > 0, name


def test_random_texture_is_deterministic_under_seed():
    first, name_first = random_texture(48, 48, np.random.default_rng(7))
    second, name_second = random_texture(48, 48, np.random.default_rng(7))
    assert name_first == name_second
    assert np.array_equal(first, second)
    assert name_first in TEXTURES


def test_seeds_cover_several_generators():
    names = {random_texture(16, 16, np.random.default_rng(seed))[1] for seed in range(40)}
    assert len(names) >= 4
