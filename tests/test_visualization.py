import numpy as np

from vipe.utils.visualization import _text_image


def test_text_image_uses_embedded_font() -> None:
    rendered = _text_image("N/A")

    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert rendered.size > 0
    assert rendered.min() < 255
    assert rendered.max() == 255
