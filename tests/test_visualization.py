import numpy as np
import pytest

from vipe.utils.visualization import _text_image


@pytest.mark.parametrize("text", ["N/A", "animal", "rgb + instance"])
def test_text_image_uses_embedded_font(text: str) -> None:
    rendered = _text_image(text)

    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert rendered.size > 0
    assert rendered.min() < 255
    assert rendered.max() == 255
