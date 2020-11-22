import pytest
import torch

from area_attention import AreaAttention


@pytest.mark.parametrize('area_key_mode', ['mean', 'max', 'sample', 'concat', 'max_concat', 'sum', 'sample_concat', 'sample_sum'])
@pytest.mark.parametrize('area_value_mode', ['mean', 'max', 'sum'])
@pytest.mark.parametrize('max_area_height', [1, 2, 3])
@pytest.mark.parametrize('max_area_width', [1, 2, 3])
@pytest.mark.parametrize('memory_height', [1, 2, 4])
@pytest.mark.parametrize('top_k_areas', [0, 3])
def test_forward(area_key_mode, area_value_mode, max_area_height, max_area_width, memory_height, top_k_areas):
    num_queries = 8
    num_keys_values = 16
    key_query_size = 32
    value_size = 64

    memory_width = num_keys_values // memory_height
    if max_area_height > memory_height:
        pytest.xfail("max_area_height can't be bigger than memory_height")
    if max_area_width > memory_width:
        pytest.xfail("max_area_width can't be bigger than memory_width")

    area_attention = AreaAttention(
        key_query_size=key_query_size,
        area_key_mode=area_key_mode,
        area_value_mode=area_value_mode,
        max_area_height=max_area_height,
        max_area_width=max_area_width,
        memory_height=memory_height,
        memory_width=memory_width,
        dropout_rate=0.2,
        top_k_areas=top_k_areas
    )

    q = torch.rand(4, num_queries, key_query_size)
    k = torch.rand(4, num_keys_values, key_query_size)
    v = torch.rand(4, num_keys_values, value_size)
    output = area_attention(q, k, v)

    assert output.shape == (4, num_queries, value_size)
    assert output.isfinite().all()
