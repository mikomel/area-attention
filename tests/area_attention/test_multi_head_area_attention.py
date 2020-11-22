import pytest
import torch

from area_attention import AreaAttention, MultiHeadAreaAttention

area_attention = AreaAttention(
    key_query_size=32,
    area_key_mode='max',
    area_value_mode='mean',
    max_area_height=2,
    max_area_width=2,
    memory_height=4,
    memory_width=4,
    dropout_rate=0.2,
    top_k_areas=0
)


@pytest.mark.parametrize('num_heads', [1, 2, 4])
def test_forward(num_heads):
    multi_head_area_attention = MultiHeadAreaAttention(
        area_attention=area_attention,
        num_heads=num_heads,
        key_query_size=32,
        key_query_size_hidden=32,
        value_size=64,
        value_size_hidden=64
    )

    q = torch.rand(4, 8, 32)
    k = torch.rand(4, 16, 32)
    v = torch.rand(4, 16, 64)
    output = multi_head_area_attention(q, k, v)

    assert output.shape == (4, 8, 64)
    assert output.isfinite().all()
