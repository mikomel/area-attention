import torch
from torch import nn

from area_attention import AreaAttention


class MultiHeadAreaAttention(nn.Module):
    """ Multi-Head version of Area Attention. """

    def __init__(self, area_attention: AreaAttention, num_heads: int, key_query_size: int,
                 key_query_size_hidden: int, value_size: int, value_size_hidden: int):
        """
        Initializes the Multi-Head Area Attention module.
        :param area_attention: initialized single head Area Attention module
        :param num_heads: number of heads
        :param key_query_size: input size of keys and queries
        :param key_query_size_hidden: hidden size of keys and queries
        :param value_size: input size of values
        :param value_size_hidden: hidden size of values
        """
        super(MultiHeadAreaAttention, self).__init__()
        self.area_attention = area_attention
        self.num_heads = num_heads
        self.key_query_size = key_query_size
        self.key_query_size_hidden = key_query_size_hidden
        self.value_size = value_size
        self.value_size_hidden = value_size_hidden

        self.query_projection = nn.Linear(key_query_size, num_heads * key_query_size_hidden)
        self.key_projection = nn.Linear(key_query_size, num_heads * key_query_size_hidden)
        self.value_projection = nn.Linear(value_size, num_heads * value_size_hidden)
        self.output_projection = nn.Linear(num_heads * value_size_hidden, value_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Area Attention module.
        :param q: queries Tensor with shape (batch_size, num_queries, key_query_size)
        :param k: keys Tensor with shape (batch_size, num_keys_values, key_query_size)
        :param v: values Tensor with shape (batch_size, num_keys_values, value_size)
        :returns a Tensor with shape (batch_size, num_queries, value_size)
        """
        batch_size, num_queries, _ = q.size()
        num_keys_values = k.size(1)
        q = self.query_projection(q).view(batch_size, num_queries, self.num_heads, self.key_query_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        k = self.key_projection(k).view(batch_size, num_keys_values, self.num_heads, self.key_query_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        v = self.value_projection(v).view(batch_size, num_keys_values, self.num_heads, self.value_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        attention = self.area_attention(q, k, v)
        attention = attention.view(batch_size, self.num_heads, num_queries, self.value_size_hidden)
        attention = attention.permute(0, 2, 1, 3).contiguous().flatten(-2, -1)
        output = self.output_projection(attention)
        return output
