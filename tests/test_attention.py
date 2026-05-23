"""
tests/test_attention.py
========================
Tests for HDCAttentionHead, MultiHeadHDCAttention,
HDCTransformerLayer, HDCSequenceAttention, CrossModalHDCAttention.
"""
import pytest
import torch
from hdc.attention import (
    HDCAttentionHead,
    MultiHeadHDCAttention,
    HDCTransformerLayer,
    HDCSequenceAttention,
    CrossModalHDCAttention,
    _gen_hv,
)

D = 128
N = 10


def _hv(seed):
    return _gen_hv(D, seed=seed)


# ── HDCAttentionHead ──────────────────────────────────────────────────────────

class TestHDCAttentionHead:
    def setup_method(self):
        self.head  = HDCAttentionHead(D, temperature=5.0)
        self.query = _hv(0)
        self.keys  = torch.stack([_hv(i) for i in range(N)])

    def test_attend_output_shape(self):
        out, w = self.head.attend(self.query, self.keys)
        assert out.shape == (D,)
        assert w.shape   == (N,)

    def test_weights_sum_to_one(self):
        _, w = self.head.attend(self.query, self.keys)
        assert abs(float(w.sum()) - 1.0) < 1e-4

    def test_weights_nonnegative(self):
        _, w = self.head.attend(self.query, self.keys)
        assert float(w.min()) >= 0.0

    def test_binary_output(self):
        out, _ = self.head.attend(self.query, self.keys)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_attend_with_values(self):
        values = torch.stack([_hv(100 + i) for i in range(N)])
        out, w = self.head.attend(self.query, self.keys, values)
        assert out.shape == (D,)

    def test_attend_empty_keys(self):
        empty = torch.zeros(0, D)
        out, w = self.head.attend(self.query, empty)
        assert out.shape == (D,)

    def test_batch_attend_shape(self):
        queries = torch.stack([_hv(i) for i in range(4)])
        out, w  = self.head.attend_batch(queries, self.keys)
        assert out.shape == (4, D)
        assert w.shape   == (4, N)

    def test_batch_weights_sum_to_one(self):
        queries = torch.stack([_hv(i) for i in range(4)])
        _, w    = self.head.attend_batch(queries, self.keys)
        sums = w.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)

    def test_high_temperature_more_peaked(self):
        head_hot  = HDCAttentionHead(D, temperature=20.0)
        head_cold = HDCAttentionHead(D, temperature=0.5)
        _, w_hot  = head_hot.attend(self.query, self.keys)
        _, w_cold = head_cold.attend(self.query, self.keys)
        # High temperature → more peaked distribution
        assert float(w_hot.max()) > float(w_cold.max())


# ── MultiHeadHDCAttention ─────────────────────────────────────────────────────

class TestMultiHeadHDCAttention:
    def setup_method(self):
        self.mh    = MultiHeadHDCAttention(D, n_heads=4)
        self.query = _hv(0)
        self.keys  = torch.stack([_hv(i) for i in range(N)])

    def test_attend_output_shape(self):
        out, hw = self.mh.attend(self.query, self.keys)
        assert out.shape == (D,)

    def test_n_head_weights(self):
        _, hw = self.mh.attend(self.query, self.keys)
        assert len(hw) == 4

    def test_attend_batch(self):
        queries = torch.stack([_hv(i) for i in range(3)])
        out, hw = self.mh.attend_batch(queries, self.keys)
        assert out.shape == (3, D)

    def test_different_heads_different_masks(self):
        m1 = self.mh._head_masks[0]
        m2 = self.mh._head_masks[1]
        assert not torch.equal(m1, m2)

    def test_binary_output(self):
        out, _ = self.mh.attend(self.query, self.keys)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})


# ── HDCTransformerLayer ───────────────────────────────────────────────────────

class TestHDCTransformerLayer:
    def setup_method(self):
        self.layer = HDCTransformerLayer(D, n_heads=2)
        self.seq   = torch.stack([_hv(i) for i in range(8)])

    def test_forward_shape(self):
        out, w = self.layer.forward(self.seq)
        assert out.shape == (D,)
        assert w.shape   == (8,)

    def test_forward_with_query(self):
        q      = _hv(99)
        out, w = self.layer.forward(self.seq, query=q)
        assert out.shape == (D,)

    def test_binary_output(self):
        out, _ = self.layer.forward(self.seq)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_forward_sequence_shape(self):
        out_seq = self.layer.forward_sequence(self.seq)
        assert out_seq.shape == (8, D)

    def test_forward_sequence_binary(self):
        out_seq = self.layer.forward_sequence(self.seq[:3])
        assert set(out_seq.unique().tolist()).issubset({0.0, 1.0})

    def test_different_positions_different_output(self):
        seq = torch.stack([_hv(i) for i in range(5)])
        out_seq = self.layer.forward_sequence(seq)
        # Different positions should generally produce different outputs
        assert not torch.equal(out_seq[0], out_seq[-1])


# ── HDCSequenceAttention ──────────────────────────────────────────────────────

class TestHDCSequenceAttention:
    def setup_method(self):
        self.sa = HDCSequenceAttention(D, buffer_size=15, n_heads=2)

    def test_first_step_returns_input(self):
        hv = _hv(0)
        ctx, w = self.sa.step(hv)
        assert ctx.shape == (D,)

    def test_buffer_fills_up(self):
        for i in range(10):
            self.sa.step(_hv(i))
        assert self.sa.buffer_len == 10

    def test_buffer_capped_at_max(self):
        for i in range(25):
            self.sa.step(_hv(i))
        assert self.sa.buffer_len == 15

    def test_context_shape_after_filling(self):
        for i in range(10):
            ctx, w = self.sa.step(_hv(i))
        assert ctx.shape == (D,)

    def test_reset_clears_buffer(self):
        for i in range(5):
            self.sa.step(_hv(i))
        self.sa.reset()
        assert self.sa.buffer_len == 0

    def test_weights_shape_grows_with_buffer(self):
        for i in range(5):
            ctx, w = self.sa.step(_hv(i))
        assert len(w) > 0  # weights exist after buffer has content


# ── CrossModalHDCAttention ────────────────────────────────────────────────────

class TestCrossModalHDCAttention:
    def setup_method(self):
        self.cm = CrossModalHDCAttention(D, n_heads=2, max_keys=20)
        for i in range(8):
            self.cm.add_context("sensor", _hv(i),       _hv(100 + i))
            self.cm.add_context("action", _hv(200 + i), _hv(300 + i))

    def test_modality_sizes(self):
        assert self.cm.modality_sizes["sensor"] == 8
        assert self.cm.modality_sizes["action"] == 8

    def test_query_shape(self):
        out, w = self.cm.query(_hv(5), "sensor")
        assert out.shape == (D,)

    def test_query_unknown_modality(self):
        out, w = self.cm.query(_hv(0), "nonexistent")
        assert out.shape == (D,)

    def test_fuse_shape(self):
        fused = self.cm.fuse(_hv(3))
        assert fused.shape == (D,)

    def test_fuse_specific_modalities(self):
        fused = self.cm.fuse(_hv(3), modalities=["sensor"])
        assert fused.shape == (D,)

    def test_query_all_returns_all_modalities(self):
        results = self.cm.query_all(_hv(0))
        assert "sensor" in results
        assert "action" in results
        for name, (out, w) in results.items():
            assert out.shape == (D,)

    def test_max_keys_eviction(self):
        cm = CrossModalHDCAttention(D, max_keys=5)
        for i in range(10):
            cm.add_context("x", _hv(i))
        assert cm.modality_sizes["x"] == 5

    def test_register_modality(self):
        self.cm.register_modality("lidar")
        assert "lidar" in self.cm._modalities
