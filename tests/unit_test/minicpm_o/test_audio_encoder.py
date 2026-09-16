# SPDX-License-Identifier: Apache-2.0
"""Tests for the native MiniCPM-o audio encoder.

The golden-parity test compares the native encoder against the checkpoint's
remote-code ``MiniCPMWhisperEncoder`` on shared random weights, so it needs a
checkpoint directory with the remote modeling files (weights not required).
Set ``MINICPMO_CHECKPOINT`` or place ``MiniCPM-o-4_6``/``MiniCPM-o-4_5`` in
the repo root; the test skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
    _audio_config_object,
    _chunked_causal_mask,
    _feature_lens_after_pooling,
    _fuse_qkv,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "modeling_minicpmo.py").exists():
            return path
    return None


def _reference_chunk_mask(size: int, chunk_size: int) -> torch.Tensor:
    ret = torch.zeros(size, size, dtype=torch.bool)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, :ending] = True
    return ret


@pytest.mark.parametrize(
    "size,chunk", [(1, 50), (49, 50), (50, 50), (301, 50), (300, 7)]
)
def test_chunked_causal_mask_matches_reference_loop(size: int, chunk: int) -> None:
    got = _chunked_causal_mask(size, chunk, torch.device("cpu"))
    assert torch.equal(got, _reference_chunk_mask(size, chunk))


def _small_whisper_config():
    from transformers import WhisperConfig

    return WhisperConfig(
        num_mel_bins=80,
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=256,
        max_source_positions=1500,
        activation_function="gelu",
    )


def test_audio_config_object_preserves_config_instances() -> None:
    audio_config = PretrainedConfig(d_model=64, num_mel_bins=80)
    config = PretrainedConfig(audio_config=audio_config)
    assert _audio_config_object(config) is audio_config


def test_audio_config_object_converts_shim_dict() -> None:
    config = PretrainedConfig(audio_config={"d_model": 64, "num_mel_bins": 80})
    audio_config = _audio_config_object(config)
    assert isinstance(audio_config, PretrainedConfig)
    assert audio_config.d_model == 64
    assert audio_config.num_mel_bins == 80


def _native_state_from_hf(encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return _fuse_qkv(dict(encoder.state_dict()))


def _build_remote_encoder(checkpoint: Path, config):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    remote_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.MiniCPMWhisperEncoder", str(checkpoint)
    )
    config._attn_implementation = "sdpa"
    remote = remote_cls(config).eval()
    # transformers v5 attention returns (out, weights); the remote layer
    # unpacks a v4-era 3-tuple. Pad the return for the golden run.
    for layer in remote.layers:
        attn = layer.self_attn
        orig_forward = attn.forward

        def _forward(*args, _orig=orig_forward, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                return (*out, None)
            return out

        attn.forward = _forward
    return remote


@pytest.mark.parametrize("lens", [[3000, 2000, 137], [700]])
def test_golden_parity_vs_remote_code(lens: list[int]) -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with remote modeling files")

    torch.manual_seed(0)
    config = _small_whisper_config()
    remote = _build_remote_encoder(checkpoint, config)

    native = MiniCPMWhisperEncoder(config).eval()
    native.load_state_dict(_native_state_from_hf(remote), strict=True)

    batch = len(lens)
    max_mel = max(lens)
    mel = torch.randn(batch, config.num_mel_bins, max_mel)
    for i, length in enumerate(lens):
        mel[i, :, length:] = 0.0
    feat_lens = torch.tensor(lens)

    max_seq_len = (max_mel - 1) // 2 + 1
    # chunk spanning frame boundaries: audio_chunk_length=1s → 50 frames
    chunk = 50
    seq_range = torch.arange(max_seq_len)
    valid = seq_range[None, :] < ((feat_lens - 1) // 2 + 1)[:, None]
    allowed = (
        _chunked_causal_mask(max_seq_len, chunk, torch.device("cpu"))[None]
        & valid[:, None, :]
    )

    # Remote path: additive -inf mask, output_hidden_states, last hidden state.
    remote_mask = torch.zeros(batch, 1, max_seq_len, max_seq_len)
    remote_mask[~allowed.unsqueeze(1).expand(batch, 1, max_seq_len, max_seq_len)] = (
        float("-inf")
    )
    with torch.no_grad():
        golden = remote(
            mel, attention_mask=remote_mask, output_hidden_states=True
        ).hidden_states[-1]

    native_mask = torch.where(allowed, 0.0, -1e9).unsqueeze(1)
    with torch.no_grad():
        got = native(mel, native_mask)

    for i, length in enumerate(lens):
        valid_frames = (length - 1) // 2 + 1
        torch.testing.assert_close(
            got[i, :valid_frames], golden[i, :valid_frames], rtol=1e-4, atol=1e-4
        )


def _tiny_audio_encoder(pool_step: int = 2) -> MiniCPMOAudioEncoder:
    """A ``MiniCPMOAudioEncoder`` with random weights and no checkpoint I/O."""
    torch.manual_seed(0)
    config = _small_whisper_config()
    encoder = object.__new__(MiniCPMOAudioEncoder)
    torch.nn.Module.__init__(encoder)
    encoder._device = torch.device("cpu")
    encoder._dtype = torch.float32
    encoder.apm = MiniCPMWhisperEncoder(config)
    encoder.audio_projection_layer = MultiModalProjector(
        in_dim=config.d_model, out_dim=16
    )
    encoder.audio_pool_step = pool_step
    encoder.audio_avg_pooler = torch.nn.AvgPool1d(pool_step, stride=pool_step)
    encoder.chunk_num_frame = 50
    encoder._chunk_mask_cache = None
    return encoder


def test_padding_content_does_not_change_valid_output() -> None:
    """A short row's embeddings must not depend on the batch's padding.

    Encode the same short mel twice in batches of equal shape that differ only
    in the padding content of the short row. ``seq_range`` indexes the
    post-conv sequence, so a validity bound taken from the raw mel lengths
    admits padding frames as attention keys and the two runs disagree.
    """
    encoder = _tiny_audio_encoder()
    short_len, long_len = 137, 3000

    torch.manual_seed(1)
    short_mel = torch.randn(80, short_len)
    long_mel = torch.randn(80, long_len)

    padded_mel = torch.zeros(2, 80, long_len)
    padded_mel[0, :, :long_len] = long_mel
    padded_mel[1, :, :short_len] = short_mel
    lens = torch.tensor([long_len, short_len])

    with torch.no_grad():
        clean = encoder(audio_features=padded_mel, audio_feature_lens=lens)

        torch.manual_seed(2)
        polluted_mel = padded_mel.clone()
        polluted_mel[1, :, short_len:] = torch.randn(80, long_len - short_len)
        polluted = encoder(audio_features=polluted_mel, audio_feature_lens=lens)

    pooled_short = int(
        _feature_lens_after_pooling(torch.tensor([short_len]), encoder.audio_pool_step)
    )
    # Rows are emitted in chunk order with each chunk trimmed to its pooled
    # length, so the short row sits at the end of the flattened output.
    torch.testing.assert_close(
        polluted["audio_embeds"][-pooled_short:],
        clean["audio_embeds"][-pooled_short:],
        rtol=0,
        atol=0,
    )


def test_projector_shapes() -> None:
    projector = MultiModalProjector(in_dim=64, out_dim=96)
    out = projector(torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 96)
