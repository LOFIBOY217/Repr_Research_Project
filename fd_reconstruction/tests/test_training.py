import pytest
import torch
from recon_fd.engine.trainer import g_step
from recon_fd.engine.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from recon_fd.provenance import state_fingerprint


def test_both_encoder_decoder_update_frozen_representation(official_small_system):
    model, objective, optimizer, images = official_small_system
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    ref_before = state_fingerprint(objective.spaces["tiny"].extractor)
    metrics = g_step(model, objective, optimizer, images[:8], 1.0)
    assert metrics["group_grad_norm"]["encoder"] > 0
    assert metrics["group_grad_norm"]["decoder"] > 0
    for group in ("encoder", "decoder"):
        assert any(not torch.equal(before[k], p) for k, p in model.named_parameters() if k.startswith(group))
    assert ref_before == state_fingerprint(objective.spaces["tiny"].extractor)
    assert metrics["statistics_updates"] == {"tiny": 1}


def test_forward_pure_and_duplicate_commit_rejected(official_small_system):
    model, objective, _, images = official_small_system
    before = state_fingerprint(objective)
    first = objective(model(images[:8]))
    objective(model(images[:8]))
    assert before == state_fingerprint(objective)
    first.loss.backward()
    objective.commit(first)
    with pytest.raises(RuntimeError, match="Duplicate"):
        objective.commit(first)


def test_resume_next_step_exact(official_small_system, tmp_path):
    model, objective, optimizer, images = official_small_system
    g_step(model, objective, optimizer, images[:8], 1.0)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, objective, optimizer, 1, "signature", {"data": "test"}, {})
    expected = g_step(model, objective, optimizer, images[8:16], 1.0)
    expected_model, expected_loss = state_fingerprint(model), state_fingerprint(objective)
    restore_checkpoint(read_checkpoint(path), model, objective, optimizer, "signature", {"data": "test"})
    actual = g_step(model, objective, optimizer, images[8:16], 1.0)
    assert expected == actual
    assert expected_model == state_fingerprint(model)
    assert expected_loss == state_fingerprint(objective)
    with pytest.raises(ValueError, match="changed"):
        restore_checkpoint(read_checkpoint(path), model, objective, optimizer, "different", {"data": "test"})


def test_eval_does_not_disable_model_parameters(official_small_system):
    model, objective, optimizer, images = official_small_system
    model.eval()
    with torch.no_grad():
        model(images[:8])
    result = g_step(model, objective, optimizer, images[:8], 1)
    assert result["group_grad_norm"]["encoder"] > 0
