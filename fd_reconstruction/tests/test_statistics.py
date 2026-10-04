import pytest
import torch
from recon_fd.objectives.statistics import RunningMoments, EMAStats, FeatureQueue
from conftest import upstream_module


def test_running_matches_full_and_partition():
    x = torch.randn(51, 7, dtype=torch.float64) + 100
    accumulator = RunningMoments()
    for chunk in x.split(8):
        accumulator.update(chunk)
    result = accumulator.moments()
    assert result.count == 51
    torch.testing.assert_close(result.mean, x.mean(0))
    torch.testing.assert_close(result.cov, torch.cov(x.T))


def test_ema_parity_gradients_and_no_forward_update(project):
    upstream = upstream_module(project / "third_party/FD-Loss/frechet_distance/queue.py", "official_queue")
    x = torch.randn(30, 5, dtype=torch.float64)
    accumulator = RunningMoments()
    accumulator.update(x)
    ours = EMAStats(5, 0.91)
    ours.initialize(accumulator.moments())
    theirs = upstream.FeatureQueue(size=30, feat_dim=5, ema_beta=0.91)
    theirs.mu_ema.copy_(x.mean(0))
    theirs.m2_ema.copy_(x.T @ x / len(x))
    new = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    state_before = {k: v.clone() for k, v in ours.state_dict().items()}
    ours_stats = ours.preview(new)
    theirs_mu, theirs_cov = theirs.build_feats_stats(new)
    ours.preview(new)
    for key, value in state_before.items():
        torch.testing.assert_close(value, ours.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(ours_stats.mean, theirs_mu)
    torch.testing.assert_close(ours_stats.cov, theirs_cov)
    grad_a = torch.autograd.grad(ours_stats.cov.square().sum(), new, retain_graph=True)[0]
    grad_b = torch.autograd.grad(theirs_cov.square().sum(), new)[0]
    torch.testing.assert_close(grad_a, grad_b)
    ours.commit(new, 0)
    assert int(ours.updates) == 1 and not ours.mean.requires_grad
    with pytest.raises(RuntimeError, match="duplicate"):
        ours.commit(new, 0)


def test_queue_wraparound_matches_official(project):
    upstream = upstream_module(project / "third_party/FD-Loss/frechet_distance/queue.py", "official_queue_wrap")
    original = torch.randn(11, 4)
    ours = FeatureQueue(4, 11)
    ours.initialize(original)
    theirs = upstream.FeatureQueue(size=11, feat_dim=4)
    theirs.feats.copy_(original)
    for step in range(6):
        new = torch.randn(4, 4, requires_grad=True)
        moments = ours.preview(new)
        snapshot = theirs.build_feats_snapshot(new).double()
        torch.testing.assert_close(moments.mean, snapshot.mean(0))
        torch.testing.assert_close(moments.cov, torch.cov(snapshot.T))
        ours.commit(new, step)
        theirs.enqueue(new)
        assert int(ours.pointer) == theirs.pointer


def test_bad_statistics_rejected():
    with pytest.raises(ValueError):
        EMAStats(3, 1.0)
    accumulator = RunningMoments()
    with pytest.raises(ValueError):
        accumulator.moments()
    with pytest.raises(FloatingPointError):
        accumulator.update(torch.full((3, 2), float("nan")))
