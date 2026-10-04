"""A must execute the pinned original core, not a numerically modified rewrite."""
from copy import deepcopy
import pytest
import torch
from conftest import upstream_module
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset
from recon_fd.engine.trainer import build_objective, initialize_statistics, g_step
from recon_fd.objectives.official_fd import OfficialFDStatistics, OfficialStaticFD
from recon_fd.provenance import state_fingerprint
from recon_fd.tokenizers import TinyReconstructor


@pytest.mark.parametrize("filename", ["queue.py", "losses.py"])
def test_vendor_is_byte_identical(project, filename):
    original = project / "third_party/FD-Loss/frechet_distance" / filename
    packaged = project / "src/recon_fd/vendor/fd_loss" / filename
    assert original.read_bytes() == packaged.read_bytes()


def original_modules(project):
    root = project / "third_party/FD-Loss/frechet_distance"
    return (upstream_module(root / "queue.py", "oracle_queue"),
            upstream_module(root / "losses.py", "oracle_losses"))


def original_fd(queue, losses, features, mean, cov, root):
    if queue.ema_stats or queue.online_accum:
        mu, sigma = queue.build_feats_stats(features)
        return losses.compute_frechet_distance_loss(mean, cov, mu=mu, sigma=sigma, sigma_ref_sqrt=root)
    return losses.compute_frechet_distance_loss(mean, cov,
        all_feats=queue.build_feats_snapshot(features), sigma_ref_sqrt=root)


@pytest.mark.parametrize("mode", ["ema", "queue", "queue_online"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initialization_loss_gradient_and_multistep_commit_exact(project, mode, dtype):
    queue_module, losses = original_modules(project)
    n, d, beta = 19, 4, 0.999
    ours = OfficialFDStatistics(d, n, mode, beta)
    original = queue_module.FeatureQueue(n, d, online_accum=mode == "queue_online",
                                         ema_beta=beta if mode == "ema" else 0)
    initial = torch.randn(n, d, dtype=dtype)
    filled = 0
    for batch in initial.split(6):
        ours.accumulate_initial(batch)
        if original.ema_stats:
            original.accumulate_batch(batch)
        else:
            original.feats[filled:filled + len(batch)] = batch.float()
        filled += len(batch)
    ours.finalize_initialization()
    if original.ema_stats:
        original._finalize_streaming_init()
    else:
        original.ptr.zero_()
        if original.online_accum:
            original._init_accumulators()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, ours.queue.state_dict()[key], rtol=0, atol=0)
    real = torch.randn(35, d, dtype=torch.float64)
    mean, cov = real.mean(0), torch.cov(real.T)
    root = losses.precompute_sigma_ref_sqrt(cov)
    for step in range(7):  # Exercises queue wraparound repeatedly.
        features = torch.randn(6, d, dtype=dtype, requires_grad=True)
        before = state_fingerprint(ours)
        a = ours.fd(features, mean, cov, root)
        b = original_fd(original, losses, features, mean, cov, root)
        assert a.dtype == b.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        grad_a = torch.autograd.grad(a, features, retain_graph=True)[0]
        grad_b = torch.autograd.grad(b, features)[0]
        torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)
        assert state_fingerprint(ours) == before
        ours.commit(features, step)
        original.enqueue(features.detach())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, ours.queue.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ema", "queue", "queue_online"])
def test_complete_reconstruction_updates_match_official_core(project, tmp_path, mode):
    config = load_config(project / "configs/smoke.yaml", [f"static.statistics={mode}"])
    config["static"]["reference_cache"] = str(tmp_path / "cache")
    config["static"]["representations"][0]["weight"] = 0.7
    validate(config)
    model = TinyReconstructor()
    data = SyntheticDataset(64, 16, 217)
    objective, _ = build_objective(config, data, "cpu")
    assert isinstance(objective, OfficialStaticFD)
    initialize_statistics(model, objective, data, config, "cpu")
    space = objective.spaces["tiny"]
    oracle_model = deepcopy(model)
    oracle_extractor = deepcopy(space.extractor)
    queue_module, losses = original_modules(project)
    queue = queue_module.FeatureQueue(32, 6, online_accum=mode == "queue_online",
                                      ema_beta=0.9 if mode == "ema" else 0)
    queue.load_state_dict(space.statistics.queue.state_dict())
    root = losses.precompute_sigma_ref_sqrt(space.reference_cov)
    torch.testing.assert_close(root, space.reference_sqrt, rtol=0, atol=0)
    ours_opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    oracle_opt = torch.optim.AdamW(oracle_model.parameters(), lr=0.001)
    fixed = state_fingerprint(space.extractor)
    for step in range(6):
        images = torch.stack([data[i]["image"] for i in range(step * 8, (step + 1) * 8)])
        oracle_model.train()
        oracle_opt.zero_grad(set_to_none=True)
        features = oracle_extractor(oracle_model(images))
        fd = original_fd(queue, losses, features, space.reference_mean, space.reference_cov, root)
        normalized = fd / (fd.detach() + 0.01)
        loss = torch.tensor(0.0) + space.weight * normalized
        loss.backward()
        queue.enqueue(features.detach())  # Original loop enqueues before optimizer.step.
        torch.nn.utils.clip_grad_norm_(oracle_model.parameters(), 1, error_if_nonfinite=True)
        oracle_opt.step()
        actual = g_step(model, objective, ours_opt, images, 1)
        assert actual["raw_fd"]["tiny"] == float(fd.detach())
        assert actual["loss"] == float(loss.detach())
        for key, value in oracle_model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        for key, value in queue.state_dict().items():
            torch.testing.assert_close(value, space.statistics.queue.state_dict()[key], rtol=0, atol=0)
        assert state_fingerprint(space.extractor) == fixed


def test_preserves_official_singular_derivative(project):
    queue_module, losses = original_modules(project)
    ours = OfficialFDStatistics(3, 8, "ema", 0.9)
    ours.accumulate_initial(torch.zeros(8, 3))
    ours.finalize_initialization()
    queue = queue_module.FeatureQueue(8, 3, ema_beta=0.9)
    queue.load_state_dict(ours.queue.state_dict())
    features = torch.zeros(4, 3, requires_grad=True)
    mean, cov = torch.zeros(3).double(), torch.eye(3).double()
    a = ours.fd(features, mean, cov, cov)
    b = original_fd(queue, losses, features, mean, cov, cov)
    ga = torch.autograd.grad(a, features, retain_graph=True)[0]
    gb = torch.autograd.grad(b, features)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0, equal_nan=True)
    # Preserve upstream behavior, including non-finite values if the installed
    # PyTorch version produces them; don't assume every version yields NaNs.


def test_a_rejects_nonofficial_initialization_and_normalization(project):
    for setting, message in (("static.initialization_samples=16", "initialization_samples"),
                             ("static.norm_eps=0.1", "normalization"),
                             ("static.ema_beta=0", "positive beta")):
        config = load_config(project / "configs/smoke.yaml", [setting])
        with pytest.raises(ValueError, match=message):
            validate(config)


def test_official_initialization_lifecycle():
    state = OfficialFDStatistics(3, 8, "ema", 0.99)
    with pytest.raises(ValueError, match="Incomplete"):
        state.finalize_initialization()
    state.accumulate_initial(torch.randn(8, 3))
    state.finalize_initialization()
    with pytest.raises(ValueError, match="once"):
        state.accumulate_initial(torch.randn(1, 3))
    state.commit(torch.randn(2, 3), 0)
    with pytest.raises(RuntimeError, match="duplicate"):
        state.commit(torch.randn(2, 3), 0)
