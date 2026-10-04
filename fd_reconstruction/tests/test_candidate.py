from copy import deepcopy
import json
import pytest
import torch
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset, sequential_loader
from recon_fd.tokenizers import TinyReconstructor
from recon_fd.engine.trainer import build_objective, initialize_statistics
from recon_fd.engine.adversarial import critic_step
from recon_fd.engine.candidate import candidate_g_step
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.cli import train_main, evaluate_main
from recon_fd.objectives.candidate_fd import CandidateFD
from recon_fd.vendor.advfd.adversarial import FeatureStatsEMA, real_whitened_frechet_distance_from_stats


def system(project, method="ours", spec=None):
    config = load_config(project / "configs/ours_smoke.yaml")
    config["method"] = method
    if method == "ours_real_ema":
        config["adaptive"]["real_stats"]["mode"] = "ema"
    if method == "ours_fixed_reference":
        config["adaptive"]["real_stats"]["mode"] = "frozen_initial"
    if spec is not None:
        config["adaptive"]["representation"] = spec
        config["adaptive"]["gradient_checkpointing"] = spec["kind"] == "timm"
    if method == "ours_lora":
        config["adaptive"]["trainable_scope"] = "lora"
    validate(config)
    dataset = SyntheticDataset(64, 16, 217)
    model = TinyReconstructor()
    objective, _ = build_objective(config, dataset, "cpu")
    initialize_statistics(model, objective, dataset, config, "cpu")
    images = torch.stack([dataset[i]["image"] for i in range(8)])
    g = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                         betas=config["train"]["betas"], weight_decay=0)
    d = torch.optim.AdamW(objective.critic_parameters(), lr=config["adaptive"]["lr"],
                         betas=config["adaptive"]["betas"], weight_decay=0)
    return config, model, objective, images, g, d


def pool_features(objective):
    with torch.no_grad():
        return torch.cat([objective.extractor(batch["image"]) for batch in
                          sequential_loader(objective.real_reference.pool, 8)]).double()


def test_C_no_static_construction_or_initialization_dependency(project, monkeypatch):
    import recon_fd.engine.trainer as trainer
    def forbidden(*args, **kwargs):
        raise AssertionError("Static dependency was invoked")
    monkeypatch.setattr(trainer, "get_reference", forbidden)
    monkeypatch.setattr(trainer, "build_representation", forbidden)
    _, _, objective, _, _, _ = system(project)
    assert objective.static is None and not objective.spaces
    assert not any(key.startswith("static.") for key in objective.state_dict())
    manifest = objective.parameter_manifest()
    assert manifest["trainable_parameters"] == manifest["total_parameters"]
    assert set(manifest["trainable_names"]) == set(dict(objective.extractor.named_parameters()))
    assert objective.fake_statistics.initialized and objective.real_reference.initialized


def test_current_reference_matches_current_weights_and_caches_only_same_version(project):
    _, _, objective, images, _, _ = system(project)
    reference = objective.real_reference
    first = reference.preview(objective.extractor, images, 0)
    features = pool_features(objective)
    torch.testing.assert_close(first.moments.mean, features.mean(0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(first.moments.cov, torch.cov(features.T), rtol=1e-12, atol=1e-12)
    reference.preview(objective.extractor, images, 0)
    assert int(reference.refreshes) == 1
    old_mean = first.moments.mean.clone()
    with torch.no_grad():
        objective.extractor.projection.weight.add_(0.05)
        objective.critic_updates.add_(1)
    current = reference.preview(objective.extractor, images, 1)
    features = pool_features(objective)
    assert int(reference.refreshes) == 2 and int(reference.feature_version) == 1
    torch.testing.assert_close(current.moments.mean, features.mean(0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(current.moments.cov, torch.cov(features.T), rtol=1e-12, atol=1e-12)
    assert not torch.equal(first.moments.mean, current.moments.mean)
    torch.testing.assert_close(first.moments.mean, old_mean, rtol=0, atol=0)
    assert not current.moments.mean.requires_grad and not current.moments.cov.requires_grad
    with pytest.raises(RuntimeError, match="Stale psi"):
        reference.validate_pending(first, 1)


@pytest.mark.parametrize("d_steps", [1, 2])
def test_C_order_refresh_frequency_and_gradient_boundaries(project, monkeypatch, d_steps):
    _, model, objective, images, g, d = system(project)
    objective.config["steps_per_update"] = d_steps
    events = []
    def wrap(owner, method, label):
        original = getattr(owner, method)
        def tracked(*args, **kwargs):
            if label == "D":
                assert all(p.grad is None for p in model.parameters())
                assert all(p.requires_grad for p in objective.critic_parameters())
            if label == "G":
                assert all(not p.requires_grad and p.grad is None for p in objective.extractor.parameters())
            events.append(label)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, method, tracked)
    wrap(model, "forward", "reconstruct")
    wrap(d, "step", "D")
    wrap(g, "step", "G")
    wrap(objective.real_reference, "refresh", "real_refresh")
    wrap(objective, "commit_dynamic", "commit")
    before = deepcopy(model.state_dict())
    rows = []
    for step in range(4):
        events.clear()
        rows.append(candidate_g_step(model, objective, g, d, images, 0, step))
        expected = ["reconstruct"] + (["D", "real_refresh"] * d_steps if step % 2 == 0 else [])
        assert events == expected + ["G", "commit"]
    assert [r["fake_ema_updates"] for r in rows] == [1, 2, 3, 4]
    assert [r["real_reference_refreshes"] for r in rows] == [1 + d_steps, 1 + d_steps, 1 + 2*d_steps, 1 + 2*d_steps]
    assert all(r["loss_psi_version"] == r["real_feature_version"] for r in rows)
    assert all(r["adv_effective_weight"] > 0 and not r["static_enabled"] for r in rows)
    assert all(v > 0 for row in rows for v in row["group_grad_norm"].values())
    for group in ("encoder", "decoder"):
        assert any(not torch.equal(before[n], p) for n, p in model.state_dict().items() if n.startswith(group))


def test_C_stale_loss_and_double_commit_rejected(project):
    _, model, objective, images, _, d = system(project)
    recon = model(images)
    old = objective.dynamic(images, recon, 0)
    critic_step(objective, d, images, recon, 0)
    with pytest.raises(RuntimeError, match="Stale psi"):
        objective.commit_dynamic(old)
    current = objective.dynamic(images, recon, 0)
    objective.commit_dynamic(current)
    with pytest.raises(RuntimeError, match="duplicate"):
        objective.commit_dynamic(current)


def test_C_iteration_matches_direct_official_kernel_with_recomputed_reference(project):
    config, model, objective, images, g, d = system(project)
    ref_model, ref_psi = deepcopy(model), deepcopy(objective.extractor).requires_grad_(True)
    ref_g = torch.optim.AdamW(ref_model.parameters(), lr=config["train"]["lr"],
                             betas=config["train"]["betas"], weight_decay=0)
    ref_d = torch.optim.AdamW(ref_psi.parameters(), lr=config["adaptive"]["lr"],
                             betas=config["adaptive"]["betas"], weight_decay=0)
    fake = FeatureStatsEMA(6, config["adaptive"]["ema_beta"])
    fake.initialize_from_mean_m2(objective.fake_statistics.mu_ema, objective.fake_statistics.m2_ema)
    pool = torch.stack([objective.real_reference.pool[i]["image"] for i in range(32)])
    reconstruction = ref_model(images)
    def distance(recon):
        with torch.no_grad():
            real = ref_psi(pool).double()
        f = ref_psi(recon)
        mean, cov = fake.build_stats(f)
        return real_whitened_frechet_distance_from_stats(real.mean(0), torch.cov(real.T), mean, cov,
                                                         eps=config["adaptive"]["whiten_eps"]), f
    value, _ = distance(reconstruction.detach())
    (-value).backward()
    torch.nn.utils.clip_grad_norm_(ref_psi.parameters(), 1)
    ref_d.step()
    ref_d.zero_grad(set_to_none=True)
    ref_psi.requires_grad_(False)
    value, f = distance(reconstruction)
    loss = config["adaptive"]["weight"] * value / (value.detach() + .01)
    loss.backward()
    ref_g.step()
    fake.update(f)
    row = candidate_g_step(model, objective, g, d, images, 0, 0)
    assert row["loss"] == pytest.approx(float(loss.detach()), abs=1e-7)
    for n, p in ref_model.state_dict().items():
        torch.testing.assert_close(p, model.state_dict()[n], rtol=1e-6, atol=1e-7)
    for n, p in ref_psi.state_dict().items():
        torch.testing.assert_close(p, objective.extractor.state_dict()[n], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(fake.m2_ema, objective.fake_statistics.m2_ema, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["ours", "ours_real_ema", "ours_fixed_reference"])
def test_C_cli_exact_resume_and_independent_evaluation(project, tmp_path, method):
    common = ["--config", str(project / "configs/ours_smoke.yaml"), "--set", f"method={method}"]
    if method != "ours":
        mode = "ema" if method == "ours_real_ema" else "frozen_initial"
        common += ["--set", f"adaptive.real_stats.mode={mode}"]
    a, b = tmp_path / "resumed", tmp_path / "continuous"
    train_main([*common, "--set", f"train.output={a}", "--set", "train.steps=2"])
    train_main([*common, "--set", f"train.output={a}", "--set", "train.steps=4", "--resume",
                str(a / "checkpoints/step_0000002.pt")])
    train_main([*common, "--set", f"train.output={b}", "--set", "train.steps=4"])
    x, y = [read_checkpoint(p / "checkpoints/step_0000004.pt") for p in (a, b)]
    def exact(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for lhs, rhs in zip(left, right):
                exact(lhs, rhs)
        else:
            assert left == right
    for field in ("model", "objective", "optimizer", "critic_optimizer"):
        exact(x[field], y[field])
    assert (a / "real_reference_manifest.json").exists()
    rows = [json.loads(line) for line in (a / "train.jsonl").read_text().splitlines()]
    if method == "ours_real_ema":
        assert rows[-1]["real_ema_updates"] == 4 and not rows[-1]["real_reference_current_psi"]
    if method == "ours_fixed_reference":
        assert rows[-1]["real_feature_version"] == 0 and rows[-1]["loss_psi_version"] == 2
    output = tmp_path / "evaluation"
    evaluate_main([*common, "--checkpoint", str(a / "checkpoints/step_0000004.pt"), "--output", str(output)])
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["engineering_only"] and metrics["num_samples"] == 32


def test_C_config_guards_and_ablations(project):
    for name in ("ours_reconstruction", "ours_reconstruction_mae", "ours_lora_mae", "ours_add_static",
                 "ours_real_ema", "ours_fixed_reference", "ours_smoke", "ours_nibi_smoke"):
        validate(load_config(project / f"configs/{name}.yaml"))
    for override in ("static.enabled=true", "adaptive.trainable_scope=lora", "adaptive.real_stats.mode=ema",
                     "adaptive.start_step=1000", "adaptive.warmup_steps=4000", "adaptive.weight=0",
                     "adaptive.ema_beta=0", "adaptive.initialization_samples=1"):
        with pytest.raises(ValueError):
            validate(load_config(project / "configs/ours_smoke.yaml", [override]))
    for override in ("adaptive.real_stats.samples=128", "adaptive.initialization_samples=128"):
        with pytest.raises(ValueError, match="50,000"):
            validate(load_config(project / "configs/ours_reconstruction.yaml", [override]))


@pytest.mark.parametrize("scope", ["full", "lora"])
def test_C_real_timm_full_vs_lora_trainable_scope(project, tmp_path, scope):
    timm = pytest.importorskip("timm")
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    path = tmp_path / "vit.pt"
    torch.save(model.state_dict(), path)
    spec = {"name": "tiny_vit", "kind": "timm", "model_name": "vit_tiny_patch16_224",
            "weights": str(path), "target_size": 32, "pool": "cls"}
    _, model, objective, images, g, d = system(project, "ours" if scope == "full" else "ours_lora", spec)
    before = deepcopy(objective.extractor.state_dict())
    candidate_g_step(model, objective, g, d, images, 0, 0)
    changed = [n for n, p in objective.extractor.state_dict().items() if not torch.equal(before[n], p)]
    assert changed
    if scope == "lora":
        assert all(".lora_" in n for n in changed)
    else:
        assert not any(".lora_" in n for n in objective.trainable_names)
        assert set(objective.trainable_names) == set(dict(objective.extractor.named_parameters()))


def test_C_add_static_is_explicit_and_retains_frozen_static(project, tmp_path):
    config = load_config(project / "configs/ours_smoke.yaml")
    config["method"] = "ours_add_static"
    config["static"] = load_config(project / "configs/smoke.yaml")["static"]
    config["static"]["reference_cache"] = str(tmp_path / "cache")
    validate(config)
    dataset = SyntheticDataset(64, 16, 217)
    model = TinyReconstructor()
    objective, _ = build_objective(config, dataset, "cpu")
    initialize_statistics(model, objective, dataset, config, "cpu")
    before = deepcopy(objective.static.spaces["tiny"].extractor.state_dict())
    g = torch.optim.AdamW(model.parameters(), lr=1e-4)
    d = torch.optim.AdamW(objective.critic_parameters(), lr=1e-4)
    row = candidate_g_step(model, objective, g, d, torch.rand(8, 3, 16, 16), 0, 0)
    assert row["static_enabled"] and row["statistics_updates"] == {"tiny": 1}
    for n, p in objective.static.spaces["tiny"].extractor.state_dict().items():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def test_C_real_autoencoder_kl_joint_training(project, tmp_path):
    diffusers = pytest.importorskip("diffusers")
    from recon_fd.tokenizers import KLReconstructor
    raw = diffusers.AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=16)
    raw.save_pretrained(tmp_path)
    model = KLReconstructor(str(tmp_path), gradient_checkpointing=True)
    config = load_config(project / "configs/ours_smoke.yaml")
    dataset = SyntheticDataset(64, 16, 217)
    objective, _ = build_objective(config, dataset, "cpu")
    initialize_statistics(model, objective, dataset, config, "cpu")
    g = torch.optim.AdamW(model.parameters(), lr=1e-5)
    d = torch.optim.AdamW(objective.critic_parameters(), lr=1e-5)
    row = candidate_g_step(model, objective, g, d, torch.rand(2, 3, 16, 16), 0, 0)
    assert all(value > 0 for value in row["group_grad_norm"].values())
    assert row["real_feature_version"] == row["loss_psi_version"] == 1
