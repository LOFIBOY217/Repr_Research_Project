from copy import deepcopy
import json
import pytest
import torch
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset
from recon_fd.tokenizers import TinyReconstructor
from recon_fd.engine.trainer import build_objective, initialize_statistics
from recon_fd.engine.current_both import full_pool_backward, current_both_g_step
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.objectives.whitening import real_whitened_frechet_distance
from recon_fd.cli import train_main, evaluate_main


def system(project, count=31, batch_size=7, model=None, spec=None):
    config = load_config(project / "configs/ours_current_both_smoke.yaml")
    config["adaptive"]["real_stats"]["samples"] = count
    config["adaptive"]["real_stats"]["batch_size"] = batch_size
    config["adaptive"]["initialization_samples"] = count
    config["adaptive"]["initialization_batch_size"] = batch_size
    config["train"]["batch_size"] = batch_size
    if spec is not None:
        config["adaptive"]["representation"] = spec
        config["adaptive"]["gradient_checkpointing"] = spec["kind"] == "timm"
    validate(config)
    dataset = SyntheticDataset(64, 16, 217)
    model = model if model is not None else TinyReconstructor()
    objective, _ = build_objective(config, dataset, "cpu")
    initialize_statistics(model, objective, dataset, config, "cpu")
    model.eval()
    g = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                         betas=config["train"]["betas"], weight_decay=0)
    d = torch.optim.AdamW(objective.critic_parameters(), lr=config["adaptive"]["lr"],
                         betas=config["adaptive"]["betas"], weight_decay=0)
    return config, model, objective, g, d


def pool_images(objective):
    return torch.stack([objective.real_reference.pool[i]["image"] for i in range(len(objective.real_reference.pool))])


def direct_fd(model, psi, images, eps):
    with torch.no_grad():
        real = psi(images).double()
    fake = psi(model(images)).double()
    return real_whitened_frechet_distance(real.mean(0), torch.cov(real.T),
                                          fake.mean(0), torch.cov(fake.T), eps)


def test_no_ema_and_cache_tracks_both_parameter_versions(project):
    _, model, objective, _, _ = system(project)
    assert objective.fake_statistics is None and objective.real_reference.ema is None
    assert not any("ema" in name for name in objective.state_dict())
    assert objective.static is None
    manifest = objective.parameter_manifest()
    assert manifest["fake_statistics"] == "current_model_reencoded_pool"
    first = objective.current_pair(model)
    images = pool_images(objective)
    with torch.no_grad():
        features = objective.extractor(model(images)).double()
    torch.testing.assert_close(first.fake.mean, features.mean(0), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(first.fake.cov, torch.cov(features.T), rtol=1e-5, atol=1e-9)
    objective.current_pair(model)
    assert int(objective.fake_reference.refreshes) == 1
    original = first.fake.mean.clone()
    with torch.no_grad():
        model.decoder[0].bias.add_(0.05)
        objective.generator_updates.add_(1)
    with pytest.raises(RuntimeError, match="Stale E\\+D"):
        objective.validate_pair(first)
    second = objective.current_pair(model)
    assert int(objective.fake_reference.refreshes) == 2
    assert int(objective.real_reference.refreshes) == 1
    assert not torch.equal(first.fake.mean, second.fake.mean)
    torch.testing.assert_close(first.fake.mean, original, rtol=0, atol=0)
    with torch.no_grad():
        objective.extractor.projection.weight.add_(0.1)
        objective.critic_updates.add_(1)
    with pytest.raises(RuntimeError, match="Stale psi"):
        objective.validate_pair(second)
    third = objective.current_pair(model)
    assert int(objective.fake_reference.refreshes) == 3
    assert int(objective.real_reference.refreshes) == 2
    assert third.real.feature_psi_version == third.psi_version == 1
    with pytest.raises(RuntimeError, match="full-pool"):
        objective.dynamic(None, None, 0)


@pytest.mark.parametrize("phase", ["D", "G"])
@pytest.mark.parametrize("batch_size", [3, 7])
def test_two_pass_gradients_match_direct_all_samples_graph(project, phase, batch_size):
    config, model, objective, _, _ = system(project, batch_size=batch_size)
    direct_model, direct_psi = deepcopy(model), deepcopy(objective.extractor)
    model.requires_grad_(phase == "G")
    objective.set_critic_trainable(phase == "D")
    direct_model.requires_grad_(phase == "G")
    direct_psi.requires_grad_(phase == "D")
    images = pool_images(objective)
    fd = direct_fd(direct_model, direct_psi, images, config["adaptive"]["whiten_eps"])
    loss = -fd if phase == "D" else config["adaptive"]["weight"] * fd / (fd.detach() + .01)
    loss.backward()
    pair, actual_fd, actual_loss = full_pool_backward(model, objective, phase)
    torch.testing.assert_close(actual_fd, fd.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual_loss, loss.detach(), rtol=1e-5, atol=1e-7)
    assert int(objective.gradient_feature_images) == len(images) == pair.fake.count
    for left, right in ((model, direct_model), (objective.extractor, direct_psi)):
        for name, parameter in left.named_parameters():
            expected = dict(right.named_parameters())[name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=5e-4, atol=2e-6)


@pytest.mark.parametrize("d_steps", [1, 2])
def test_full_pool_optimizer_order_and_versions(project, monkeypatch, d_steps):
    _, model, objective, g, d = system(project)
    objective.config["steps_per_update"] = d_steps
    events = []
    old_d, old_g = d.step, g.step
    def d_step():
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        events.append("D")
        return old_d()
    def g_step():
        assert all(p.grad is None and not p.requires_grad for p in objective.extractor.parameters())
        events.append("G")
        return old_g()
    monkeypatch.setattr(d, "step", d_step)
    monkeypatch.setattr(g, "step", g_step)
    rows = []
    for step in range(4):
        events.clear()
        rows.append(current_both_g_step(model, objective, g, d, 0, step))
        assert events == (["D"] * d_steps if step % 2 == 0 else []) + ["G"]
    assert [r["generator_updates"] for r in rows] == [1, 2, 3, 4]
    assert [r["fake_reference_refreshes"] for r in rows] == [1+d_steps, 2+d_steps, 3+2*d_steps, 4+2*d_steps]
    for row in rows:
        assert row["fake_feature_psi_version"] == row["real_feature_version"] == row["loss_psi_version"]
        assert row["fake_feature_generator_version"] == row["loss_generator_version"] == row["generator_updates"]-1
        assert not row["fake_cache_current_after_G"]
        assert not row["fake_statistics_historical"] and row["fake_ema_updates"] == row["real_ema_updates"] == 0
        assert row["real_reference_pool_samples"] == row["fake_reference_pool_samples"] == 31
        assert all(value > 0 for value in row["group_grad_norm"].values())


def test_complete_D_then_G_update_matches_direct_pool_oracle(project):
    config, model, objective, g, d = system(project)
    expected_model, expected_psi = deepcopy(model), deepcopy(objective.extractor)
    expected_g = torch.optim.AdamW(expected_model.parameters(), lr=config["train"]["lr"],
                                  betas=config["train"]["betas"], weight_decay=0)
    expected_d = torch.optim.AdamW(expected_psi.parameters(), lr=config["adaptive"]["lr"],
                                  betas=config["adaptive"]["betas"], weight_decay=0)
    images = pool_images(objective)
    expected_model.requires_grad_(False)
    expected_psi.requires_grad_(True)
    (-direct_fd(expected_model, expected_psi, images, .001)).backward()
    torch.nn.utils.clip_grad_norm_(expected_psi.parameters(), 1)
    expected_d.step()
    expected_d.zero_grad(set_to_none=True)
    expected_psi.requires_grad_(False)
    expected_model.requires_grad_(True)
    fd = direct_fd(expected_model, expected_psi, images, .001)
    (config["adaptive"]["weight"] * fd / (fd.detach()+.01)).backward()
    expected_g.step()
    row = current_both_g_step(model, objective, g, d, 0, 0)
    assert row["adv_fd"] == pytest.approx(float(fd.detach()), rel=2e-5)
    for left, right in ((model, expected_model), (objective.extractor, expected_psi)):
        for name, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[name], rtol=1e-5, atol=1e-6)


def test_replay_mismatch_fails_before_optimizer_update(project, monkeypatch):
    _, model, objective, g, d = system(project)
    before = deepcopy(model.state_dict())
    # Stale fake cache deliberately emulates a stochastic/stateful forward.
    objective.fake_reference.mean.add_(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("An optimizer must not step after replay mismatch")
    monkeypatch.setattr(g, "step", forbidden)
    monkeypatch.setattr(d, "step", forbidden)
    with pytest.raises(RuntimeError, match="Replayed features differ"):
        current_both_g_step(model, objective, g, d, 0, 0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_current_both_cli_resume_and_independent_evaluation(project, tmp_path):
    common = ["--config", str(project / "configs/ours_current_both_smoke.yaml")]
    a, b = tmp_path / "resumed", tmp_path / "continuous"
    train_main([*common, "--set", f"train.output={a}", "--set", "train.steps=2"])
    train_main([*common, "--config", str(a / "config.json"),
                "--set", f"train.output={a}", "--set", "train.steps=4", "--resume",
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
    real = json.loads((a / "real_reference_manifest.json").read_text())
    fake = json.loads((a / "reconstruction_reference_manifest.json").read_text())
    assert real["sample_ids"] == fake["sample_ids"] and fake["paired_with_real"]
    rows = [json.loads(line) for line in (a / "train.jsonl").read_text().splitlines()]
    assert [r["samples_seen"] for r in rows] == [32, 64, 96, 128]
    assert all(r["fd_batch_size"] == 32 and r["feature_microbatch_size"] == 8 for r in rows)
    assert not any("ema" in n for n in x["objective"])
    output = tmp_path / "evaluation"
    evaluate_main([*common, "--checkpoint", str(a / "checkpoints/step_0000004.pt"), "--output", str(output)])
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["engineering_only"] and metrics["num_samples"] == 32


def test_current_both_config_guards(project):
    for name in ("ours_current_both", "ours_current_both_smoke", "ours_current_both_nibi_smoke"):
        validate(load_config(project / f"configs/{name}.yaml"))
    for override in ("adaptive.fake_stats.mode=ema", "adaptive.fake_stats.gradient=batch_only",
                     "adaptive.initialization_samples=16", "adaptive.real_stats.mode=ema", "static.enabled=true",
                     "adaptive.initialization_batch_size=4"):
        with pytest.raises(ValueError):
            validate(load_config(project / "configs/ours_current_both_smoke.yaml", [override]))
    with pytest.raises(ValueError, match="50,000"):
        validate(load_config(project / "configs/ours_current_both.yaml", [
            "adaptive.real_stats.samples=128", "adaptive.initialization_samples=128"]))


def test_current_both_AutoencoderKL_checkpointed_joint_gradients(project, tmp_path):
    diffusers = pytest.importorskip("diffusers")
    from recon_fd.tokenizers import KLReconstructor
    raw = diffusers.AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4, norm_num_groups=4, sample_size=16)
    raw.save_pretrained(tmp_path)
    model = KLReconstructor(str(tmp_path), gradient_checkpointing=True)
    _, model, objective, g, d = system(project, count=8, batch_size=2, model=model)
    row = current_both_g_step(model, objective, g, d, 0, 0)
    assert row["tokenizer_mode"] == "eval_with_grad" and not model.training
    assert all(value > 0 for value in row["group_grad_norm"].values())


def test_current_both_checkpointed_timm_full_parameter_update(project, tmp_path):
    timm = pytest.importorskip("timm")
    raw = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    path = tmp_path / "vit.pt"
    torch.save(raw.state_dict(), path)
    spec = {"name": "tiny_vit", "kind": "timm", "model_name": "vit_tiny_patch16_224",
            "weights": str(path), "target_size": 32, "pool": "cls"}
    _, model, objective, g, d = system(project, count=8, batch_size=2, spec=spec)
    before = deepcopy(objective.extractor.state_dict())
    row = current_both_g_step(model, objective, g, d, 0, 0)
    assert any(not torch.equal(value, before[name]) for name, value in objective.extractor.state_dict().items())
    assert objective.parameter_manifest()["trainable_parameters"] == objective.parameter_manifest()["total_parameters"]
    assert all(value > 0 for value in row["group_grad_norm"].values())


def test_eval_mode_preserves_BN_buffers_while_training_affine(project):
    model = TinyReconstructor()
    model.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1),
                                        torch.nn.BatchNorm2d(8), torch.nn.Tanh())
    _, model, objective, g, d = system(project, model=model)
    before = {name: value.clone() for name, value in model.named_buffers()}
    current_both_g_step(model, objective, g, d, 0, 0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert model.encoder[1].weight.grad is not None
