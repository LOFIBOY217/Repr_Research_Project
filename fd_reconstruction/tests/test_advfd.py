from copy import deepcopy
import ast
import json
import pytest
import torch
from torch import nn
from conftest import upstream_module
from recon_fd.config import load_config, validate
from recon_fd.cli import train_main, evaluate_main
from recon_fd.engine.adversarial import adversarial_g_step, critic_step
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.objectives.adaptive_fd import AdvFD, AdvStatsEMA
from recon_fd.objectives.whitening import real_whitened_frechet_distance
from recon_fd.objectives.static_fd import StaticSpace, StaticFD
from recon_fd.objectives.statistics import RunningMoments, EMAStats
from recon_fd.representations.lora import LoRAQKVLinear, apply_qkv_lora


def settings(project):
    return load_config(project / "configs/advfd_smoke.yaml")["adaptive"]


def upstream(project):
    return upstream_module(project / "third_party/AdvFD/frechet_distance/adversarial.py", "upstream_advfd")


@pytest.mark.parametrize("samples", [4, 40])
def test_whitened_fd_value_gradient_and_detach_parity(project, samples):
    official = upstream(project)
    real = torch.randn(samples, 7, dtype=torch.float64, requires_grad=True)
    fake = torch.randn(samples + 1, 7, dtype=torch.float64, requires_grad=True)
    arguments = (real.mean(0), torch.cov(real.T), fake.mean(0), torch.cov(fake.T))
    ours = real_whitened_frechet_distance(*arguments, eps=0.001)
    theirs = official.real_whitened_frechet_distance_from_stats(*arguments, eps=0.001)
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
    a = torch.autograd.grad(ours, (real, fake), allow_unused=True, retain_graph=True)
    b = torch.autograd.grad(theirs, (real, fake), allow_unused=True)
    assert a[0] is b[0] is None  # Includes the detached whitening geometry.
    torch.testing.assert_close(a[1], b[1], rtol=1e-10, atol=1e-10)
    assert torch.isfinite(a[1]).all()
    identity = real_whitened_frechet_distance(arguments[0], arguments[1], arguments[0], arguments[1])
    assert abs(float(identity.detach())) < 1e-6


def test_adv_ema_initialization_preview_update_parity(project):
    official = upstream(project).FeatureStatsEMA(5, 0.99)
    ours = AdvStatsEMA(5, 0.99)
    initial = torch.randn(24, 5, dtype=torch.float64)
    mu, cov = initial.mean(0), torch.cov(initial.T)
    official.initialize_from_mean_cov(mu, cov)
    ours.initialize_mean_cov(mu, cov)
    # Deliberately NOT sample->population rescaled, unlike FD-only init.
    torch.testing.assert_close(ours.second, official.m2_ema, rtol=0, atol=0)
    for step in range(3):
        features = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        before = deepcopy(ours.state_dict())
        a = ours.preview(features)
        b_mu, b_cov = official.build_stats(features)
        torch.testing.assert_close(a.mean, b_mu, rtol=0, atol=0)
        torch.testing.assert_close(a.cov, b_cov, rtol=1e-13, atol=1e-13)
        for key, value in before.items():
            torch.testing.assert_close(value, ours.state_dict()[key], rtol=0, atol=0)
        a_grad = torch.autograd.grad(a.cov.square().sum(), features, retain_graph=True)[0]
        b_grad = torch.autograd.grad(b_cov.square().sum(), features)[0]
        torch.testing.assert_close(a_grad, b_grad, rtol=1e-12, atol=1e-12)
        ours.commit(features, step)
        official.update(features)
        torch.testing.assert_close(ours.mean, official.mu_ema)
        torch.testing.assert_close(ours.second, official.m2_ema)
    with pytest.raises(RuntimeError, match="duplicate"):
        ours.commit(features, 2)


def test_full_iteration_matches_upstream_loss_and_update_rules(project, small_system):
    model, static, _, images = small_system
    config = settings(project)
    config.update(start_step=1, warmup_steps=0, lr=0.001)
    objective = AdvFD(static, config)
    reference_model = deepcopy(model)
    reference_static = deepcopy(static)
    reference_psi = deepcopy(objective.extractor).requires_grad_(True)
    official = upstream(project)
    reference_real = official.FeatureStatsEMA(6, config["ema_beta"])
    reference_fake = official.FeatureStatsEMA(6, config["ema_beta"])
    space = reference_static.spaces["tiny"]
    reference_real.initialize_from_mean_cov(space.reference_mean, space.reference_cov)
    reference_fake.initialize_from_mean_m2(space.statistics.mean, space.statistics.second)
    g_opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0)
    ref_g_opt = torch.optim.AdamW(reference_model.parameters(), lr=0.0001, weight_decay=0)
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=config["lr"], betas=config["betas"], weight_decay=0)
    ref_d_opt = torch.optim.AdamW(reference_psi.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=0)

    reconstruction = reference_model(images)

    def ref_dynamic(fake):
        with torch.no_grad():
            real_features = reference_psi(images)
        fake_features = reference_psi(fake)
        r_mu, r_cov = reference_real.build_stats(real_features)
        f_mu, f_cov = reference_fake.build_stats(fake_features)
        value = official.real_whitened_frechet_distance_from_stats(r_mu, r_cov, f_mu, f_cov, eps=0.001)
        return value, real_features, fake_features.detach()

    critic_value, _, _ = ref_dynamic(reconstruction.detach())
    (-critic_value).backward()
    torch.nn.utils.clip_grad_norm_(reference_psi.parameters(), config["grad_clip"])
    ref_d_opt.step()
    ref_d_opt.zero_grad(set_to_none=True)
    reference_psi.requires_grad_(False)
    main = reference_static(reconstruction)
    dynamic, real_update, fake_update = ref_dynamic(reconstruction)
    loss = main.loss + config["weight"] * dynamic / (dynamic.detach() + 0.01)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1)
    ref_g_opt.step()
    reference_static.commit(main)
    reference_real.update(real_update)
    reference_fake.update(fake_update)

    metrics = adversarial_g_step(model, objective, g_opt, d_opt, images, 1, 1)
    assert metrics["loss"] == pytest.approx(float(loss.detach()), abs=1e-7)
    assert metrics["critic_fd"] == pytest.approx(float(critic_value.detach()), abs=1e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference_model.state_dict()[name], rtol=0, atol=0)
    for name, value in objective.extractor.state_dict().items():
        torch.testing.assert_close(value, reference_psi.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(objective.real_statistics.second, reference_real.m2_ema)
    torch.testing.assert_close(objective.fake_statistics.second, reference_fake.m2_ema)


def test_critic_detach_and_ema_commit_purity(project, small_system):
    model, static, _, images = small_system
    config = settings(project)
    config.update(start_step=1, warmup_steps=0)
    objective = AdvFD(static, config)
    objective.initialize_fake_at_activation()
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=1e-4)
    reconstruction = model(images)
    state = deepcopy(model.state_dict())
    stats = deepcopy(objective.real_statistics.state_dict())
    critic_step(objective, d_opt, images, reconstruction, 1)
    assert all(p.grad is None for p in model.parameters())
    for name, value in state.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    for name, value in stats.items():
        torch.testing.assert_close(value, objective.real_statistics.state_dict()[name], rtol=0, atol=0)
    result = objective.dynamic(images, reconstruction, 1)
    again = objective.dynamic(images, reconstruction, 1)
    torch.testing.assert_close(result.fd, again.fd)
    objective.commit_dynamic(result)
    with pytest.raises(RuntimeError, match="Duplicate"):
        objective.commit_dynamic(again)


def test_activation_schedule_real_frequency_and_static_retention(project, small_system):
    model, static, g_opt, images = small_system
    config = settings(project)
    config.update(start_step=2, warmup_steps=2)
    config["real_stats"]["update_freq"] = 2
    objective = AdvFD(static, config)
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=1e-4)
    initial_static = deepcopy(static.spaces["tiny"].extractor.state_dict())
    initial_psi = deepcopy(objective.extractor.state_dict())
    rows = []
    for step in range(1, 7):
        if step == 2:
            # Activation copies the UPDATED static EMA, not its initial value.
            objective.initialize_fake_at_activation()
            torch.testing.assert_close(objective.fake_statistics.second, static.spaces["tiny"].statistics.second)
        rows.append(adversarial_g_step(model, objective, g_opt, d_opt, images, 1, step))
        if step == 1:
            for name, value in initial_psi.items():
                torch.testing.assert_close(value, objective.extractor.state_dict()[name], rtol=0, atol=0)
    assert [r["adv_effective_weight"] for r in rows] == [0, 0, 0.05, 0.1, 0.1, 0.1]
    assert [r["critic_updates"] for r in rows] == [0, 1, 1, 2, 2, 3]
    assert [r["adv_statistics_updates"]["real"] for r in rows] == [0, 1, 1, 2, 2, 3]
    assert [r["adv_statistics_updates"]["fake"] for r in rows] == [0, 1, 2, 3, 4, 5]
    assert all(row["statistics_updates"]["tiny"] == i for i, row in enumerate(rows, 1))
    for name, value in initial_static.items():
        torch.testing.assert_close(value, static.spaces["tiny"].extractor.state_dict()[name], rtol=0, atol=0)


def test_lora_numerical_parity_with_official_class(project):
    # Read only the self-contained upstream class; avoid its unrelated imports.
    path = project / "third_party/AdvFD/frechet_distance/repr_models.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LoRAQKVLinear")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    base = nn.Linear(12, 36)
    a = LoRAQKVLinear(deepcopy(base), 16, 16)
    b = namespace["LoRAQKVLinear"](deepcopy(base), 16, 16)
    a.lora_B.weight.data.normal_(std=0.01)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 4, 12, requires_grad=True)
    y_a, y_b = a(x), b(x)
    torch.testing.assert_close(y_a, y_b, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(y_a.square().sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(y_b.square().sum(), x)[0], rtol=0, atol=0)


def test_timm_lora_only_updates_adapters_and_retains_input_grad():
    timm = pytest.importorskip("timm")
    wrapper = nn.Module()
    wrapper.model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                                     dynamic_img_size=True, dynamic_img_pad=True)
    wrapper.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        original_output = wrapper.model.forward_features(x)
    count = apply_qkv_lora(wrapper)
    assert count == 12
    selected = [(name, p) for name, p in wrapper.named_parameters() if p.requires_grad]
    assert selected and all(".lora_A." in n or ".lora_B." in n for n, p in selected)
    initial = deepcopy(wrapper.state_dict())
    wrapper.eval()
    with torch.no_grad():
        torch.testing.assert_close(wrapper.model.forward_features(x), original_output, rtol=0, atol=0)
    wrapper.model.set_grad_checkpointing(True)
    optimizer = torch.optim.AdamW([p for _, p in selected], lr=0.01)
    wrapper.model.forward_features(x).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in selected)
    optimizer.step()
    for name, value in wrapper.state_dict().items():
        if ".lora_" not in name:
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    wrapper.requires_grad_(False)
    x.requires_grad_(True)
    wrapper.model.forward_features(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_actual_timm_lora_in_B_training_engine(project, tmp_path):
    timm = pytest.importorskip("timm")
    from recon_fd.representations import build_representation
    from recon_fd.tokenizers import TinyReconstructor
    raw = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    weights = tmp_path / "random_vit.pt"
    torch.save(raw.state_dict(), weights)
    extractor = build_representation({"kind": "timm", "name": "tiny", "model_name": "vit_tiny_patch16_224",
                                      "weights": str(weights), "target_size": 32, "pool": "cls"})
    images = torch.rand(8, 3, 16, 16)
    model = TinyReconstructor()
    real, fake = RunningMoments(), RunningMoments()
    real.update(extractor(images))
    fake.update(extractor(model(images)))
    state = EMAStats(192, 0.999)
    state.initialize(fake.moments())
    static = StaticFD({"tiny": StaticSpace(extractor, real.moments(), state)})
    config = settings(project)
    config.update(start_step=0, warmup_steps=0, gradient_checkpointing=True)
    # Direct construction exercises adapter plumbing on a small random ViT;
    # the scientific CLI still rejects this non-paper backbone for B.
    objective = AdvFD(static, config)
    before = deepcopy(objective.extractor.state_dict())
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=1e-4)
    g_opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    result = adversarial_g_step(model, objective, g_opt, d_opt, images, 1, 0)
    assert result["critic_updates"] == 1 and result["adv_statistics_updates"] == {"real": 1, "fake": 1}
    assert all(value > 0 for value in result["group_grad_norm"].values())
    changed = [name for name, value in objective.extractor.state_dict().items() if not torch.equal(before[name], value)]
    assert changed and all(".lora_" in name for name in changed)


def test_missing_adversarial_optimizer_checkpoint_rejected(project, tmp_path):
    config = str(project / "configs/advfd_smoke.yaml")
    common = ["--config", config, "--set", f"static.reference_cache={tmp_path / 'cache'}",
              "--set", f"train.output={tmp_path / 'train'}", "--set", "train.steps=1"]
    train_main(common)
    checkpoint = read_checkpoint(tmp_path / "train/checkpoints/step_0000001.pt")
    checkpoint.pop("critic_optimizer")
    malformed = tmp_path / "missing_critic.pt"
    torch.save(checkpoint, malformed)
    with pytest.raises(ValueError, match="adversarial optimizer missing"):
        train_main([*common, "--set", "train.steps=2", "--resume", str(malformed)])


def test_full_scope_keeps_bn_buffers_but_trains_affine(project):
    class BatchNormFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(3, 6, 1), nn.BatchNorm2d(6))
            self.dimension = 6
            self.identity = {"spec": {"kind": "inception"}}

        def forward(self, x):
            return self.layers(x).mean((2, 3))

    extractor = BatchNormFixture().eval().requires_grad_(False)
    images = torch.rand(12, 3, 8, 8)
    accumulator = RunningMoments()
    accumulator.update(extractor(images))
    state = EMAStats(6, 0.999)
    state.initialize(accumulator.moments())
    config = settings(project)
    config["start_step"] = 0
    objective = AdvFD(StaticFD({"tiny": StaticSpace(extractor, accumulator.moments(), state)}), config)
    objective.initialize_fake_at_activation()
    before = deepcopy(objective.extractor.state_dict())
    assert set(objective.trainable_names) == set(dict(objective.extractor.named_parameters()))
    optimizer = torch.optim.AdamW(objective.critic_parameters(), lr=0.01)
    objective.train()
    assert not objective.extractor.layers[1].training
    critic_step(objective, optimizer, images, images * 0.5, 0)
    after = objective.extractor.state_dict()
    for name in ("layers.1.running_mean", "layers.1.running_var", "layers.1.num_batches_tracked"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert not torch.equal(before["layers.1.weight"], after["layers.1.weight"])


def test_B_config_guards_and_recipes(project):
    for name in ("advfd_reconstruction", "advfd_reconstruction_inception", "advfd_reconstruction_mae",
                 "advfd_reconstruction_siglip", "advfd_smoke"):
        validate(load_config(project / f"configs/{name}.yaml"))
    for override in ("static.enabled=false", "adaptive.trainable_scope=full", "adaptive.real_stats.mode=reencode_pool",
                     "adaptive.lora.rank=8", "adaptive.whiten_eps=0"):
        with pytest.raises(ValueError):
            validate(load_config(project / "configs/advfd_reconstruction.yaml", [override]))


@pytest.mark.parametrize("split", [1, 3])
def test_advfd_cli_resume_exact_before_and_after_activation(project, tmp_path, split):
    config = str(project / "configs/advfd_smoke.yaml")
    common = ["--config", config, "--set", f"static.reference_cache={tmp_path / 'cache'}"]
    resumed, continuous = tmp_path / "resumed", tmp_path / "continuous"
    train_main([*common, "--set", f"train.output={resumed}", "--set", f"train.steps={split}"])
    train_main([*common, "--set", f"train.output={resumed}", "--set", "train.steps=6", "--resume",
                str(resumed / f"checkpoints/step_{split:07d}.pt")])
    train_main([*common, "--set", f"train.output={continuous}", "--set", "train.steps=6"])
    a = read_checkpoint(resumed / "checkpoints/step_0000006.pt")
    b = read_checkpoint(continuous / "checkpoints/step_0000006.pt")

    def exact(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x:
                exact(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for lhs, rhs in zip(x, y):
                exact(lhs, rhs)
        else:
            assert x == y

    for field in ("model", "objective", "optimizer", "critic_optimizer"):
        exact(a[field], b[field])
    output = tmp_path / "evaluation"
    evaluate_main([*common, "--checkpoint", str(resumed / "checkpoints/step_0000006.pt"), "--output", str(output)])
    report = json.loads((output / "metrics.json").read_text())
    assert report["engineering_only"] and report["checkpoint_step"] == 6
    assert set(report["fd"]) == {"tiny", "tiny_holdout"}
    assert (resumed / "advfd_parameters.json").is_file()
    rows = [json.loads(line) for line in (resumed / "train.jsonl").read_text().splitlines()]
    assert [row["adv_schedule_step"] for row in rows] == list(range(6))
    assert [row["adv_active"] for row in rows] == [False, False, True, True, True, True]
    assert [row["critic_updates"] for row in rows] == [0, 0, 1, 1, 2, 2]
