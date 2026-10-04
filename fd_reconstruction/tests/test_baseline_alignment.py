"""Check published-source identity and the reconstruction adapters' boundaries."""
import ast
from copy import deepcopy
import pytest
import torch
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset
from recon_fd.engine.trainer import build_objective, initialize_statistics
from recon_fd.engine.gradients import checked_grad_norm
from recon_fd.objectives.adaptive_fd import AdvStatsEMA
from recon_fd.objectives.official_fd import OfficialStaticFD
from conftest import upstream_module


def test_official_adversarial_file_and_lora_class_unchanged(project):
    original = project / "third_party/AdvFD/frechet_distance"
    packaged = project / "src/recon_fd/vendor/advfd"
    assert (original / "adversarial.py").read_bytes() == (packaged / "adversarial.py").read_bytes()

    def class_source(path):
        source = path.read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "LoRAQKVLinear")
        return ast.get_source_segment(source, node)

    assert class_source(original / "repr_models.py") == class_source(packaged / "lora.py")
    # apply_patch adds a terminal newline; the original file has none.
    assert ((project / "third_party/FD-Loss/utils/grad_util.py").read_bytes().rstrip(b"\n")
            == (project / "src/recon_fd/vendor/fd_loss/grad_util.py").read_bytes().rstrip(b"\n"))


def test_a_and_b_static_branches_have_identical_values_gradients_and_state(project, tmp_path):
    configs = [load_config(project / "configs" / name) for name in ("smoke.yaml", "advfd_smoke.yaml")]
    # Compare algorithms under identical static settings, not different smoke betas.
    configs[1]["static"] = deepcopy(configs[0]["static"])
    dataset = SyntheticDataset(64, 16, 217)
    from recon_fd.tokenizers import TinyReconstructor
    model = TinyReconstructor()
    objectives = []
    for config in configs:
        config["static"]["reference_cache"] = str(tmp_path / "reference")
        objective, _ = build_objective(config, dataset, "cpu")
        initialize_statistics(model, objective, dataset, config, "cpu")
        objectives.append(objective)
    a, b = objectives[0], objectives[1].static
    assert isinstance(a, OfficialStaticFD) and isinstance(b, OfficialStaticFD)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    x = torch.rand(8, 3, 16, 16, requires_grad=True)
    fa, fb = a(x), b(x)
    torch.testing.assert_close(fa.loss, fb.loss, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(fa.loss, x, retain_graph=True)[0],
                               torch.autograd.grad(fb.loss, x)[0], rtol=0, atol=0)
    a.commit(fa)
    b.commit(fb)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("initialized", [False, True])
def test_adversarial_ema_multistep_exact_original_buffers(project, initialized):
    module = upstream_module(project / "third_party/AdvFD/frechet_distance/adversarial.py", "adv_stats_oracle")
    ours, oracle = AdvStatsEMA(5, .99), module.FeatureStatsEMA(5, .99)
    if initialized:
        real = torch.randn(21, 5).double()
        ours.initialize_mean_cov(real.mean(0), torch.cov(real.T))
        oracle.initialize_from_mean_cov(real.mean(0), torch.cov(real.T))
    for step in range(12):
        features = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        before = deepcopy(ours.state_dict())
        a = ours.preview(features)
        mu, cov = oracle.build_stats(features)
        for key, value in before.items():
            torch.testing.assert_close(value, ours.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(a.mean, mu, rtol=0, atol=0)
        torch.testing.assert_close(a.cov, cov, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a.cov.square().sum(), features, retain_graph=True)[0],
                                   torch.autograd.grad(cov.square().sum(), features)[0], rtol=0, atol=0)
        ours.commit(features, step)
        oracle.update(features)
        for key, value in oracle.state_dict().items():
            torch.testing.assert_close(value, ours.state_dict()[key], rtol=0, atol=0)


def test_generator_no_clipping_matches_original_and_rejects_nonfinite(project):
    config = load_config(project / "configs/smoke.yaml")
    validate(config)
    assert config["train"]["grad_clip"] == 0
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = torch.tensor([30., 40., 0.])
    before = parameter.grad.clone()
    assert float(checked_grad_norm([parameter], 0)) == 50
    torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
    parameter.grad[0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        checked_grad_norm([parameter], 0)


def test_B_1000_start_4000_warmup_and_frequency(project, official_small_system):
    from recon_fd.objectives.adaptive_fd import AdvFD
    adaptive = load_config(project / "configs/advfd_reconstruction.yaml")["adaptive"]
    objective = AdvFD(official_small_system[1], {**adaptive, "representation": "tiny"})
    assert not objective.active(999)
    assert objective.active(1000) and objective.critic_due(1000)
    assert not objective.critic_due(1001) and objective.critic_due(1002)
    assert objective.effective_weight(1000) == 0
    assert objective.effective_weight(3000) == .05
    assert objective.effective_weight(5000) == .1
    assert adaptive["steps_per_update"] == 1 and adaptive["grad_clip"] == 1


def test_B_code_order_one_reconstruction_then_D_then_G_then_commits(project, official_small_system, monkeypatch):
    from recon_fd.engine.adversarial import adversarial_g_step
    from recon_fd.objectives.adaptive_fd import AdvFD
    model, static, _, images = official_small_system
    adaptive = load_config(project / "configs/advfd_smoke.yaml")["adaptive"]
    adaptive.update(start_step=0, warmup_steps=0)
    objective = AdvFD(static, adaptive)
    g_opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=adaptive["lr"],
                              betas=adaptive["betas"], weight_decay=0)
    events = []

    def record(owner, method, label):
        original = getattr(owner, method)
        def wrapped(*args, **kwargs):
            events.append(label)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, method, wrapped)

    record(model, "forward", "reconstruct")
    record(d_opt, "step", "D")
    record(g_opt, "step", "G")
    record(objective, "commit", "static_commit")
    record(objective, "commit_dynamic", "dynamic_commit")
    for step in (0, 1):
        events.clear()
        adversarial_g_step(model, objective, g_opt, d_opt, images, 0, step)
        expected = ["reconstruct"] + (["D"] if step == 0 else [])
        assert events == expected + ["G", "static_commit", "dynamic_commit"]
