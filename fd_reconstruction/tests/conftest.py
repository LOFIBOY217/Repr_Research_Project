import importlib.util
from pathlib import Path
import pytest
import torch
from recon_fd.objectives.statistics import RunningMoments
from recon_fd.objectives.static_fd import StaticSpace
from recon_fd.objectives.official_fd import OfficialFDStatistics, OfficialStaticFD, precompute_sigma_ref_sqrt
from recon_fd.representations import build_representation
from recon_fd.tokenizers import TinyReconstructor


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def project():
    return Path(__file__).resolve().parents[1]


def upstream_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_system(official_small_system):
    return official_small_system


@pytest.fixture
def official_small_system():
    model = TinyReconstructor()
    extractor = build_representation({"name": "tiny", "kind": "tiny", "seed": 173})
    data = torch.rand(24, 3, 8, 8)
    real = RunningMoments()
    state = OfficialFDStatistics(6, 24, "ema", 0.9)
    with torch.no_grad():
        real.update(extractor(data))
        for batch in data.split(8):
            state.accumulate_initial(extractor(model(batch)))
    state.finalize_initialization()
    space = StaticSpace(extractor, real.moments(), state, root_function=precompute_sigma_ref_sqrt)
    objective = OfficialStaticFD({"tiny": space})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, objective, optimizer, data
