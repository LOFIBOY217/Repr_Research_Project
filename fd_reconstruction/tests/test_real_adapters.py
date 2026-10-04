import pytest
import torch
from recon_fd.tokenizers import KLReconstructor


def test_diffusers_real_encoder_decoder_contract(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    # Real architecture and checkpoint round trip, small RANDOM model: no weights download.
    model = diffusers.AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=16)
    model.save_pretrained(tmp_path)
    adapter = KLReconstructor(str(tmp_path), local_files_only=True)
    image = torch.rand(2, 3, 16, 16)
    reconstruction = adapter(image)
    assert reconstruction.shape == image.shape
    assert reconstruction.min() >= 0 and reconstruction.max() <= 1
    reconstruction.square().mean().backward()
    for group in adapter.parameter_groups().values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group)


def test_inception_backbone_shape_and_input_grad():
    from recon_fd.representations.inception import InceptionV3
    model = InceptionV3(normalize=False).eval().requires_grad_(False)
    image = torch.rand(1, 3, 32, 32, requires_grad=True)
    features, _ = model(image)
    assert features.shape == (1, 2048)
    features.sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_timm_local_weights_and_pooling(tmp_path):
    timm = pytest.importorskip("timm")
    from recon_fd.representations import build_representation
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    path = tmp_path / "vit.pt"
    torch.save(model.state_dict(), path)
    for pool in ("cls", "avg"):
        representation = build_representation({"kind": "timm", "name": "vit", "weights": str(path),
            "model_name": "vit_tiny_patch16_224", "target_size": 32, "pool": pool})
        image = torch.rand(2, 3, 16, 16, requires_grad=True)
        features = representation(image)
        assert features.shape == (2, 192)
        features.square().sum().backward()
        assert image.grad is not None and image.grad.abs().sum() > 0
        assert all(not parameter.requires_grad for parameter in representation.parameters())
