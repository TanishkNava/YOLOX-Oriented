import pytest
import torch

from yolox.models import (
    ECABlock,
    ImprovedECA,
    LayerAttention,
    MultiScaleDilatedConv,
    OrientedYOLOPAFPN,
    SpatialAttention,
    TaskDecomposition,
    YOLOX,
    YOLOXHeadOriented,
)
from yolox.models.network_blocks import BaseConv


def test_dilated_base_conv_preserves_spatial_size():
    conv = BaseConv(8, 8, 3, stride=1, dilation=3)
    assert conv(torch.randn(1, 8, 16, 16)).shape == (1, 8, 16, 16)


def test_multiscale_dilated_conv_keeps_shape_and_uses_four_branches():
    module = MultiScaleDilatedConv(64)
    x = torch.randn(2, 64, 20, 20)
    assert module(x).shape == x.shape
    # three dilated branches plus the self-branch
    assert len(module.branches) == 3
    assert [b.conv.dilation[0] for b in module.branches] == [1, 2, 3]
    assert module.reduce.conv.out_channels == 32


def test_multiscale_dilated_conv_parameter_count_matches_paper():
    # The paper reports +0.16M parameters for YOLOX-S, whose stride-8
    # feature carries 128 channels.
    params = sum(p.numel() for p in MultiScaleDilatedConv(128).parameters())
    assert 0.15e6 < params < 0.17e6


def test_eca_block_scales_channels_without_reduction():
    module = ECABlock(64)
    x = torch.randn(2, 64, 8, 8)
    out = module(x)
    assert out.shape == x.shape
    # a channel gate is constant across spatial positions
    ratio = out[0, :, 0, 0] / x[0, :, 0, 0]
    assert torch.allclose(ratio, out[0, :, 3, 5] / x[0, :, 3, 5], atol=1e-5)


def test_spatial_attention_is_shared_across_channels():
    module = SpatialAttention()
    x = torch.randn(2, 16, 12, 12)
    out = module(x)
    assert out.shape == x.shape
    ratio = out[0, :, 4, 7] / x[0, :, 4, 7]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)


def test_improved_eca_applies_channel_then_spatial_attention():
    module = ImprovedECA(32)
    x = torch.randn(2, 32, 10, 10)
    assert module(x).shape == x.shape
    expected = module.spatial_attention(module.channel_attention(x))
    assert torch.allclose(module(x), expected, atol=1e-6)


def test_improved_eca_is_nearly_parameter_free():
    # The paper reports no measurable parameter increase for this block.
    assert sum(p.numel() for p in ImprovedECA(128).parameters()) < 200


def test_layer_attention_gates_channels_in_unit_range():
    module = LayerAttention(32)
    x = torch.ones(1, 32, 6, 6)
    out = module(x)
    assert out.shape == x.shape
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_task_decomposition_projects_stacked_features():
    module = TaskDecomposition(channels=16, stacked_convs=2)
    stacked = torch.randn(2, 32, 9, 9)
    assert module(stacked).shape == (2, 16, 9, 9)


@pytest.mark.parametrize(
    "dilated,eca", [(True, False), (False, True), (True, True)]
)
def test_oriented_pafpn_outputs_match_baseline_shapes(dilated, eca):
    neck = OrientedYOLOPAFPN(
        0.33, 0.5, multiscale_dilated=dilated, improved_eca=eca
    )
    outputs = neck(torch.randn(1, 3, 128, 128))
    assert [tuple(o.shape) for o in outputs] == [
        (1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)
    ]


def test_task_decomposition_head_drops_replaced_conv_stacks():
    head = YOLOXHeadOriented(4, 0.5, task_decomposition=True)
    names = [name for name, _ in head.named_parameters()]
    assert not any(name.startswith("cls_convs") for name in names)
    assert not any(name.startswith("reg_convs") for name in names)
    assert any(name.startswith("inter_convs") for name in names)
    assert any(name.startswith("cls_decomposition") for name in names)
    assert any(name.startswith("reg_decomposition") for name in names)


def test_paper_model_trains_and_infers_with_oriented_targets():
    neck = OrientedYOLOPAFPN(
        0.33, 0.5, multiscale_dilated=True, improved_eca=True
    )
    head = YOLOXHeadOriented(4, 0.5, task_decomposition=True)
    model = YOLOX(neck, head)

    images = torch.randn(2, 3, 128, 128)
    # one labelled object per image: [class, cx, cy, w, h, angle]
    targets = torch.zeros(2, 5, 6)
    targets[:, 0] = torch.tensor([1.0, 60.0, 60.0, 30.0, 12.0, 40.0])

    model.train()
    losses = model(images, targets)
    total_loss = losses["total_loss"]
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )

    model.eval()
    with torch.no_grad():
        predictions = model(images)
    # [cx, cy, w, h, angle, obj, cls...] over the three strides of a 128px image
    assert predictions.shape == (2, 16 * 16 + 8 * 8 + 4 * 4, 4 + 1 + 1 + 4)
    angles = predictions[..., 4]
    assert torch.all((angles >= 0.0) & (angles <= 180.0))
