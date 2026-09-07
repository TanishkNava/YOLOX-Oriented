import torch

from yolox.models.losses import GaussianKLDLoss, gaussian_kld_similarity


def test_kld_is_zero_for_identical_boxes_and_one_similarity():
    boxes = torch.tensor([[10.0, 12.0, 8.0, 3.0, 25.0]])
    loss = GaussianKLDLoss()(boxes, boxes)
    similarity = gaussian_kld_similarity(boxes, boxes)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    torch.testing.assert_close(similarity, torch.ones_like(similarity), atol=1e-6, rtol=0)


def test_kld_equivalent_parameterizations_and_gradients_are_finite():
    pred = torch.tensor(
        [[10.0, 12.0, 8.0, 3.0, 25.0]], requires_grad=True
    )
    equivalent = torch.tensor([[10.0, 12.0, 3.0, 8.0, 115.0]])
    loss = GaussianKLDLoss(reduction="mean")(pred, equivalent)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_kld_large_offsets_remain_bounded_and_finite():
    pred = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0]])
    target = torch.tensor([[1e6, -1e6, 2.0, 1.0, 90.0]])
    loss = GaussianKLDLoss()(pred, target)
    assert torch.isfinite(loss).all()
    assert torch.all((loss >= 0) & (loss <= 1))


def test_kld_promotes_half_precision_covariance_math():
    pred = torch.tensor(
        [[10.0, 12.0, 64.0, 1e-5, 25.0]], dtype=torch.float16,
        requires_grad=True,
    )
    target = torch.tensor(
        [[10.0, 12.0, 8.0, 3.0, 25.0]], dtype=torch.float16
    )
    loss = GaussianKLDLoss(reduction="mean")(pred, target)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
