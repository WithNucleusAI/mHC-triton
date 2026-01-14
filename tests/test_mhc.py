"""Tests for mHC kernels and module."""

import pytest
import torch

# Skip if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required"
)


def _sinkhorn_ref(M, iters=20, eps=1e-8):
    """Reference Sinkhorn-Knopp in pure PyTorch."""
    M = M.abs() + eps
    for _ in range(iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class TestSinkhornKnopp:
    """Tests for Sinkhorn-Knopp projection."""

    def test_forward_correctness(self):
        from mhc import sinkhorn_knopp

        torch.manual_seed(42)
        batch = 4
        M = torch.randn(batch, 4, 4, device='cuda', dtype=torch.float32)

        M_ref = _sinkhorn_ref(M.clone())
        M_tri = sinkhorn_knopp(M.clone())

        assert (M_ref - M_tri).abs().max() < 1e-5

    def test_doubly_stochastic(self):
        from mhc import sinkhorn_knopp

        torch.manual_seed(42)
        M = torch.randn(8, 4, 4, device='cuda')
        P = sinkhorn_knopp(M)

        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)

    def test_backward(self):
        from mhc import sinkhorn_knopp

        torch.manual_seed(42)
        M = torch.randn(4, 4, 4, device='cuda', requires_grad=True)
        M_clone = M.clone().detach().requires_grad_(True)

        # Reference
        P_ref = _sinkhorn_ref(M)
        P_ref.sum().backward()

        # Triton
        P_tri = sinkhorn_knopp(M_clone)
        P_tri.sum().backward()

        assert (M.grad - M_clone.grad).abs().max() < 1e-4


class TestStreamMix:
    """Tests for fused stream mixing."""

    def test_forward_correctness(self):
        from mhc import fused_stream_mix

        torch.manual_seed(42)
        batch, seq, dim = 2, 64, 256

        H = torch.randn(batch, seq, 4, dim, device='cuda')
        H_pre = torch.softmax(torch.randn(batch, 4, device='cuda'), dim=-1)
        H_res = _sinkhorn_ref(torch.randn(batch, 4, 4, device='cuda'))

        # Reference
        branch_ref = torch.einsum('bn,bsnd->bsd', H_pre, H)
        res_ref = torch.einsum('bnm,bsmd->bsnd', H_res, H)

        # Triton
        branch_tri, res_tri = fused_stream_mix(H, H_pre, H_res)

        assert (branch_ref - branch_tri).abs().max() < 1e-4
        assert (res_ref - res_tri).abs().max() < 1e-4

    def test_backward(self):
        from mhc import fused_stream_mix

        torch.manual_seed(42)
        batch, seq, dim = 2, 32, 128

        H = torch.randn(batch, seq, 4, dim, device='cuda', requires_grad=True)
        H_pre = torch.softmax(torch.randn(batch, 4, device='cuda'), dim=-1).requires_grad_(True)
        H_res = _sinkhorn_ref(torch.randn(batch, 4, 4, device='cuda')).detach().requires_grad_(True)

        H_c = H.clone().detach().requires_grad_(True)
        H_pre_c = H_pre.clone().detach().requires_grad_(True)
        H_res_c = H_res.clone().detach().requires_grad_(True)

        # Reference
        branch_ref = torch.einsum('bn,bsnd->bsd', H_pre, H)
        res_ref = torch.einsum('bnm,bsmd->bsnd', H_res, H)
        (branch_ref.sum() + res_ref.sum()).backward()

        # Triton
        branch_tri, res_tri = fused_stream_mix(H_c, H_pre_c, H_res_c)
        (branch_tri.sum() + res_tri.sum()).backward()

        assert (H.grad - H_c.grad).abs().max() < 1e-3


class TestAddResidual:
    """Tests for fused add residual."""

    def test_forward_correctness(self):
        from mhc import fused_add_residual

        torch.manual_seed(42)
        batch, seq, dim = 2, 64, 256

        H_residual = torch.randn(batch, seq, 4, dim, device='cuda')
        branch_out = torch.randn(batch, seq, dim, device='cuda')
        H_post = torch.sigmoid(torch.randn(batch, 4, device='cuda'))

        # Reference
        H_new_ref = H_residual + branch_out.unsqueeze(2) * H_post.unsqueeze(1).unsqueeze(-1)

        # Triton
        H_new_tri = fused_add_residual(H_residual, branch_out, H_post)

        assert (H_new_ref - H_new_tri).abs().max() < 1e-4

    def test_backward(self):
        from mhc import fused_add_residual

        torch.manual_seed(42)
        batch, seq, dim = 2, 32, 128

        H_res = torch.randn(batch, seq, 4, dim, device='cuda', requires_grad=True)
        branch = torch.randn(batch, seq, dim, device='cuda', requires_grad=True)
        H_post = torch.sigmoid(torch.randn(batch, 4, device='cuda')).requires_grad_(True)

        H_res_c = H_res.clone().detach().requires_grad_(True)
        branch_c = branch.clone().detach().requires_grad_(True)
        H_post_c = H_post.clone().detach().requires_grad_(True)

        # Reference
        out_ref = H_res + branch.unsqueeze(2) * H_post.unsqueeze(1).unsqueeze(-1)
        out_ref.sum().backward()

        # Triton
        out_tri = fused_add_residual(H_res_c, branch_c, H_post_c)
        out_tri.sum().backward()

        assert (H_res.grad - H_res_c.grad).abs().max() < 1e-3


class TestHyperConnection:
    """Tests for HyperConnection module."""

    def test_forward_shape(self):
        from mhc import HyperConnection

        hc = HyperConnection(dim=256, dynamic=True).cuda()

        H = torch.randn(2, 64, 4, 256, device='cuda')
        branch_input, add_residual = hc(H)

        assert branch_input.shape == (2, 64, 256)

        branch_output = torch.randn(2, 64, 256, device='cuda')
        H_new = add_residual(branch_output)

        assert H_new.shape == (2, 64, 4, 256)

    def test_training(self):
        from mhc import HyperConnection

        torch.manual_seed(42)
        hc = HyperConnection(dim=128, dynamic=True).cuda()
        layer = torch.nn.Linear(128, 128, device='cuda')
        optimizer = torch.optim.Adam(list(hc.parameters()) + list(layer.parameters()))

        # Save initial params
        init_params = {n: p.clone() for n, p in hc.named_parameters()}

        # Training steps
        for _ in range(3):
            optimizer.zero_grad()
            H = torch.randn(2, 32, 4, 128, device='cuda', requires_grad=True)
            branch_input, add_res = hc(H)
            H_new = add_res(layer(branch_input))
            H_new.sum().backward()
            optimizer.step()

        # Check params changed
        changed = sum(
            1 for n, p in hc.named_parameters()
            if not torch.allclose(p, init_params[n])
        )
        assert changed > 0, "Parameters should update during training"

    def test_static_mode(self):
        from mhc import HyperConnection

        hc = HyperConnection(dim=128, dynamic=False).cuda()
        H = torch.randn(2, 32, 4, 128, device='cuda')

        branch_input, add_residual = hc(H)
        assert branch_input.shape == (2, 32, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

