# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch.nn.functional as F

# Disable TF32 for matmul to ensure consistency between the fused and reference implementations
torch.backends.cuda.matmul.allow_tf32 = False


def mhc_projection_ref(x, phi):
    """
    Reference operator for mHC's projection building operation.

    x: (M, nC) where M = s * b
    phi: (2n + n^2, nC), which consists of the following matrices
        - phi_pre: (n, nC)
        - phi_post: (n, nC)
        - phi_res: (n^2, nC)
    n: number of Hyper Connection streams
    C: hidden dimension per stream
    """
    x_dtype = x.dtype
    x = x.to(torch.float32)
    phi = phi.to(torch.float32)

    Hs = x @ phi.T  # (M, 2n + n^2)

    x_fp32 = x.to(torch.float32)  # Use fp32 for better numerical stability in variance calculation
    ms = (x_fp32 * x_fp32).mean(dim=1)

    return Hs.to(x_dtype), ms


def mhc_scale_ref(H, alpha, beta, ms, n):
    """
    Reference operator for mHC's H matrices scaling operation

    :param: H: (M, 2n + n^2), the unprocessed H matrices where M = s * b
    :param: alpha: (3,), three scalar parameters
    :param: beta: (1, 2n + n^2), bias term
    :param: r: (M,), the denominator for RMSNorm
    :param: n: int, the width of Hyper-Connection

    :return Hs: (M, 2n + n^2), the processed H matrices
    """

    input_dtype = H.dtype
    H = H.to(torch.float32)
    alpha = alpha.to(torch.float32)
    beta = beta.to(torch.float32)
    eps = torch.finfo(torch.float32).eps
    rms = torch.sqrt(ms + eps)  # (M,)
    rms = rms.to(torch.float32)

    H_pre = H[:, :n]  # (M, n)
    H_post = H[:, n : 2 * n]  # (M, n)
    H_res = H[:, 2 * n :]  # (M, n^2)

    beta_pre = beta[0, :n]
    beta_post = beta[0, n : 2 * n]
    beta_res = beta[0, 2 * n : 2 * n + n * n]

    alpha_pre, alpha_post, alpha_res = alpha[0], alpha[1], alpha[2]

    H_pre = H_pre * alpha_pre
    H_post = H_post * alpha_post
    H_res = H_res * alpha_res

    H_pre = H_pre / rms[:, None]
    H_post = H_post / rms[:, None]
    H_res = H_res / rms[:, None]

    H_pre = H_pre + beta_pre
    H_post = H_post + beta_post
    H_res = H_res + beta_res

    H_pre = F.sigmoid(H_pre)
    H_post = 2 * F.sigmoid(H_post)

    return H_pre.to(input_dtype), H_post.to(input_dtype), H_res.to(input_dtype)


def mhc_sinkhorn_ref(H_res, n=4, iterations=20):
    """
    Reference operator for mHC's Sinkhorn-Knopp algorithm to convert a matrix into a doubly stochastic matrix.
    Calculated in log space for numerical stability.

    :param H_res: a tensor of shape (s, b, n, n)
    :return: a tensor of shape (s, b, n, n)
    """
    s, b = H_res.shape[:2]
    device = H_res.device
    dtype = H_res.dtype

    H_res_f = H_res.to(
        torch.float32
    ).clone()  # Use float32 for better numerical stability during Sinkhorn iterations

    log_mu = torch.zeros(s, b, n, device=device, dtype=torch.float32)
    log_nu = torch.zeros(s, b, n, device=device, dtype=torch.float32)

    f = torch.zeros(s, b, n, device=device, dtype=torch.float32)
    g = torch.zeros(s, b, n, device=device, dtype=torch.float32)

    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res_f + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res_f + f.unsqueeze(3), dim=2)

    log_P = f.unsqueeze(3) + H_res_f + g.unsqueeze(2)
    H_res_out = torch.exp(log_P).to(dtype)  # Convert back to original dtype

    return H_res_out


def mhc_aggregate_ref(x, H_pre, n):
    """
    Reference operator for applying mHC's aggregation transformation

    x: (s, b, C, n)
    H_pre: (s, b, n)
    """
    H_pre = H_pre.contiguous()

    s, b, C, n = x.shape
    H_pre = H_pre.view(s, b, n, 1)

    out = (x @ H_pre).view(s, b, C)

    return out


def mhc_expand_combine_ref(f, bias, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's expansion and combination transformation

    f: (s, b, C)
    bias: (C,) or None
    H_post: (s, b, n)
    x: (s, b, C, n)
    H_res: (s, b, n, n)
    """

    s, b, C, n = x.shape

    # My triton kernels use FMA and MMA instructions with fp32 accumulator for bf16 test cases
    # which has better numerical stability than this pytorch implementation
    # To match the kernel's accuracy we need to cast to fp32 here to match kernels' result
    input_dtype = f.dtype
    f = f.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None
    H_post = H_post.to(torch.float32)
    x = x.to(torch.float32)
    H_res = H_res.to(torch.float32)

    if bias is not None:
        f = f + bias[None, None, :]

    f = f.view(s, b, C, 1)
    H_post = H_post.view(s, b, 1, n)

    out = f @ H_post + x @ H_res  # (s, b, C, n)

    return out.to(input_dtype)


