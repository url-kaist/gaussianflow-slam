import torch
from gaussian_splatting.utils.loss_utils import ssim
from typing import List, Dict, Tuple, Any

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    ssim_rgb = 1 - ssim(image, gt_image)

    return l1_rgb.mean() * 0.8 + ssim_rgb * 0.2
    # return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def update_acm(idx: int, acm_list: List[Dict], values: Tuple[Any, ...]) -> None:
    for container, val in zip(acm_list, values):
        container[idx] = val

def _promote_dtypes(dtypes):
    it = iter(dtypes)
    dt = next(it)
    for d in it:
        dt = torch.promote_types(dt, d)
    return dt

@torch.no_grad()
def vectorize_and_stack_values(acm_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    vals = list(acm_dict.values())
    if not vals:
        raise ValueError("acm_dict is empty")

    dtypes = [v.detach().squeeze().dtype for v in vals]
    target_dtype = _promote_dtypes(dtypes)

    t0 = vals[0].detach().squeeze().to(dtype=target_dtype)
    K = len(vals)
    out_shape = list(t0.shape); out_shape.insert(1, K)
    out = t0.new_empty(out_shape)
    out.select(1, 0).copy_(t0)

    for k, v in enumerate(vals[1:], start=1):
        tv = v.detach().squeeze()
        if tv.device != out.device:
            tv = tv.to(out.device)
        tv = tv.to(dtype=target_dtype)
        out.select(1, k).copy_(tv)
    return out


@torch.no_grad()
def vectorize_and_stack_grads(acm_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    vals = list(acm_dict.values())
    if not vals:
        raise ValueError("acm_dict is empty")

    def base_of(t: torch.Tensor) -> torch.Tensor:
        return (t.grad.detach() if t.grad is not None else t.detach()).squeeze()

    dtypes = [base_of(v).dtype for v in vals]
    target_dtype = _promote_dtypes(dtypes)

    b0 = base_of(vals[0]).to(dtype=target_dtype)
    K = len(vals)
    out_shape = list(b0.shape); out_shape.insert(1, K)
    out = b0.new_empty(out_shape)

    # col 0
    if vals[0].grad is None:
        out.select(1, 0).zero_()
    else:
        t = vals[0].grad.detach().squeeze().to(dtype=target_dtype)
        if t.device != out.device:
            t = t.to(out.device)
        out.select(1, 0).copy_(t)

    # col 1..K-1
    for k, v in enumerate(vals[1:], start=1):
        if v.grad is None:
            out.select(1, k).zero_()
        else:
            t = v.grad.detach().squeeze().to(dtype=target_dtype)
            if t.device != out.device:
                t = t.to(out.device)
            out.select(1, k).copy_(t)
    return out

# def vectorize_and_stack_values(acm_dict:Dict):
#     return torch.stack([v.detach().squeeze() for v in acm_dict.values()], dim=1)
#     # it = iter(acm_dict.values())
#     # first = next(it)
#     # dev, dt = first.device, first.dtype
#     # cols = [first.detach().squeeze().to(dev, dtype=dt)]
#     # for v in it:
#     #     cols.append(v.detach().squeeze().to(dev, dtype=dt))
#     # return torch.stack(cols, dim=1)

# def vectorize_and_stack_grads(acm_dict:Dict):
#     # return torch.stack([v.grad.detach().squeeze() for v in acm_dict.values()], dim=1)
#     return torch.stack(
#         [
#             (v.grad.detach() if v.grad is not None else torch.zeros_like(v))
#             .squeeze()
#             for v in acm_dict.values()
#         ],
#         dim=1,
#     )
#     # it = iter(acm_dict.values())
#     # first = next(it)
#     # base = (first.grad if first.grad is not None else torch.zeros_like(first))
#     # dev, dt = base.device, base.dtype
#     # cols = [(base.detach()).squeeze()]
#     # for v in it:
#     #     g = v.grad if v.grad is not None else torch.zeros_like(v)
#     #     cols.append(g.detach().squeeze().to(dev, dtype=dt))
#     # return torch.stack(cols, dim=1)

# def normalize_stacked_tensor(stacked_tensor: torch.Tensor, weight_tensor: torch.Tensor, total_found_filter):
#     assert stacked_tensor.shape[:2] == weight_tensor.shape[:2]
#     # normalized_tensor = torch.zeros_like(stacked_tensor)
#     # normalized_tensor[total_found_filter] = stacked_tensor[total_found_filter] / (weight_tensor[total_found_filter] + 1e-6)
#     # torch.nan_to_num(normalized_tensor, nan=0.0)
#     # return normalized_tensor

#     mask = total_found_filter
#     while mask.dim() < stacked_tensor.dim():
#         mask = mask.unsqueeze(-1)
#     mask_f = mask.to(dtype=stacked_tensor.dtype)
#     denom = weight_tensor + 1e-6
#     normalized = torch.divide(stacked_tensor, denom)
#     torch.nan_to_num_(normalized, nan=0.0, posinf=0.0, neginf=0.0)
#     return normalized

def normalize_stacked_tensor(
    stacked_tensor: torch.Tensor,       # e.g. [B, N, C] or [N, C] ...
    weight_tensor: torch.Tensor,        # e.g. [B, N] or [N]
    total_found_filter: torch.Tensor,   # e.g. [B, N] or [B, N, 1] ...
    eps: float = 1e-6,
):
    if not weight_tensor.is_floating_point():
        weight_tensor = weight_tensor.to(stacked_tensor.dtype)

    mask = total_found_filter.to(torch.bool)
    lead_ndims = weight_tensor.dim()

    while mask.dim() > lead_ndims and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    while mask.dim() < lead_ndims:
        mask = mask.unsqueeze(-1)

    for ms, ws in zip(mask.shape, weight_tensor.shape):
        if ms != ws and ms != 1:
            raise ValueError(f"Incompatible mask {mask.shape} vs weight {weight_tensor.shape}")

    expand_dims = stacked_tensor.dim() - lead_ndims
    mask_exp = mask[(...,) + (None,) * expand_dims]

    stacked_tensor.masked_fill_(~mask_exp, 0)

    den = (weight_tensor + eps)
    den_exp = den[(...,) + (None,) * expand_dims]     # e.g. [B,N] → [B,N,1]
    stacked_tensor.div_(den_exp)                      # in-place

    return stacked_tensor

def squared_normalize_stacked_tensor(stacked_tensor: torch.Tensor, weight_tensor: torch.Tensor, total_found_filter):
    assert stacked_tensor.shape[:2] == weight_tensor.shape[:2]
    normalized_tensor = torch.zeros_like(stacked_tensor)
    normalized_tensor[total_found_filter] = stacked_tensor[total_found_filter] \
        / (weight_tensor[total_found_filter]**2 + 1e-6)
    torch.nan_to_num(normalized_tensor, nan=0.0)
    return normalized_tensor

def reduce_max(stacked_tensor:torch.Tensor):
    if stacked_tensor.ndim >= 3:
        stacked_tensor = stacked_tensor[..., :2].norm(dim=2, keepdim=True)
    max_tensor, _ = stacked_tensor.max(dim=1)
    return max_tensor.squeeze()

# def reduce_min(stacked_tensor: torch.Tensor):
#     if stacked_tensor.ndim >= 3:
#         stacked_tensor = stacked_tensor[..., :2].norm(dim=2, keepdim=True)

#     masked_tensor = torch.where(stacked_tensor > 0, stacked_tensor, torch.tensor(float('inf'), device=stacked_tensor.device))
#     min_tensor, _ = masked_tensor.min(dim=1)
#     min_tensor = torch.where(torch.isinf(min_tensor), torch.tensor(0.0, device=min_tensor.device), min_tensor)
#     return min_tensor.squeeze()
        
#     # masked = stacked_tensor.clone()
#     # masked.masked_fill_(masked <= 0, float("inf"))
#     # min_tensor, _ = masked.min(dim=1)
#     # # inf → 0
#     # inf_mask = torch.isinf(min_tensor)
#     # if inf_mask.any():
#     #     min_tensor[inf_mask] = 0.0
#     # return min_tensor.squeeze()

@torch.no_grad()
def reduce_min(x: torch.Tensor, min_chunk_cols: int = 32):
    """
    x: accepts (B, N) or (B, N, C>=2).
    Takes the minimum over strictly positive entries, falling back to 0.
    Uses adaptive chunking and avoids copies to stay robust against OOM.
    """
    if x.ndim >= 3:
        t = torch.linalg.norm(x[..., :2].to(torch.float32), dim=-1)
    else:
        t = x.to(torch.float32)

    B, N = t.shape
    inf = torch.tensor(torch.finfo(t.dtype).max, device=t.device, dtype=t.dtype)

    running = torch.full((B,), inf.item(), device=t.device, dtype=t.dtype)

    try:
        free, total = torch.cuda.mem_get_info(t.device.index if t.is_cuda else 0)
        budget = int(free * 0.60)
        bytes_per_elem = t.element_size()
        approx_per_col = B * (2 * bytes_per_elem + 1)
        start_chunk = max(min_chunk_cols, budget // max(approx_per_col, 1))
        chunk = int(max(min_chunk_cols, min(start_chunk, 1_000_000)))
    except Exception:
        chunk = 8192

    i = 0
    while i < N:
        c = min(chunk, N - i)
        sl = t.narrow(1, i, c)

        try:
            pos = sl > 0
            masked = torch.where(pos, sl, inf)
            part_min = masked.amin(dim=1)  # (B,)

            running = torch.minimum(running, part_min)
            i += c
            # chunk = min(chunk * 2, 1_000_000)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and chunk > min_chunk_cols:
                torch.cuda.empty_cache()
                chunk = max(min_chunk_cols, chunk // 2)
                continue
            raise

    running = torch.where(torch.isinf(running),
                          torch.zeros((), device=t.device, dtype=t.dtype),
                          running)
    return running

def reduce_mean(stacked_tensor:torch.Tensor, found_count, total_found_filter):
    if stacked_tensor.ndim >= 3:
        stacked_tensor = stacked_tensor[..., :2].norm(dim=2, keepdim=True).squeeze(2)
    # mean_tensor = stacked_tensor.sum(dim=1).to(torch.float32)
    # mean_tensor[total_found_filter] = mean_tensor[total_found_filter] / found_count[total_found_filter]
    # torch.nan_to_num(mean_tensor, nan=0.0)
    # return mean_tensor.squeeze()
    mean_tensor = stacked_tensor.sum(dim=1).to(torch.float32)   # [N, ...] → [N]
    mask = total_found_filter
    while mask.dim() < mean_tensor.dim():
        mask = mask.unsqueeze(-1)
    denom = torch.where(mask, found_count.clamp_min(1), torch.ones_like(found_count))
    mean_tensor = mean_tensor / denom.to(mean_tensor.dtype)
    torch.nan_to_num_(mean_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return mean_tensor.squeeze()

def reduce_median(stacked_tensor:torch.Tensor):
    if stacked_tensor.ndim >= 3:
        stacked_tensor = stacked_tensor[..., :2].norm(dim=2, keepdim=True)
    # mask = stacked_tensor > 0                        
    # nan_const = torch.tensor(float("nan"), device=stacked_tensor.device)
    # masked = torch.where(mask, stacked_tensor, nan_const)
    # median_tensor = torch.nanmedian(masked, dim=1).values
    # median_tensor = torch.nan_to_num(median_tensor, nan=0.0)
    # return median_tensor.squeeze()
        
    mask = stacked_tensor > 0
    masked = torch.where(mask, stacked_tensor, torch.tensor(float("nan"), device=stacked_tensor.device))
    median_tensor = torch.nanmedian(masked, dim=1).values
    median_tensor = torch.nan_to_num(median_tensor, nan=0.0)
    return median_tensor.squeeze()


def set_occ_aware_visibility(n_touched_acm:Dict):
    occ_aware_visibility = {}
    for e in n_touched_acm:
        n_touched = n_touched_acm[e]
        occ_aware_visibility[e] = (n_touched >0).long()
    
    return occ_aware_visibility


def solve_pose_update1(updating_indices, edges, Jii, Jjj, r, wJii, wJjj, wr, damping=None):
    N = len(updating_indices)
    H = torch.zeros((6*N, 6*N), device="cuda")
    b = torch.zeros((6*N, 1), device="cuda")

    for e, (i, j) in enumerate(edges):
        wJi = wJii[e].squeeze(0)   # (N, 6)
        wJj = wJjj[e].squeeze(0)
        wrij = wr[e].squeeze(0)    # (N,)

        vi = wJi.T @ wrij          # (6,)
        vj = wJj.T @ wrij

        Hii = wJi.T @ wJi
        Hij = wJi.T @ wJj
        Hjj = wJj.T @ wJj

        # symmetric coupling
        H[6*i:6*i+6, 6*i:6*i+6] += Hii
        H[6*i:6*i+6, 6*j:6*j+6] += Hij
        H[6*j:6*j+6, 6*i:6*i+6] += Hij.T
        H[6*j:6*j+6, 6*j:6*j+6] += Hjj

        b[6*i:6*i+6] += vi
        b[6*j:6*j+6] += vj

    # delta_x = torch.linalg.solve(H, b)
    delta_x, *_ = torch.linalg.lstsq(H + 1e-6 * torch.eye(6*N, device="cuda"), -b)
    return delta_x.view(N,6)


def solve_pose_update2(updating_indices, edges, Jii, Jjj, r, weights, damping=None):
    N = len(updating_indices)
    H = torch.zeros((6*N, 6*N), device="cuda")
    b = torch.zeros((6*N, 1), device="cuda")

    for e, (i,j) in enumerate(edges):
        Ji = Jii[e].squeeze(0)
        Jj = Jjj[e].squeeze(0)
        w = weights[e].squeeze(0)
        
        rij = r[e].squeeze(0)

        vi = (w * Ji).transpose(0,1) @ rij
        vj = (w * Jj).transpose(0,1) @ rij

        Hii = (w * Ji).transpose(0,1) @ Ji
        Hij = (w * Ji).transpose(0,1) @ Jj
        Hji = (w * Jj).transpose(0,1) @ Ji
        Hjj = (w * Jj).transpose(0,1) @ Jj

        Hij_sym = (Hij + Hji)/2.0

        H[6*i:6*i+6, 6*i:6*i+6] += Hii
        H[6*i:6*i+6, 6*j:6*j+6] += Hij_sym
        H[6*j:6*j+6, 6*i:6*i+6] += Hij_sym
        H[6*j:6*j+6, 6*j:6*j+6] += Hjj

        b[6*i:6*i+6] += vi
        b[6*j:6*j+6] += vj

    # delta_x = torch.linalg.solve(H, -b)
    delta_x, *_ = torch.linalg.lstsq(H + 1e-6 * torch.eye(6*N, device="cuda"), -b)
    return delta_x.view(N,6)


def check_tensor_validity(tensor, name):
    if torch.isnan(tensor).any():
        print(f"[WARNING] {name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"[WARNING] {name} contains Inf values")
    if not torch.isfinite(tensor).all():
        print(f"[WARNING] {name} contains non-finite values (NaN or Inf)")