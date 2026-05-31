#!/usr/bin/env python3
"""Loss terms for behavior-cloning the BEV policy.

Total BC loss:
  L = w_action * MSE(a_pred, a_expert)
    + w_smooth * smoothness(a_pred)
    + w_collision * collision_penalty(a_pred, bev)
    + w_uncert * uncertainty_reg(unc, residual)

Everything is written in terms of plain tensors so the same functions can be
reused inside the PPO auxiliary losses.
"""
import torch
import torch.nn.functional as F


def action_mse(pred, target):
    """Standard supervised action regression."""
    return F.mse_loss(pred, target)


def smoothness_loss(pred, prev=None):
    """Penalize large step-to-step command changes.

    If ``prev`` (previous timestep action) is given, penalize the temporal
    difference; otherwise penalize raw magnitude as a weak prior toward gentle
    commands.
    """
    if prev is None:
        return pred.pow(2).mean()
    return (pred - prev).pow(2).mean()


def collision_penalty(pred, bev, occ_channel=0, forward_only=True):
    """Discourage forward speed when the BEV shows occupancy ahead.

    ``bev`` is (B, C, H, W). We take the occupancy channel, look at the cells
    in front of the robot (top half rows by convention: row 0 is farthest
    ahead) and weight the requested forward speed (vx) by how occupied that
    region is. This is a soft prior, not a hard constraint -- the safety
    supervisor remains the real guarantee at runtime.
    """
    occ = bev[:, occ_channel]                       # (B, H, W)
    h = occ.shape[1]
    region = occ[:, : h // 2, :] if forward_only else occ
    occ_frac = region.reshape(region.shape[0], -1).mean(dim=1)  # (B,)
    vx = pred[:, 0].clamp(min=0.0)                  # only forward motion is risky
    return (occ_frac * vx).mean()


def uncertainty_regularization(uncertainty, residual):
    """Tie predicted uncertainty to the actual action error (heteroscedastic).

    Uses a Gaussian NLL form: small uncertainty must be justified by small
    error. ``residual`` is |a_pred - a_expert| reduced over the action dim.
    """
    var = uncertainty.squeeze(-1).pow(2) + 1e-4
    err = residual.detach()
    return (0.5 * (err.pow(2) / var + torch.log(var))).mean()


def bc_total_loss(pred_full, target, bev, weights, prev=None):
    """Combine all BC terms.

    pred_full : (B, 4)  -> [vx, vy, wz, uncertainty]
    target    : (B, 3)  expert [vx, vy, wz]
    weights   : dict with action/smooth/collision/uncertainty keys
    Returns (total, breakdown_dict).
    """
    pred_act = pred_full[:, :3]
    unc = pred_full[:, 3:4]
    l_action = action_mse(pred_act, target)
    l_smooth = smoothness_loss(pred_act, prev)
    l_coll = collision_penalty(pred_act, bev)
    residual = (pred_act - target).abs().mean(dim=1)
    l_unc = uncertainty_regularization(unc, residual)
    total = (weights['action'] * l_action
             + weights['smooth'] * l_smooth
             + weights['collision'] * l_coll
             + weights['uncertainty'] * l_unc)
    return total, {
        'action': float(l_action.detach()),
        'smooth': float(l_smooth.detach()),
        'collision': float(l_coll.detach()),
        'uncertainty': float(l_unc.detach()),
        'total': float(total.detach()),
    }
