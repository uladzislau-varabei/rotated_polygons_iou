import numpy as np
import torch


def box2corners_np(boxes):
    # Thanks to: https://github.com/lilanxiao/Rotated_IoU/blob/debug/oriented_iou_loss.py
    theta = boxes[:, 4:5]
    cx = boxes[:, 0:1]
    cy = boxes[:, 1:2]
    w = boxes[:, 2:3]
    h = boxes[:, 3:4]
    # Apply rotation
    x_deltas = np.array([0.5, -0.5, -0.5, 0.5])
    y_deltas = np.array([0.5, 0.5, -0.5, -0.5])
    x_scaled = x_deltas * w
    y_scaled = y_deltas * h
    coords_scaled = np.stack([x_scaled, y_scaled], axis=-1)
    cos = np.cos(theta)
    sin = np.sin(theta)
    M = np.stack([
        np.concatenate([cos, sin], axis=-1),
        np.concatenate([-sin, cos], axis=-1)
    ], axis=-2)
    corners = coords_scaled @ M
    # Apply translation
    corners[..., 0] += cx
    corners[..., 1] += cy
    return corners


def box2corners(boxes):
    # Thanks to: https://github.com/lilanxiao/Rotated_IoU/blob/debug/oriented_iou_loss.py
    theta = boxes[:, 4:5]
    cx = boxes[:, 0:1]
    cy = boxes[:, 1:2]
    w = boxes[:, 2:3]
    h = boxes[:, 3:4]
    # Apply rotation
    x_deltas = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=boxes.dtype, device=boxes.device)
    y_deltas = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=boxes.dtype, device=boxes.device)
    x_scaled = x_deltas * w
    y_scaled = y_deltas * h
    coords_scaled = torch.stack([x_scaled, y_scaled], dim=-1)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    M = torch.stack([
        torch.cat([cos, sin], dim=-1),
        torch.cat([-sin, cos], dim=-1)
    ], dim=-2)
    corners = coords_scaled @ M
    # Apply translation
    corners[..., 0] += cx
    corners[..., 1] += cy
    return corners
