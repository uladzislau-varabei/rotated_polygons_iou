import torch

from polygons_intersection import intersect_convex_polygons
from utils import points2box


def compute_area(points):
    # Thanks to https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    # by Darius Bacon
    # Explanation: https://en.wikipedia.org/wiki/Shoelace_formula
    rolled_points = torch.roll(points, 1, dims=0)
    area = 0.5 * (points[:, 0] * rolled_points[:, 1] - rolled_points[:, 0] * points[:, 1]).sum().abs()
    return area


def compute_iou(points1, points2):
    area1 = compute_area(points1)
    area2 = compute_area(points2)
    intersection_points = intersect_convex_polygons(points1, points2)
    intersection_area = compute_area(intersection_points)
    iou = intersection_area / (area1 + area2 - intersection_area + 1e-16)
    return iou


def compute_bboxes_iou(bboxes1, bboxes2, points_mode=True):
    if points_mode:
        bboxes1 = points2box(bboxes1)[None, ...]
        bboxes2 = points2box(bboxes2)[None, ...]
    else:
        bboxes1 = bboxes1[:, :4]
        bboxes2 = bboxes2[:, :4]

    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    area1 = torch.prod(bboxes1[:, 2:] - bboxes1[:, :2], 1)
    area2 = torch.prod(bboxes2[:, 2:] - bboxes2[:, :2], 1)
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, dim=1) * en  # * ((tl < br).all())
    iou = area_i / (area1 + area2 - area_i + 1e-16)
    return iou[0] # previously added dim
