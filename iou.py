import torch

from polygons_intersection import intersect_convex_polygons


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
