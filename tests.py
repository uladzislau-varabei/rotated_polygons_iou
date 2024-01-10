import os

import cv2
import numpy as np
import torch

from iou import compute_area, compute_iou
from polygons_intersection import intersect_convex_polygons
from utils import box2corners, box2corners_np
from vis_utils import draw_polygons, add_grid_on_image, add_mult_titles_to_image


def create_boxes(data):
    return torch.tensor(data, dtype=torch.float32)[None, :]


def test_boxes(boxes1, boxes2):
    points1 = box2corners(boxes1)[0]
    points2 = box2corners(boxes2)[0]

    points1.requires_grad = True

    intersection_points = intersect_convex_polygons(points1, points2)
    if intersection_points is not None:
        area = compute_area(intersection_points)
        iou = compute_iou(points1, points2)
        print(f'Area grad: {area.requires_grad}, IOU grad: {iou.requires_grad}')
    else:
        print('Empty intersection')

    np_intersection_corners = intersection_points.detach().cpu().numpy()[None, ...]
    np_area1 = int(compute_area(points1).item())
    np_area2 = int(compute_area(points2).item())
    np_inter_rea = int(compute_area(intersection_points).item())
    np_iou = iou.item()
    np_boxes1 = boxes1.detach().cpu().numpy()
    np_boxes2 = boxes2.detach().cpu().numpy()
    np_corners1 = box2corners_np(np_boxes1)
    np_corners2 = box2corners_np(np_boxes2)
    intersection_vis_image = draw_polygons(image_size_yx, np_corners1, np_corners2, np_intersection_corners)

    title1 = f'IOU={np_iou:.3f}, inter_area = {np_inter_rea}'
    title2 = f'area1 = {np_area1}, area2 = {np_area2}'
    titles = [title1, title2]
    init_title_h_ratio = 0.1

    final_image = intersection_vis_image
    final_image = add_grid_on_image(final_image, (30, 30))
    final_image = add_mult_titles_to_image(final_image, titles, init_title_h_ratio)

    save_dir = os.path.join('.', 'results')
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, 'demo1.png')
    cv2.imwrite(image_path, final_image)
    print(f'Saved final image to {image_path}')


if __name__ == '__main__':
    scale = 5
    image_size_yx = (scale * 200, scale * 200)
    image = 255 + np.zeros((image_size_yx) + (3,), dtype=np.uint8)

    cx1, cy1 = scale * 100, scale * 100
    cx2, cy2 = scale * 100, scale * 100
    data1 = [cx1, cy1, scale * 80, scale * 20, np.pi / 3]
    data2 = [cx2, cy2, scale * 80, scale * 40, -np.pi / 6]

    boxes1 = create_boxes(data1)
    boxes2 = create_boxes(data2)

    test_image = test_boxes(boxes1, boxes2)
