import os

import cv2
import numpy as np
import torch

from iou import compute_area, compute_iou, compute_bboxes_iou
from polygons_intersection import intersect_convex_polygons
from utils import box2corners, box2corners_np
from vis_utils import draw_polygons, add_grid_on_image, add_mult_titles_to_image, \
    create_images_grid


def convert_to_boxes(data):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # everything works with cuda
    device = torch.device('cpu')  # no need to init gpu for this
    return torch.tensor(data, dtype=torch.float32, device=device)[None, :]


def save_image(image, path_idx):
    save_dir = os.path.join('.', 'results')
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f'demo{path_idx}.png')
    cv2.imwrite(image_path, image)
    print(f'Saved final image to {image_path}')


def prepare_boxes_image(boxes1, boxes2, grid_size):
    points1 = box2corners(boxes1)[0]
    points2 = box2corners(boxes2)[0]

    points1.requires_grad = True

    intersection_points = intersect_convex_polygons(points1, points2)
    if intersection_points is not None:
        area = compute_area(intersection_points)
        iou = compute_iou(points1, points2)
        print(f'Area grad: {area.requires_grad}, IOU grad: {iou.requires_grad}')
        print(f'Area device: {area.device}, IOU device: {iou.device}')
        np_intersection_corners = intersection_points.detach().cpu().numpy()[None, ...]
        np_inter_area = int(compute_area(intersection_points).item())
        np_iou = iou.item()
    else:
        print('Empty intersection')
        np_intersection_corners = None
        np_inter_area = 0.0
        np_iou = 0.0

    np_std_iou = compute_bboxes_iou(points1, points2, points_mode=True).item()
    np_area1 = int(compute_area(points1).item())
    np_area2 = int(compute_area(points2).item())
    np_boxes1 = boxes1.detach().cpu().numpy()
    np_boxes2 = boxes2.detach().cpu().numpy()
    np_corners1 = box2corners_np(np_boxes1)
    np_corners2 = box2corners_np(np_boxes2)
    intersection_vis_image = draw_polygons(image_size_yx, np_corners1, np_corners2, np_intersection_corners)

    title1 = f'Rtd_IOU={np_iou:.3f}, bbox_IOU = {np_std_iou:.3f},'
    title2 = f'inter_area = {(np_inter_area / 1000):.1f}k, area1 = {(np_area1 / 1000):.1f}k, area2 = {(np_area2 / 1000):.1f}k'
    titles = [title1, title2]
    titles = [title1]
    init_title_h_ratio = 0.1

    final_image = intersection_vis_image
    final_image = add_grid_on_image(final_image, (grid_size, grid_size))
    final_image = add_mult_titles_to_image(final_image, titles, init_title_h_ratio)
    return final_image


def generate_boxes_data(idx):
    # Assume 1000x1000 image
    if idx == 1:
        cx_a, cy_a, h_a, w_a, t_a = 500, 500, 700, 200, +np.pi / 3
        cx_b, cy_b, h_b, w_b, t_b = 500, 500, 800, 400, -np.pi / 6
    elif idx == 2:
        cx_a, cy_a, h_a, w_a, t_a = 400, 400, 500, 400, -0.45 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 600, 600, 500, 400, +0.4 * np.pi
    elif idx == 3:
        cx_a, cy_a, h_a, w_a, t_a = 650, 500, 500, 400, -0.45 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 350, 500, 500, 400, -0.35 * np.pi
    elif idx == 4:
        cx_a, cy_a, h_a, w_a, t_a = 750, 500, 500, 400, -0.45 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 350, 500, 500, 400, -0.35 * np.pi
    elif idx == 5:
        cx_a, cy_a, h_a, w_a, t_a = 500, 500, 500, 400, -0.45 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 500, 500, 500, 400, +0.4 * np.pi
    elif idx == 6:
        cx_a, cy_a, h_a, w_a, t_a = 500, 500, 500, 400, 0.5 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 500, 500, 500, 400, 0 * np.pi
    elif idx == 7:
        cx_a, cy_a, h_a, w_a, t_a = 500, 500, 250, 200, 0.5 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 500, 500, 500, 400, 0 * np.pi
    elif idx == 8:
        cx_a, cy_a, h_a, w_a, t_a = 300, 400, 400, 300, -0.45 * np.pi
        cx_b, cy_b, h_b, w_b, t_b = 700, 600, 400, 300, +0.4 * np.pi
    else:
        assert False, f'Data for idx={idx} is not generated'
    data_a = [cx_a, cy_a, h_a, w_a, t_a]
    data_b = [cx_b, cy_b, h_b, w_b, t_b]
    boxes_a = convert_to_boxes(data_a)
    boxes_b = convert_to_boxes(data_b)
    return boxes_a, boxes_b


if __name__ == '__main__':
    image_size_yx = (1000, 1000)
    grid_size = 20

    boxes_1a, boxes_1b = generate_boxes_data(1)
    test_image1 = prepare_boxes_image(boxes_1a, boxes_1b, grid_size=grid_size)

    boxes_2a, boxes_2b = generate_boxes_data(2)
    test_image2 = prepare_boxes_image(boxes_2a, boxes_2b, grid_size=grid_size)

    boxes_3a, boxes_3b = generate_boxes_data(3)
    test_image3 = prepare_boxes_image(boxes_3a, boxes_3b, grid_size=grid_size)

    boxes_4a, boxes_4b = generate_boxes_data(4)
    test_image4 = prepare_boxes_image(boxes_4a, boxes_4b, grid_size=grid_size)

    boxes_5a, boxes_5b = generate_boxes_data(5)
    test_image5 = prepare_boxes_image(boxes_5a, boxes_5b, grid_size=grid_size)

    boxes_6a, boxes_6b = generate_boxes_data(6)
    test_image6 = prepare_boxes_image(boxes_6a, boxes_6b, grid_size=grid_size)

    boxes_7a, boxes_7b = generate_boxes_data(7)
    test_image7 = prepare_boxes_image(boxes_7a, boxes_7b, grid_size=grid_size)

    boxes_8a, boxes_8b = generate_boxes_data(8)
    test_image8 = prepare_boxes_image(boxes_8a, boxes_8b, grid_size=grid_size)

    all_test_images = [
        test_image1, test_image2, test_image3, test_image4,
        test_image5, test_image6, test_image7, test_image8
    ]
    n_cols = 4
    n_rows = 2
    padding = 3
    images_grid = create_images_grid(all_test_images, n_cols=n_cols, n_rows=n_rows, padding=padding)
    save_image(images_grid, path_idx=1)
