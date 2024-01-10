import cv2
import numpy as np


def draw_polygons(image_size_yx, corners1, corners2, intersection_corners=None):
    image = 255 + np.zeros(image_size_yx + (3,), dtype=np.uint8)

    vis_image = image.copy()
    corners1_image = 255 + np.zeros(vis_image.shape, dtype=np.uint8)
    for corner in corners1:
        points = corner.astype(np.int32).reshape(1, -1, 2)
        corners1_image = cv2.fillPoly(corners1_image, points, (0, 0, 255))

    corners2_image = 255 + np.zeros(vis_image.shape, dtype=np.uint8)
    for corner in corners2:
        points = corner.astype(np.int32).reshape(1, -1, 2)
        corners2_image = cv2.fillPoly(corners2_image, points, (255, 0, 0))

    # corners1_image loses color twice
    w1, w2 = 0.9, 0.5
    vis_image = cv2.addWeighted(corners1_image, w1, vis_image, 1.0 - w1, 0.0)
    vis_image = cv2.addWeighted(corners2_image, w2, vis_image, 1.0 - w2, 0.0)

    if intersection_corners is not None:
        for corner in intersection_corners:
            points = corner.astype(np.int32).reshape(1, -1, 2)
            thickness = max(image_size_yx) // 100
            vis_image = cv2.polylines(vis_image, [points], True, (0, 0, 255), thickness)

    return vis_image


def add_title_to_image(image, title, title_h_ratio=0.2):
    # For centering thanks to: https://gist.github.com/evilmtv/af2a023e472e6303fd2d3cc02aa1a83a
    h, w, c = image.shape[:3]
    title_h = int(title_h_ratio * h)
    title_image = np.zeros((title_h, w, c), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 640  # 1 for image size 640 is fine
    font_thickness = 2
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = int(w / 2 - (text_size[0] / 2))
    text_y = int(title_h / 2 + (text_size[1] / 2))
    title_image = cv2.putText(title_image, title, (text_x, text_y), font,
                              font_scale, (255, 255, 255), font_thickness)

    return np.vstack([title_image, image])


def add_mult_titles_to_image(image, titles, init_title_h_ratio):
    vis_image = image.copy()
    # Reverse titles
    for idx, line_title in enumerate(titles[::-1]):
        title_h_ratio = init_title_h_ratio / (1 + init_title_h_ratio * idx)
        vis_image = add_title_to_image(vis_image, line_title, title_h_ratio)
    return vis_image


def add_grid_on_image(image, grid_shape, color=(0, 0, 0), thickness=1, alpha=0.2):
    h, w, _ = image.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    vis_image = image.copy()
    # draw vertical lines
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(vis_image, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(vis_image, (0, y), (w, y), color=color, thickness=thickness)

    vis_image = cv2.addWeighted(vis_image, alpha, image, 1.0 - alpha, 0.0)
    return vis_image


def create_images_grid(images, n_cols, n_rows, padding=0):
    h, w, c = images[0].shape
    grid_h = h * n_rows + int(padding * (n_rows - 1))
    grid_w = w * n_cols + int(padding * (n_cols - 1))
    grid_image = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            x_start = int(w + padding) * col
            x_end = x_start + w
            y_start = int(h + padding) * row
            y_end = y_start + h
            image_idx = (row * n_cols) + col
            if image_idx < len(images):
                grid_image[y_start: y_end, x_start: x_end] = images[image_idx]
    return grid_image
