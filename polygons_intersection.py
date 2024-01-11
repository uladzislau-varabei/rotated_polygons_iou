import torch


class Line:
    # Line representation in the form ax + by + c = 0
    def __init__(self, p1, p2):
        # a, b - n vector is a perpendicular to (p2 - p1) vector
        p1x, p1y = p1
        p2x, p2y = p2
        self.p1 = p1
        self.p2 = p2
        # Set line params
        self.a = -(p2y - p1y)
        self.b = p2x - p1x
        # p1 and p2 are on this line, so ax1 + by1 = -c
        self.c = -(self.a * p1x + self.b * p1y)
        # Set segment ranges
        self.x_min = min(p1x, p2x)
        self.x_max = max(p1x, p2x)
        self.y_min = min(p1y, p2y)
        self.y_max = max(p1y, p2y)

    def check_location_cond(self, point):
        eps = 1e-4  # in case one of the lines is parallel to coordinate axis (improve numerical stability)
        x_cond = self.x_min - eps <= point[0] <= self.x_max + eps
        y_cond = self.y_min - eps <= point[1] <= self.y_max + eps
        return x_cond and y_cond

    def apply_equation(self, points):
        return self.a * points[:, 0] + self.b * points[:, 1] + self.c

    def intersect(self, other_line):
        # TODO: add support of the case when two lines are parallel
        # Cramer rule for 2 linear equations
        w = self.a * other_line.b - other_line.a * self.b + 1e-8
        x = -self.c * other_line.b + other_line.c * self.b
        y = -other_line.c * self.a + self.c * other_line.a
        return torch.stack([x / w, y / w])


def convert_to_edges(points):
    points1 = points
    points2 = torch.cat([points[1:, :], points[0:1, :]])
    return torch.stack([points1, points2], dim=0)  # [2, N, 2]


def check_nested_points(points1, lines2):
    # Check if points1 are all nested within polygon2 which is represented by its edges.
    # To do this, apply all line equations of polygon2 to all points1.
    # If all signs are the same then points1 are nested within polygon2
    all_signs = []
    with torch.no_grad():
        for line2 in lines2:
            signs = torch.sign(line2.apply_equation(points1))
            all_signs.append(signs)
    all_signs = torch.stack(all_signs).unique().numel()
    return all_signs == 1


def intersect_convex_polygons(points1, points2):
    # Algorithm description:
    # 1. Prepare edges of both polygons from points
    # 2. Check the case of nested polygons
    # 3. Iterate over edges of the 1st polygon
    # 4. Find intersection points with the 2nd polygon edges
    # 5. Check location of intersection points. If intersection is on both edges then add it to final polygon
    # 6. If intersection point was added to final polygon, then check location of the first points of both lines.
    # One of them of both can be within another polygon
    # 7. Order points based on polar angle from their center
    # Lines intersection:
    # 1. Each line should be represented in form ax + by + c = 0
    # 2. Intersection point is then (see Cramer rule)
    # 3. Check that intersection point is located on segments forming each line (betwwen p1 and p2)
    # Create polygon edges
    edges1 = convert_to_edges(points1)
    edges2 = convert_to_edges(points2)
    lines1 = [Line(p1, p2) for p1, p2 in zip(edges1[0], edges1[1])]
    lines2 = [Line(p1, p2) for p1, p2 in zip(edges2[0], edges2[1])]

    # First check for nested polygons case
    is_polygon1_nested = check_nested_points(points1, lines2)
    if is_polygon1_nested:
        return points1
    is_polygon2_nested = check_nested_points(points2, lines1)
    if is_polygon2_nested:
        return points2

    # Find all intersection points on lines
    intersection_points = []
    for line1 in lines1:
        for line2 in lines2:
            intersection_point = line1.intersect(line2)
            line1_location_cond = line1.check_location_cond(intersection_point)
            line2_location_cond = line2.check_location_cond(intersection_point)
            if line1_location_cond and line2_location_cond:
                intersection_points.append(intersection_point)
                # If intersection point is on lines, then check vertexes
                line1_point = line1.p1
                line1_vertex_cond = check_nested_points(line1_point[None, :], lines2)
                if line1_vertex_cond:
                    intersection_points.append(line1_point)
                line2_point = line2.p1
                line2_vertex_cond = check_nested_points(line2_point[None, :], lines1)
                if line2_vertex_cond:
                    intersection_points.append(line2_point)

    if len(intersection_points) > 0:
        intersection_points = torch.stack(intersection_points)
        # Sort based on polar angle. Not sure about gradient disabling here
        # with torch.no_grad():
        center_point = intersection_points.mean(axis=0)
        r = intersection_points - center_point[None, :]
        angles = torch.atan2(r[:, 1], r[:, 0])  # coords should be provided in y, x order
        mask = torch.argsort(angles)
        intersection_points = intersection_points[mask]
    else:
        intersection_points = None
    return intersection_points
