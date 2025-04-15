# Return the center of bounding box --> to find the center of ellipse
def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center


# Return the width of bounding box --> to find the major and minor axis of ellipse
def get_width_bbox(bbox):
    x1, x2 = bbox[0], bbox[2]
    return x2 - x1

# Return the distance between two points
def measure_distance(p1, p2):
    return ((p2[0]- p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

# Measure the x &  y movement between two points
def measure_x_y_distance(p1, p2):
    return (p2[0] - p1[0]), (p2[1] - p1[1])

# Get the foot position of bbox
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)