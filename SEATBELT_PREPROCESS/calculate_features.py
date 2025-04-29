"""

    This is used for calculating features used for model to be trained/tested

    - types of features:
        - A) Distances between points [4]:
            d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(LEFT_SHOULDER, LEFT_WRIST), d(RIGHT_SHOULDER, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - B) Distances between points [4]:
            d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - C) Area between points [4]:
            S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            S(NOSE, LEFT_ELBOW, RIGHT_ELBOW), S(NOSE, LEFT_WRIST, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - D) Distances + Area between points [4]:
            S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - E) Angles between points [4]:
            a(NOSE, LEFT_SHOULDER, LEFT_ELBOW), a(NOSE, RIGHT_SHOULDER, RIGHT_ELBOW),
            a(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), a(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

"""


import numpy as np

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def normalized_triangle_volume(a, b, c, normalization_value):

    if normalization_value == 0:
        raise ValueError("normalization_value is zero, normalization is not possible.")

    # normalize the points by shoulder distance
    a, b, c = np.array(a) / normalization_value, np.array(b) / normalization_value, np.array(c) / normalization_value

    # compute vectors AB and AC
    ab = b - a
    ac = c - a

    cross_product = np.cross(ab, ac) # compute the cross product of AB and AC

    # the area of the triangle is half the magnitude of the cross product
    area = 0.5 * np.linalg.norm(cross_product)
    return area


def angle(a, b, c):

    ab = a - b  # vector from b to a
    bc = c - b  # vector from b to c

    # calculate dot product and norms
    dot_product = np.dot(ab, bc)
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    # if points are the same, angle is zero
    if norm_ab == 0 or norm_bc == 0: return 0

    angle = np.arccos(dot_product / (norm_ab * norm_bc))
    return np.degrees(angle)  # convert to degrees

"""
    Distances between points:
        d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(LEFT_SHOULDER, LEFT_WRIST), d(RIGHT_SHOULDER, RIGHT_WRIST)
        - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
"""
def calculate_features_A(nose_x, nose_y,
                       left_shoulder_x, left_shoulder_y,
                       right_shoulder_x, right_shoulder_y,
                       left_elbow_x, left_elbow_y,
                       right_elbow_x, right_elbow_y,
                       left_wrist_x, left_wrist_y,
                       right_wrist_x, right_wrist_y):


    # convert into numpy arrays
    nose = np.array([nose_x, nose_y])
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    left_wrist = np.array([left_wrist_x, left_wrist_y])
    right_wrist = np.array([right_wrist_x, right_wrist_y])

    shoulder_distance = distance(left_shoulder, right_shoulder)

    # avoid division by zero by checking shoulder distance
    if shoulder_distance < 1e-5:  # threshold
        return None  # return none shoulder distance is too small

    features = np.array([
        distance(nose, left_elbow),
        distance(nose, right_elbow),
        distance(left_shoulder, left_wrist),
        distance(right_shoulder, right_wrist)
    ]) / shoulder_distance

    return features.tolist()

"""
    Distances between points:
        d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
        - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
"""
def calculate_features_B(nose_x, nose_y,
                       left_shoulder_x, left_shoulder_y,
                       right_shoulder_x, right_shoulder_y,
                       left_elbow_x, left_elbow_y,
                       right_elbow_x, right_elbow_y,
                       left_wrist_x, left_wrist_y,
                       right_wrist_x, right_wrist_y):


    # convert into numpy arrays
    nose = np.array([nose_x, nose_y])
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    left_wrist = np.array([left_wrist_x, left_wrist_y])
    right_wrist = np.array([right_wrist_x, right_wrist_y])

    shoulder_distance = distance(left_shoulder, right_shoulder)

    # avoid division by zero by checking shoulder distance
    if shoulder_distance < 1e-5:  # threshold
        return None  # return none shoulder distance is too small

    features = np.array([
        distance(nose, left_elbow),
        distance(nose, right_elbow),
        distance(nose, left_wrist),
        distance(nose, right_wrist)
    ]) / shoulder_distance

    return features.tolist()

"""
    Area between points:
        S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
"""
def calculate_features_C(nose_x, nose_y,
                       left_shoulder_x, left_shoulder_y,
                       right_shoulder_x, right_shoulder_y,
                       left_elbow_x, left_elbow_y,
                       right_elbow_x, right_elbow_y,
                       left_wrist_x, left_wrist_y,
                       right_wrist_x, right_wrist_y):


    # convert into numpy arrays
    nose = np.array([nose_x, nose_y])
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    left_wrist = np.array([left_wrist_x, left_wrist_y])
    right_wrist = np.array([right_wrist_x, right_wrist_y])

    shoulder_distance = distance(left_shoulder, right_shoulder)

    # avoid division by zero by checking shoulder distance
    if shoulder_distance < 1e-5:  # threshold
        return None  # return none shoulder distance is too small

    features = np.array([
        normalized_triangle_volume(left_shoulder, left_elbow, left_wrist, shoulder_distance),
        normalized_triangle_volume(right_shoulder, right_elbow, right_wrist, shoulder_distance),
        normalized_triangle_volume(nose, left_elbow, right_elbow, shoulder_distance),
        normalized_triangle_volume(nose, left_wrist, right_wrist, shoulder_distance)
    ])

    return features.tolist()

"""
    Distances + Area between points:
            S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
"""
def calculate_features_D(nose_x, nose_y,
                       left_shoulder_x, left_shoulder_y,
                       right_shoulder_x, right_shoulder_y,
                       left_elbow_x, left_elbow_y,
                       right_elbow_x, right_elbow_y,
                       left_wrist_x, left_wrist_y,
                       right_wrist_x, right_wrist_y):


    # convert into numpy arrays
    nose = np.array([nose_x, nose_y])
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    left_wrist = np.array([left_wrist_x, left_wrist_y])
    right_wrist = np.array([right_wrist_x, right_wrist_y])

    shoulder_distance = distance(left_shoulder, right_shoulder)

    # avoid division by zero by checking shoulder distance
    if shoulder_distance < 1e-5:  # threshold
        return None  # return none shoulder distance is too small

    features = np.array([
        normalized_triangle_volume(left_shoulder, left_elbow, left_wrist, shoulder_distance),
        normalized_triangle_volume(right_shoulder, right_elbow, right_wrist, shoulder_distance),
        distance(nose, left_wrist),
        distance(nose, right_wrist)
    ])

    return features.tolist()

"""
    Angles between points:
            a(NOSE, LEFT_SHOULDER, LEFT_ELBOW), a(NOSE, RIGHT_SHOULDER, RIGHT_ELBOW),
            a(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), a(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
"""
def calculate_features_E(nose_x, nose_y,
                       left_shoulder_x, left_shoulder_y,
                       right_shoulder_x, right_shoulder_y,
                       left_elbow_x, left_elbow_y,
                       right_elbow_x, right_elbow_y,
                       left_wrist_x, left_wrist_y,
                       right_wrist_x, right_wrist_y):


    # convert into numpy arrays
    nose = np.array([nose_x, nose_y])
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    left_wrist = np.array([left_wrist_x, left_wrist_y])
    right_wrist = np.array([right_wrist_x, right_wrist_y])

    features = np.array([
        angle(nose, left_shoulder, left_elbow),
        angle(nose, right_shoulder, right_elbow),
        angle(left_shoulder, left_elbow, left_wrist),
        angle(right_shoulder, right_elbow, right_wrist)
    ])

    return features.tolist()


def calculate_features(feature_type, params):
    if feature_type == 'A':
        return calculate_features_A(*params)
    elif feature_type == 'B':
        return calculate_features_B(*params)
    elif feature_type == 'C':
        return calculate_features_C(*params)
    elif feature_type == 'D':
        return calculate_features_D(*params)
    elif feature_type == 'E':
        return calculate_features_E(*params)
    else:
        raise ValueError("Invalid feature type")