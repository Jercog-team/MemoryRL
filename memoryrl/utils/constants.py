import numpy as np

# Mapping of port numbers (1–8) to angles in radians
ANG_RAD_DICT = {
    1: np.pi / 4,
    2: 0,
    3: -np.pi / 4,
    4: -np.pi / 2,
    5: -3 * np.pi / 4,
    6: np.pi,
    7: 3 * np.pi / 4,
    8: np.pi / 2,
}

# Optional: convenient vectorized version for arrays
vec_port_to_angle = np.vectorize(ANG_RAD_DICT.get)
