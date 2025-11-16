# constants.py
import numpy as np

# Number of ports
N_PORTS: int = 8

# Mapping from port index (1..8) to angle in radians on the circle
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

# Handy list of ports in the canonical order
PORTS_ORDER = list(ANG_RAD_DICT.keys())
