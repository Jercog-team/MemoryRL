# Function to convert histogram data to angular poked format
def convert_to_angular_poked(hist):
    """
    Convert histogram data to angular poked format.

    Parameters:
        hist (list of np.ndarray): List of session histograms.

    Returns:
        np.ndarray: Padded angular poked data.
    """
    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 
                   5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}

    ang_ports_sess = []
    for sess in range(len(hist)):
        ang_ports = []
        for i, row in enumerate(hist[sess]):
            positions = np.where(row != 0)[0] 
            if positions.size > 0: 
                ang_ports.extend([AngRad_dict.get(pos + 1, np.nan) for pos in positions])
        ang_ports_sess.append(ang_ports)
    max_length = max(len(lst) for lst in ang_ports_sess)
    ang_ports_sess_padded = np.array([lst + [np.nan] * (max_length - len(lst)) for lst in ang_ports_sess])

    return ang_ports_sess_padded

# Function to calculate radial average poked
def rad_avg_poked(hist):
    """
    Calculate the radial average poked from histogram data.

    Parameters:
        hist (list of np.ndarray): List of session histograms.

    Returns:
        list: Radial average poked for each session.
    """
    new_AngRad = {1: -3*np.pi/4, 2: -np.pi/2, 3: -np.pi/4, 4: 0, 5: np.pi/4, 6: np.pi/2, 7: 3*np.pi/4, 8: np.pi}
    avg_ang_sess = []
    for sess in range(len(hist)):
        avg_ang = []
        for i, row in enumerate(hist[sess]):
            positions = np.where(row != 0)[0]  # Columns where there is a 1
            if positions.size > 0:  # Only rows with at least 1 value
                mapped_values = [new_AngRad.get(pos + 1, np.nan) for pos in positions]
                avg_ang.append(np.angle(np.exp(1j * np.array(mapped_values)).mean()))
        avg_ang_sess.extend(avg_ang)
    return avg_ang_sess