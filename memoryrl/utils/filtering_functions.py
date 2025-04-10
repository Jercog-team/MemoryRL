import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

def filter_data_by_criteria(hist_data, port_data, yes_port_data, criteria, ranges, use_intervals=False):
    """
    Filters data arrays based on criteria and specified ranges, ignoring NaNs.

    Parameters:
    - hist_data (np.array): 2D array with historical data, shape (N, M)
    - port_data (np.array): 2D array with port data, shape (N, M)
    - yes_port_data (np.array): 2D array with yes port data, shape (N, M)
    - criteria (np.array): 3D array with criteria used for filtering, shape (P, Q, R)
    - ranges (list): List of checkpoints or intervals depending on `use_intervals`
      - if `use_intervals` is False, `ranges` should be a list of checkpoints (e.g., [0, 26, 36, 46])
      - if `use_intervals` is True, `ranges` should be a list of (start, end) tuples (e.g., [(0, 26), (29, 39), (44, 50)])
    - use_intervals (bool): Whether `ranges` contains intervals (True) or checkpoints (False)

    Returns:
    - filtered_hist (np.array): Filtered historical data concatenated along the column axis
    - filtered_port (np.array): Filtered port data concatenated along the column axis
    - filtered_yes_port (np.array): Filtered yes port data concatenated along the column axis
    """
    
    hist_filt = []
    port_filt = []
    yes_port_filt = []
    
    # Convert arrays to numpy arrays if they aren't already
    hist_data = np.array(hist_data)
    port_data = np.array(port_data)
    yes_port_data = np.array(yes_port_data)

    if use_intervals:
        # Handle the case where `ranges` are intervals (list of tuples)
        for i, (start, end) in enumerate(ranges):
            hist_tp = hist_data[:, start:end]
            port_tp = port_data[:, start:end]
            yes_port_tp = yes_port_data[:, start:end]

            # Initialize empty filters with the same shape as the interval slices
            hist_tp_filt = np.zeros_like(hist_tp)
            port_tp_filt = np.zeros_like(port_tp)
            yes_port_tp_filt = np.zeros_like(yes_port_tp)

            # Indexing mask to filter out NaNs based on criteria
            mask = ~np.isnan(criteria)
            hist_tp_filt[mask] = hist_tp[mask]
            port_tp_filt[mask] = port_tp[mask]
            yes_port_tp_filt[mask] = yes_port_tp[mask]

            # Append filtered arrays for this interval
            hist_filt.append(hist_tp_filt)
            port_filt.append(port_tp_filt)
            yes_port_filt.append(yes_port_tp_filt)
    
    else:
        # Handle the case where `ranges` are checkpoints (list of integers)
        for i in range(len(ranges) - 1):
            start, end = ranges[i], ranges[i + 1]
            hist_tp = hist_data[:, start:end]
            port_tp = port_data[:, start:end]
            yes_port_tp = yes_port_data[:, start:end]

            # Initialize empty filters with the same shape as the checkpoint slices
            hist_tp_filt = np.zeros_like(hist_tp)
            port_tp_filt = np.zeros_like(port_tp)
            yes_port_tp_filt = np.zeros_like(yes_port_tp)

            # Indexing mask to filter out NaNs based on criteria
            mask = ~np.isnan(criteria)
            hist_tp_filt[mask] = hist_tp[mask]
            port_tp_filt[mask] = port_tp[mask]
            yes_port_tp_filt[mask] = yes_port_tp[mask]

            # Append filtered arrays for this checkpoint range
            hist_filt.append(hist_tp_filt)
            port_filt.append(port_tp_filt)
            yes_port_filt.append(yes_port_tp_filt)

    # Concatenate the filtered data along the column axis
    filtered_hist = np.concatenate(hist_filt, axis=1)
    filtered_port = np.concatenate(port_filt, axis=1)
    filtered_yes_port = np.concatenate(yes_port_filt, axis=1)
    
    return filtered_hist, filtered_port, filtered_yes_port

def get_animals(datasets):
    """ 
    input: datasets
    output: animals reaching criteria """
    min_len = 3
    Anims_criteria = []
    for datas in datasets:
        anim_criteria=[]
        if not np.isnan(datas).all():  # If not all values are NaN
            for i,d in enumerate(datas):
                if len(d[~np.isnan(d)]) > min_len:  # Only keep if more than min_len non-NaN values
                    anim_criteria.append(i)
                else:
                    anim_criteria.append(np.nan)  # Append empty if entire dataset is NaN # Append empty if entire dataset is NaN
        Anims_criteria.append(anim_criteria)
    return Anims_criteria
def filtered_trials_by_min_perf(performance, trials, cuetimes,threshold):
    # Initialize the result list
    real_trial_number=[]
    filtered_trials = []
    # Iterate over animals and sessions
    for anim in range(len(performance)):  # For each animal
        animal_filtered_trials = []
        animal_real_trials=[]
        for sess in range(len(performance[anim])):  # For each session
            if not np.isnan(performance[anim][sess]).all():
                maxtrialN=len(cuetimes[anim][sess][~np.isnan(cuetimes[anim][sess])])
                smoothed_perf = np.array(performance[anim][sess][:maxtrialN])
                cumulative_trials = np.array(trials[anim][sess][:maxtrialN])

                # Ensure we have arrays for indexing
                if smoothed_perf.size > 0 and cumulative_trials.size > 0:
                    # Filter trials where smoothed performance exceeds the threshold
                    mask = smoothed_perf > threshold
                    filtered_trials_for_sess = cumulative_trials[mask]

                    # Append the first value if available, otherwise append NaN
                    if len(filtered_trials_for_sess) > 0:
                        animal_filtered_trials.append(filtered_trials_for_sess[0])
                        animal_real_trials.append(filtered_trials_for_sess[0])

                    else:
                        animal_filtered_trials.append(1000)
                        animal_real_trials.append(cumulative_trials[-1])
                        
            else:
                # Handle cases where smoothed_perf or cumulative_trials might not be valid arrays
                animal_filtered_trials.append(np.nan)
                animal_real_trials.append(np.nan)

        filtered_trials.append(animal_filtered_trials)
        real_trial_number.append(animal_real_trials)

    # Convert to a NumPy array for consistency
    filtered_trials = np.array(filtered_trials, dtype=float)
    real_trial_number=np.array(real_trial_number, dtype=float)
    return filtered_trials, real_trial_number
def hist_all(port_seq, ports_poked,NumTrials):
  # ADJUSTING THE ARRAYS TO CREATE MATRIX
  AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}
  cosAng = np.cos([np.pi/4, 0, -np.pi/4, -np.pi/2, -3/4*np.pi, np.pi, 3/4*np.pi, np.pi/2])
  ALL_portsPoked_3D_arr_REC=np.array(ports_poked)
  max_length_nested_arrays=np.shape(ports_poked)[2]
  port_seq_REC=np.array(port_seq)
  # HIST SEQUENCE
  hist_seq=[]
  for aa in range(len(ALL_portsPoked_3D_arr_REC)):
    hist_anim_REC=[]
    for s, sess in enumerate(ALL_portsPoked_3D_arr_REC [aa]):
        hist_sess_REC=[]
        for trl in range(NumTrials):
          if np.shape(ALL_portsPoked_3D_arr_REC [aa,s,trl])==() or  np.shape(ALL_portsPoked_3D_arr_REC [aa,s,trl])==(0,):
            hist_sess_REC.append(np.array([0 for _ in range(8)]))
          else:
            filtered_pokes = np.array(np.hstack(ALL_portsPoked_3D_arr_REC [aa,s,trl]).tolist())
            port_counts = Counter(filtered_pokes)
            hist_sess_REC.append(np.array([port_counts[port] for port in AngRad_dict]))
        hist_anim_REC.append(hist_sess_REC)
    hist_seq.append(np.array(hist_anim_REC))

  hist_seq_arr=np.array(hist_seq)
  return hist_seq_arr

def water_availability(datas):
    new_list = np.array([[array.item() for array in inner_list] for inner_list in datas])
    first_occurrences = [
        (sub_list.tolist().index(1) if 1 in sub_list.tolist() else len(sub_list) - 1) if len(sub_list) > 0 else None
        for sub_list in new_list
    ]
    first_occurrences_ = np.array([[] if value is None else [value] for value in first_occurrences])
    return first_occurrences_[0][0]

def filt_variables_NMDA_LGI1(variable, indexes, animals_range, days_lenght, flag_for_drug):
    if flag_for_drug=='CONTROL-BEFORE':
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][0]-days_lenght-50:indexes[0][0]-days_lenght] for aa in range(0,len(variable))])
        else:
            variable_filt = np.array([variable[aa][indexes[1][0]-days_lenght-50:indexes[1][0]-days_lenght] for aa in range(0,len(variable))])
    
    if flag_for_drug=='CONTROL-BEFORE NMDA':
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][0]-days_lenght:indexes[0][0]] for aa in range(0,len(variable))])
        else:
            variable_filt = np.array([variable[aa][indexes[1][0]-days_lenght:indexes[1][0]] for aa in range(0,len(variable))])

    if flag_for_drug=='CONTROL-AFTER':
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][1]:indexes[0][1]+days_lenght] for aa in range(0,len(variable))])
        else:
            variable_filt = np.array([variable[aa][indexes[1][1]:indexes[1][1]+days_lenght] for aa in range(0,len(variable))])
    
    if flag_for_drug=='CONTROL-BETWEEN':
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][0]-days_lenght:indexes[0][0]] for aa in range(0,len(variable))])
        else:
            variable_filt = np.array([variable[aa][indexes[1][0]-days_lenght:indexes[1][0]] for aa in range(0,len(variable))])

    if (flag_for_drug=='NMDA' or flag_for_drug=='LGI1') :
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][1]-days_lenght:indexes[0][1]] for aa in range(0,len(variable)) ])
        else:
            variable_filt = np.array([variable[aa][indexes[1][1]-days_lenght:indexes[1][1]] for aa in range(0,len(variable)) ])

    if flag_for_drug=='CONTROL-LAST':
        if animals_range==(0,18):
            variable_filt = np.array([variable[aa][indexes[0][1]+4:indexes[0][1]+days_lenght] for aa in range(0,len(variable))])
        else:
            variable_filt = np.array([variable[aa][indexes[1][1]+4:indexes[1][1]+days_lenght] for aa in range(0,len(variable))])
    
    return variable_filt

def trials_emergency(CorrectLicks,FlagZeroSess):
    trials_aa=[]
    for aa in range(len(CorrectLicks)):
        trials_sess=[]
        for sess in range(len(CorrectLicks[aa])):
            if np.shape(CorrectLicks[aa][sess])==():
                trials_sess.append(np.nan)
            else:
                mask=CorrectLicks[aa][sess]>4
                if  np.all(~mask)==True:
                    trials_sess.append(FlagZeroSess)
                else:
                    trials=len(CorrectLicks[aa][sess][mask])
                    trials_sess.append(trials)
        trials_aa.append(np.array(trials_sess))
    return np.array(trials_aa)

def calculate_smoothed_performance(array, sigma=5):
    """
    This function flattens a multi-dimensional array, calculates cumulative performance
    based on correct responses (non-NaN values), and applies Gaussian smoothing.

    Parameters:
    - array: A multi-dimensional array where non-NaN values represent correct responses.
    - sigma: Standard deviation for Gaussian smoothing. Default is 5.

    Returns:
    - smoothed_performance: The Gaussian-smoothed cumulative performance.
    - cumulative trials
    """
    # Flatten the array to a 1D array
    if not np.isnan(array).all():
      flat_array = array.flatten()

      # Create a boolean mask for correct responses (non-NaN values)
      is_correct = ~np.isnan(flat_array)

      # Calculate cumulative sums (correct responses)
      cumulative_correct = np.cumsum(is_correct)

      # Total number of trials
      cumulative_trials = np.arange(1, len(flat_array) + 1)

      # Calculate cumulative performance as a percentage
      cumulative_performance = (cumulative_correct / cumulative_trials) * 100

      # Apply Gaussian smoothing to the cumulative performance
      smoothed_performance = gaussian_filter1d(cumulative_performance, sigma=sigma)
    else:
      smoothed_performance=np.nan
      cumulative_trials=np.nan

    return smoothed_performance, cumulative_trials

def circular_distance(angle1, angle2):
  '''Computes the circular distance in between 2 ports in radians
  '''
  angle_difference = np.abs(angle2 - angle1)
  wrapped_angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
  return wrapped_angle_difference

# distance=np.array(circular_distance(prts,prts2))*8/(2*np.pi)