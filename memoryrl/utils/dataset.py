from pathlib import Path
import numpy as np
import scipy.io
from constants import ANG_RAD_DICT
from histogram import hist_all
from angles import convert_to_angular_poked, circular_distance



def CleaningRawDataset_REC_TsW(
    DrugType: str = "CONTROL",
    FlagforSessions: str = "CONTROL",
):
    """
    Load and clean raw 8-port maze recall data from multiple MATLAB batches.

    This function:
    - Chooses a data path depending on availability (Z: or C:).
    - Loads multiple `Output_8PortMazeAnalAVerRecallAllAnimals*.mat` files.
    - Filters sessions by:
        * drug type (CONTROL, NMDA, LGI1),
        * date ranges depending on batch and `FlagforSessions`,
        * state parameters (StaMin/StaMax),
        * minimum number of trials.
    - Extracts per-animal, per-session variables:
        * ports poked per trial,
        * circular average of pokes per trial,
        * correct port per session,
        * “previous day” correct port,
        * session dates,
        * water availability trial index.

    Parameters
    ----------
    DrugType : {"CONTROL", "NMDA", "LGI1"}, default "CONTROL"
        Which animals to include based on infusion.
    FlagforSessions : {"CONTROL", "NMDA", "LGI1"}, default "CONTROL"
        Which date window to use for each batch (e.g. before/around infusion).

    Returns
    -------
    ALL_portsPoked_REC : list
        For each animal (across all batches), a [n_sessions, 56] array of
        per-trial arrays with port indices poked (variable-length per trial).
    ALL_avgportsPoked_REC : list
        Same structure as ALL_portsPoked_REC but with circular-mean angles
        of pokes (in radians) per trial (NaN if no pokes).
    ALL_PORTS_REC : list
        For each animal, a vector of session-wise correct ports.
    ALL_PORTS_YES_REC : list
        For each animal, a vector of correct ports from the previous session.
    ALL_dates : list
        For each animal, a vector of session dates (ExpDay).
    ALL_availability : list
        For each animal, a vector giving the index (trial) at which water
        became available in that session.
    """
    # Two possible root paths where the MATLAB data may live
    path1 = Path(r"Z:\Raw_Data_8PortsMaze")
    path2 = Path(r"C:\Users\User\Documents\cajal\code\data")

    # Select the first path that exists
    pathData = path1 if path1.exists() else path2
    if not path1.exists():
        print(f"Path {path1} does not exist, using {path2} instead.")

    # Filenames for batches (3 batches used)
    fileName1 = "Output_8PortMazeAnalAVerRecallAllAnimals8Batch.mat"
    fileName2 = "Output_8PortMazeAnalAVerRecallAllAnimals9Batch.mat"
    fileName3 = "Output_8PortMazeAnalAVerRecallAllAnimals11Batch.mat"

    # Load data from files into a list of dict-like objects (MATLAB structs)
    S = [scipy.io.loadmat(pathData / fileName)
         for fileName in [fileName1, fileName2, fileName3]]

    # Parameters for filtering sessions
    StaMin = 3
    StaMax = 4
    MinNumTrials = 1

    # Map DrugType string to a list of animal IDs (coding from MATLAB)
    if DrugType == "CONTROL":
        DrugType = [0, 1, 3, 5, 8, 9, 7, 14, 15, 16]  # control / non-infused
    elif DrugType == "NMDA":
        DrugType = [6, 10]  # anti-NMDAR infused animals
    elif DrugType == "LGI1":
        DrugType = [11]  # anti-LGI1 infused animals

    # Containers for aggregated results across batches
    ALL_FirstTrlWWater_REC = []
    ALL_FirstTrlWaterAvai_REC = []
    ALL_portsPoked_REC = []
    ALL_avgportsPoked_REC = []
    ALL_OnlyCorrectPortsPoked_REC = []
    ALL_IntTrialportsPoked_REC = []
    ALL_PORTS_REC = []
    ALL_PORTS_YES_REC = []
    ALL_dates = []
    ALL_availability = []

    # Loop over batches
    for bb in range(0, 3):
        # -----------------------------
        # Select date windows per batch
        # -----------------------------
        if bb == 0:
            if FlagforSessions == "CONTROL":
                YearStart, MonthStart, DayStart = 2019, 10, 21
                YearEnd, MonthEnd, DayEnd = 2020, 12, 31
            elif FlagforSessions == "NMDA":
                YearStart, MonthStart, DayStart = 2020, 1, 11
                YearEnd, MonthEnd, DayEnd = 2020, 3, 8
            elif FlagforSessions == "LGI1":
                YearStart, MonthStart, DayStart = 2020, 6, 12
                YearEnd, MonthEnd, DayEnd = 2020, 7, 17

        elif bb == 1:
            if FlagforSessions == "CONTROL":
                YearStart, MonthStart, DayStart = 2021, 1, 1
                YearEnd, MonthEnd, DayEnd = 2021, 7, 12
            elif FlagforSessions == "NMDA":
                YearStart, MonthStart, DayStart = 2021, 3, 26
                YearEnd, MonthEnd, DayEnd = 2021, 5, 19
            elif FlagforSessions == "LGI1":
                YearStart, MonthStart, DayStart = 2021, 5, 24
                YearEnd, MonthEnd, DayEnd = 2021, 6, 24

        elif bb == 2:
            if FlagforSessions == "CONTROL":
                YearStart, MonthStart, DayStart = 2022, 1, 13
                YearEnd, MonthEnd, DayEnd = 2022, 12, 31
            elif FlagforSessions == "NMDA":
                YearStart, MonthStart, DayStart = 2022, 4, 8
                YearEnd, MonthEnd, DayEnd = 2022, 5, 5
            elif FlagforSessions == "LGI1":
                YearStart, MonthStart, DayStart = 2022, 4, 8
                YearEnd, MonthEnd, DayEnd = 2022, 4, 25

        # NOTE: bb == 3 and bb == 4 branches are kept in case of future batches,
        # but currently bb only runs 0,1,2.

        # Build numpy datetime64 for filtering
        start_date = np.datetime64(f"{YearStart:04d}-{MonthStart:02d}-{DayStart:02d}")
        end_date = np.datetime64(f"{YearEnd:04d}-{MonthEnd:02d}-{DayEnd:02d}")

        Nanimals = S[bb]["CorrectPortPOSTG"].shape[0]
        Nsess = S[bb]["CorrectPortPOSTG"].shape[1]

        # Allocate per-animal containers for this batch
        anim_portsPoked = np.full((Nanimals, 157, 56), np.nan, dtype=object)
        anim_avgportsPoked = np.full((Nanimals, 157, 56), np.nan, dtype=object)
        anim_FirstTrlWWater = np.full((Nanimals, 157), np.nan, dtype=object)
        anim_FirstTrlWaterAvai = np.full((Nanimals, 157), np.nan, dtype=object)
        anim_OnlyCorrectPortsPoked = []
        anim_IntTrialportsPoked = []
        anim_PORTS = np.full((Nanimals, 157), 0)
        anim_PORTS_yes = np.full((Nanimals, 157), 0)
        anim_dates = np.full((Nanimals, 157), np.nan, dtype=object)
        anim_avaiability = np.full((Nanimals, 157), np.nan, dtype=object)

        # Reshape ExpDay to (n_animals * n_sessions, 3) for unique dates
        ExpDay_reshape = S[bb]["ExpDay"].reshape(-1, S[bb]["ExpDay"].shape[-1])
        ExpDates = np.unique(ExpDay_reshape, axis=0)[1:]  # all experiment dates

        # -----------------------------
        # Loop over animals in batch bb
        # -----------------------------
        for aa in range(Nanimals):
            # Per-animal, per-session containers
            sess_portsPoked = np.full((157, 56), np.nan, dtype=object)
            sess_avgportsPoked = np.full((157, 56), np.nan, dtype=object)
            sess_OnlyCorrectPortsPoked = []
            sess_IntTrialportsPoked = []
            sess_PORTS = np.full(157, 0, dtype=object)
            sess_PORTS_yes = np.full(157, 0, dtype=object)
            sess_Dates = np.full(157, 0, dtype=object)
            sess_availability = np.full(157, np.nan, dtype=object)

            FirstTrlWWater = np.full(157, np.nan, dtype=object)
            FirstTrlWaterAvai = np.full(157, np.nan, dtype=object)

            # Precompute water availability / cue / correct & incorrect times
            waterAvailable = (
                S[bb]["POSTwS"][aa] - S[bb]["POSTExpStartTime"][aa] + 0.01
            )
            cues = S[bb]["POSTCueTimes"][aa] - S[bb]["POSTExpStartTime"][aa]
            correctTimes = (
                S[bb]["AllAnimalsSessTrialsPOSTCorrectTrialRT"][aa]
                + S[bb]["POSTCueTimes"][aa]
                - S[bb]["POSTExpStartTime"][aa]
            )
            incorrectTimes = (
                S[bb]["AllAnimalsSessTrialsPOSTIncorrectLicks"][aa]
                + S[bb]["POSTCueTimes"][aa]
                - S[bb]["POSTExpStartTime"][aa]
            )
            incorrectPorts = S[bb]["AllAnimalsSessTrialsPOSTIncorrPortLicked"][aa]
            correctPorts = S[bb]["CorrectPortPOSTG"][aa]

            sess = 0  # index into S[bb]['...'][aa, sess, ...]

            # -----------------------------
            # Loop over unique dates (ExpDates)
            # -----------------------------
            for count in range(Nsess):
                # For each date, prepare containers per-trial
                avgportsPoked = np.full(56, np.nan, dtype=object)
                portsPoked = np.full(56, np.nan, dtype=object)
                OnlyCorrectPortsPoked = []
                IntTrialportsPoked = []

                # Check if this date is the one of this animal/session
                if np.array_equal(ExpDates[count], S[bb]["ExpDay"][aa][sess]):
                    IN = 0

                    # Check that water availability field is not empty
                    if (
                        np.shape(S[bb]["POSTWaterAvailabilityPerAnimalSess"][aa][sess])
                        != (1, 0)
                    ):
                        # Build numpy datetime64 for this specific session date
                        vec_day = S[bb]["POSTExpDay"][aa, sess, :].squeeze()
                        VecDay = np.datetime64(
                            f"{int(vec_day[0]):04d}-{int(vec_day[1]):02d}-{int(vec_day[2]):02d}",
                            "D",
                        )
                        # Date within [start_date, end_date]?
                        if start_date <= VecDay <= end_date:
                            IN = 1

                    # Basic cue presence check
                    if S[bb]["POSTCueTimes"][aa, sess][0].all():
                        # Filter sessions by:
                        # - minimum number of incorrect lick trials,
                        # - presence of ExpParams
                        if (
                            MinNumTrials
                            <= S[bb]["AllAnimalsSessTrialsPOSTIncorrPortLicked"][
                                aa, sess
                            ].shape[1]
                            and S[bb]["ExpParams"][aa, sess] is not None
                        ):
                            # Check state range and that animal belongs to DrugType group
                            if (
                                StaMin
                                <= S[bb]["ExpParams"][aa, sess][0][0]
                                <= StaMax
                                and IN == 1
                                and any(
                                    (
                                        S[bb]["ExpParams"][aa, sess][0][1] - DrugType
                                    )
                                    == 0
                                )
                            ):
                                # Store correct port for this session and previous session
                                sess_PORTS[count] = np.array(
                                    S[bb]["CorrectPortPOSTG"][aa, sess]
                                )
                                sess_PORTS_yes[count] = np.array(
                                    S[bb]["CorrectPortPOSTG"][aa, sess - 1]
                                )
                                sess_Dates[count] = np.array(
                                    S[bb]["ExpDay"][aa][sess]
                                )
                                print(
                                    "Animal: ",
                                    aa,
                                    "Session: ",
                                    sess,
                                    "Date: ",
                                )

                                # Water availability trial index
                                sess_availability[count] = np.where(
                                    S[bb]["POSTWaterAvailabilityPerAnimalSess"][aa][
                                        sess
                                    ]
                                    == 1
                                )[1][0]

                                # -----------------------------
                                # Build trial-wise portsPoked and avgportsPoked
                                # -----------------------------
                                if (
                                    cues[sess].flatten()[0]
                                    > waterAvailable[sess]
                                ):
                                    # Case where water becomes available after first cue
                                    if (
                                        not np.isnan(
                                            incorrectTimes[sess].flatten()[0]
                                        ).all()
                                        and not np.isnan(
                                            correctTimes[sess].flatten()[0]
                                        )
                                    ):
                                        # Merge incorrect and correct pokes with times,
                                        # then sort by time and keep all up to (and including) first correct poke
                                        diccionario = {}
                                        for i, poke in enumerate(
                                            incorrectPorts[sess]
                                            .flatten()[0][0]
                                        ):
                                            diccionario[poke] = (
                                                incorrectTimes[sess]
                                                .flatten()[0][0][i]
                                            )
                                        diccionario[
                                            correctPorts[sess]
                                        ] = correctTimes[sess].flatten()[0]

                                        times = np.array(
                                            [
                                                value
                                                for _, value in sorted(
                                                    diccionario.items(),
                                                    key=lambda item: item[1],
                                                )
                                            ]
                                        )
                                        pokes = np.array(
                                            [
                                                key
                                                for key, _ in sorted(
                                                    diccionario.items(),
                                                    key=lambda item: item[1],
                                                )
                                            ]
                                        )

                                        # Take all pokes up to and including correct port
                                        if correctPorts[sess] in pokes:
                                            correct_index = list(pokes).index(
                                                correctPorts[sess]
                                            )
                                            portsPoked[0] = np.atleast_1d(
                                                np.array(pokes[: correct_index + 1])
                                            )
                                        else:
                                            portsPoked[0] = np.atleast_1d(pokes)

                                        Poked = portsPoked[0]
                                        rad_pokes = np.array(
                                            [ANG_RAD_DICT[p] for p in Poked]
                                        )
                                        # Circular mean of pokes
                                        avgportsPoked[0] = np.angle(
                                            np.exp(1j * rad_pokes).mean()
                                        )

                                    elif (
                                        not np.isnan(
                                            incorrectTimes[sess].flatten()[0]
                                        ).all()
                                        and np.isnan(
                                            correctTimes[sess].flatten()[0]
                                        )
                                    ):
                                        # Only incorrect pokes
                                        portsPoked[0] = np.atleast_1d(
                                            np.array(
                                                [
                                                    incorrectPorts[sess]
                                                    .flatten()[0][0][0]
                                                ]
                                            )
                                        )
                                        Poked = portsPoked[0]
                                        rad_pokes = np.array(
                                            [ANG_RAD_DICT[p] for p in Poked]
                                        )
                                        avgportsPoked[0] = np.angle(
                                            np.exp(1j * rad_pokes).mean()
                                        )

                                    elif (
                                        np.isnan(
                                            incorrectTimes[sess].flatten()[0]
                                        ).all()
                                        and not np.isnan(
                                            correctTimes[sess].flatten()[0]
                                        )
                                    ):
                                        # Only correct poke
                                        portsPoked[0] = np.atleast_1d(
                                            np.array([correctPorts[sess]])
                                        )
                                        Poked = portsPoked[0]
                                        rad_pokes = np.array(
                                            [ANG_RAD_DICT[p] for p in Poked]
                                        )
                                        avgportsPoked[0] = np.angle(
                                            np.exp(1j * rad_pokes).mean()
                                        )

                                else:
                                    # Case where water becomes available before some cues:
                                    # consider all trials up to water availability
                                    valid_trials = np.where(
                                        cues[sess].flatten()
                                        < waterAvailable[sess]
                                    )[0]

                                    for trial in valid_trials:
                                        last_trial = (
                                            trial == valid_trials[-1]
                                        )

                                        inc_t = incorrectTimes[sess].flatten()[
                                            trial
                                        ]
                                        cor_t = correctTimes[sess].flatten()[
                                            trial
                                        ]

                                        inc_all_nan = np.isnan(inc_t).all()
                                        cor_is_nan = np.isnan(cor_t)

                                        if last_trial:
                                            # Only include pokes up to waterAvailable
                                            if (
                                                not inc_all_nan
                                                and not cor_is_nan
                                            ):
                                                diccionario = {}
                                                for i, poke in enumerate(
                                                    incorrectPorts[sess]
                                                    .flatten()[trial][0]
                                                ):
                                                    diccionario[poke] = (
                                                        incorrectTimes[sess]
                                                        .flatten()[trial][0][i]
                                                    )
                                                diccionario[
                                                    correctPorts[sess]
                                                ] = cor_t

                                                times = np.array(
                                                    [
                                                        value
                                                        for _, value in sorted(
                                                            diccionario.items(),
                                                            key=lambda item: item[
                                                                1
                                                            ],
                                                        )
                                                    ]
                                                )
                                                pokes = np.array(
                                                    [
                                                        key
                                                        for key, _ in sorted(
                                                            diccionario.items(),
                                                            key=lambda item: item[
                                                                1
                                                            ],
                                                        )
                                                    ]
                                                )

                                                portsPoked[trial] = np.atleast_1d(
                                                    pokes[
                                                        times
                                                        < waterAvailable[sess]
                                                    ]
                                                )

                                            elif (
                                                not inc_all_nan
                                                and cor_is_nan
                                            ):
                                                pokes = incorrectPorts[
                                                    sess
                                                ].flatten()[trial][0]
                                                times = incorrectTimes[
                                                    sess
                                                ].flatten()[trial][0]
                                                mask = (
                                                    times
                                                    < waterAvailable[sess]
                                                )
                                                portsPoked[trial] = np.atleast_1d(
                                                    pokes[mask]
                                                )
                                                Poked = pokes[mask]
                                                rad_pokes = np.array(
                                                    [
                                                        ANG_RAD_DICT[p]
                                                        for p in Poked
                                                    ]
                                                )
                                                avgportsPoked[trial] = np.angle(
                                                    np.exp(1j * rad_pokes).mean()
                                                )

                                            elif (
                                                inc_all_nan
                                                and not cor_is_nan
                                            ):
                                                pokes = np.array(
                                                    [correctPorts[sess]]
                                                )
                                                times = np.array([cor_t])
                                                mask = (
                                                    times
                                                    < waterAvailable[sess]
                                                )
                                                portsPoked[trial] = np.atleast_1d(
                                                    pokes[mask]
                                                )
                                                Poked = pokes[mask]
                                                rad_pokes = np.array(
                                                    [
                                                        ANG_RAD_DICT[p]
                                                        for p in Poked
                                                    ]
                                                )
                                                avgportsPoked[trial] = np.angle(
                                                    np.exp(1j * rad_pokes).mean()
                                                )

                                        else:
                                            # Before last trial: just include all pokes
                                            if (
                                                not inc_all_nan
                                                and not cor_is_nan
                                            ):
                                                portsPoked[trial] = np.concatenate(
                                                    [
                                                        incorrectPorts[sess]
                                                        .flatten()[trial][0],
                                                        correctPorts[sess]
                                                        .flatten(),
                                                    ]
                                                )
                                                Poked = portsPoked[trial]
                                                rad_pokes = np.array(
                                                    [
                                                        ANG_RAD_DICT[p]
                                                        for p in Poked
                                                    ]
                                                )
                                                avgportsPoked[trial] = np.angle(
                                                    np.exp(1j * rad_pokes).mean()
                                                )

                                            elif (
                                                not inc_all_nan
                                                and cor_is_nan
                                            ):
                                                portsPoked[trial] = np.atleast_1d(
                                                    incorrectPorts[sess]
                                                    .flatten()[trial][0]
                                                )
                                                Poked = portsPoked[trial]
                                                rad_pokes = np.array(
                                                    [
                                                        ANG_RAD_DICT[p]
                                                        for p in Poked
                                                    ]
                                                )
                                                avgportsPoked[trial] = np.angle(
                                                    np.exp(1j * rad_pokes).mean()
                                                )

                                            elif (
                                                inc_all_nan
                                                and not cor_is_nan
                                            ):
                                                portsPoked[trial] = np.atleast_1d(
                                                    correctPorts[sess]
                                                    .flatten()
                                                    .astype(int)
                                                )
                                                Poked = portsPoked[trial]
                                                rad_pokes = np.array(
                                                    [
                                                        ANG_RAD_DICT[p]
                                                        for p in Poked
                                                    ]
                                                )
                                                avgportsPoked[trial] = np.angle(
                                                    np.exp(1j * rad_pokes).mean()
                                                )

                                # Store per-session results for this count
                                sess_portsPoked[count] = np.array(portsPoked)
                                sess_avgportsPoked[count] = np.array(avgportsPoked)
                                sess_OnlyCorrectPortsPoked.append(
                                    np.array(OnlyCorrectPortsPoked)
                                )
                                sess_IntTrialportsPoked.append(IntTrialportsPoked)
                                FirstTrlWaterAvai[count] = len(
                                    cues[sess].flatten()[
                                        waterAvailable[sess]
                                        > cues[sess].flatten()
                                    ]
                                )

                    # Move to next session index in S
                    sess += 1

                else:
                    # Date mismatch: just increment session index or leave as-is
                    sess = sess

            # Attach per-animal data to batch-level containers
            anim_portsPoked[aa] = np.array(sess_portsPoked)
            anim_avgportsPoked[aa] = np.array(sess_avgportsPoked)
            anim_FirstTrlWWater[aa] = np.array(FirstTrlWWater)
            anim_FirstTrlWaterAvai[aa] = np.array(FirstTrlWaterAvai)
            anim_OnlyCorrectPortsPoked.append(sess_OnlyCorrectPortsPoked)
            anim_IntTrialportsPoked.append(sess_IntTrialportsPoked)
            anim_PORTS[aa] = np.array(sess_PORTS)
            anim_PORTS_yes[aa] = np.array(sess_PORTS_yes)
            anim_dates[aa] = np.array(sess_Dates)
            anim_avaiability[aa] = np.array(sess_availability)

        # Append batch-level arrays to global containers
        ALL_portsPoked_REC.extend(anim_portsPoked)
        ALL_PORTS_REC.extend(np.array(anim_PORTS))
        ALL_PORTS_YES_REC.extend(np.array(anim_PORTS_yes))
        ALL_avgportsPoked_REC.extend(anim_avgportsPoked)
        ALL_dates.extend(np.array(anim_dates))
        ALL_availability.extend(np.array(anim_avaiability))

    return (
        ALL_portsPoked_REC,
        ALL_avgportsPoked_REC,
        ALL_PORTS_REC,
        ALL_PORTS_YES_REC,
        ALL_dates,
        ALL_availability,
    )




def dataset_MI(ALL_portsPoked_REC,
               ALL_PORTS_REC,
               ALL_PORTS_LAGS_REC) -> Tuple[
                   np.ndarray,  # Big_Hist_data
                   np.ndarray,  # arr_ports
                   np.ndarray,  # lags_arr_ports
                   np.ndarray,  # distance_seq
                   np.ndarray,  # arr_all_trials_pokes
                   np.ndarray,  # arr_all_trials_pokes_polar
                   np.ndarray,  # arr_polar_pokes
                   np.ndarray,  # mask_nan
               ]:
    """
    Build the dataset needed for mutual information analysis.

    Parameters
    ----------
    ALL_portsPoked_REC : nested sequence
        All poke events per trial/session/animal.
    ALL_PORTS_REC : sequence
        Sequence of ports (e.g. reward ports) per session/animal.
    ALL_PORTS_LAGS_REC : sequence
        Sequence of lagged ports aligned with ALL_PORTS_REC.

    Returns
    -------
    Big_Hist_data : np.ndarray
        Total number of pokes per trial (sum over ports).
    arr_ports : np.ndarray
        Flattened array of current ports, filtered where lag > 0.
    lags_arr_ports : np.ndarray
        Flattened array of lagged ports corresponding to arr_ports.
    distance_seq : np.ndarray
        Circular distance in "port units" between arr_ports and lags_arr_ports.
    arr_all_trials_pokes : np.ndarray
        Flattened array of poke histograms (n_trials, N_PORTS) after selection.
    arr_all_trials_pokes_polar : np.ndarray
        2D array (n_sessions_like, max_len) of angular sequences (flattened).
    arr_polar_pokes : np.ndarray
        Object array of sessions, each (n_trials, N_PORTS) with angular values.
    mask_nan : np.ndarray
        Boolean mask indicating which entries in the original sequences
        had lag > 0 (used as selection mask).
    """
    # Concatenate port sequences across all sessions/animals
    port_seqs = np.concatenate([ALL_PORTS_REC[aa] for aa in range(len(ALL_PORTS_REC))])
    lags_port_seqs = np.concatenate(
        [ALL_PORTS_LAGS_REC[aa] for aa in range(len(ALL_PORTS_LAGS_REC))]
    )

    # Mask: only entries with lag > 0
    mask_nan = lags_port_seqs > 0

    arr_ports = port_seqs[mask_nan]
    lags_arr_ports = lags_port_seqs[mask_nan]

    # Build histograms of pokes per trial for all animals/sessions
    all_trials_hist = hist_all(ALL_PORTS_REC, ALL_portsPoked_REC, num_trials=56)

    # Flatten across animals/sessions and align with lag>0 mask
    flat_hist = np.concatenate(
        [all_trials_hist[aa] for aa in range(len(all_trials_hist))],
        axis=0,
    )
    arr_all_trials_pokes = flat_hist[mask_nan]

    # Map ports (1..8) to angles (radians)
    ports_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(arr_ports))
    lags_ports_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(lags_arr_ports))

    # Circular distance in radians
    dist_radians = circular_distance(ports_rad, lags_ports_rad)

    # Convert from radians to "port units" (0..N_PORTS)
    distance_seq = dist_radians * N_PORTS / (2 * np.pi)

    # Total pokes per trial
    Big_Hist_data = np.sum(arr_all_trials_pokes, axis=1)

    # Convert histograms to angular representation
    arr_all_trials_pokes_polar = convert_to_angular_poked(arr_all_trials_pokes, by_trial=False)
    arr_polar_pokes = convert_to_angular_poked(arr_all_trials_pokes, by_trial=True)

    return (
        Big_Hist_data,
        arr_ports,
        lags_arr_ports,
        distance_seq,
        arr_all_trials_pokes,
        arr_all_trials_pokes_polar,
        arr_polar_pokes,
        mask_nan,
    )


def dataset_MI_TRN(
    ALL_portsPoked_REC,
    ALL_PORTS_REC,
    ALL_PORTS_LAGS_REC,
    ALL_PORTS_2DAYS_REC,
    ALL_dates_REC,
):
    """
    Build a dataset for MI analysis between:
    - port at day t,
    - port at day t-1 (lag 1),
    - port at day t-2 (lag 2 days).

    Parameters
    ----------
    ALL_portsPoked_REC : list
        Output from CleaningRawDataset_REC_TsW (ports poked per trial).
    ALL_PORTS_REC : list
        Correct port per session/day.
    ALL_PORTS_LAGS_REC : list
        Lag-1 correct port per session/day.
    ALL_PORTS_2DAYS_REC : list
        Lag-2 correct port per session/day.
    ALL_dates_REC : list
        Dates per session/day.

    Returns
    -------
    Big_Hist_data : np.ndarray
        Total number of pokes per trial.
    arr_ports : np.ndarray
        Current correct ports (day t).
    lags_arr_ports : np.ndarray
        Lag-1 correct ports (day t-1).
    lag_2Days_ports : np.ndarray
        Lag-2 correct ports (day t-2).
    distance_seq : np.ndarray
        Circular distance (in port units) between lag-1 and lag-2 ports.
    arr_all_trials_pokes : np.ndarray
        Trial-wise histograms (n_trials, N_PORTS) for selected sessions.
    arr_all_trials_pokes_polar : np.ndarray
        Flattened angular sequences per session (by_trial=False).
    arr_polar_pokes : np.ndarray
        Per-trial, per-port angular matrices (by_trial=True).
    dates : np.ndarray
        Corresponding dates for selected sessions.
    animals : np.ndarray
        Animal index for each selected session.
    """
    # Animal index for each (animal, session) position
    animals = np.repeat(np.arange(len(ALL_PORTS_REC)), len(ALL_PORTS_REC[0]))

    port_seqs = np.concatenate(
        [ALL_PORTS_REC[aa] for aa in range(len(ALL_PORTS_REC))]
    )
    lags_port_seqs = np.concatenate(
        [ALL_PORTS_LAGS_REC[aa] for aa in range(len(ALL_PORTS_LAGS_REC))]
    )
    lag_2Days_ports = np.concatenate(
        [ALL_PORTS_2DAYS_REC[aa] for aa in range(len(ALL_PORTS_2DAYS_REC))]
    )
    dates = np.concatenate(
        [ALL_dates_REC[aa] for aa in range(len(ALL_dates_REC))]
    )

    # Build full histograms for all trials
    all_trials_hist = hist_all(ALL_PORTS_REC, ALL_portsPoked_REC, 56)

    # Filter by sessions where lag_2Days_ports > 0
    mask = lag_2Days_ports > 0
    arr_all_trials_pokes = np.concatenate(
        [all_trials_hist[aa] for aa in range(len(all_trials_hist))]
    )[mask]

    arr_ports = port_seqs[mask]
    lags_arr_ports = lags_port_seqs[mask]
    dates = dates[mask]
    animals = animals[mask]
    lag_2Days_ports = lag_2Days_ports[mask]

    # Map ports to angles
    port_seq_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(arr_ports))
    lags_port_seq_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(lags_arr_ports))
    lags_2days_port_seq_rad = np.vectorize(ANG_RAD_DICT.get)(
        np.array(lag_2Days_ports)
    )

    # Distance between lag-1 and lag-2 ports in port units
    distance_seq = (
        circular_distance(lags_port_seq_rad, lags_2days_port_seq_rad)
        * N_PORTS
        / (2 * np.pi)
    )

    Big_Hist_data = np.sum(arr_all_trials_pokes, axis=1)

    # Angular representations:
    # - by_trial=False: flattened angles per "session"
    arr_all_trials_pokes_polar = convert_to_angular_poked(
        arr_all_trials_pokes, by_trial=False
    )
    # - by_trial=True: keep (trial, port) structure
    arr_polar_pokes = convert_to_angular_poked(
        arr_all_trials_pokes, by_trial=True
    )

    return (
        Big_Hist_data,
        arr_ports,
        lags_arr_ports,
        lag_2Days_ports,
        distance_seq,
        arr_all_trials_pokes,
        arr_all_trials_pokes_polar,
        arr_polar_pokes,
        dates,
        animals,
    )


def dataset_MI_avg_ang_poke(
    ALL_portsPoked_REC,
    ALL_PORTS_REC,
    ALL_PORTS_LAGS_REC,
    ALL_avgportsPoked_REC,
):
    """
    Alternative MI dataset using average angular pokes per trial.

    Parameters
    ----------
    ALL_portsPoked_REC : list
        Ports poked per trial (output from CleaningRawDataset_REC_TsW).
    ALL_PORTS_REC : list
        Correct port per session/day.
    ALL_PORTS_LAGS_REC : list
        Lag-1 correct port per session/day.
    ALL_avgportsPoked_REC : list
        Circular-mean angles of pokes per trial (same structure as portsPoked).

    Returns
    -------
    Big_Hist_data : np.ndarray
        Total number of pokes per trial.
    arr_pokes : np.ndarray
        Average angular poke per trial (radians), filtered by lag>0.
    arr_ports : np.ndarray
        Current correct ports, filtered by lag>0.
    lags_arr_ports : np.ndarray
        Lag-1 correct ports, filtered by lag>0.
    distance_seq : np.ndarray
        Circular distance (in port units) between current and lag-1 ports.
    arr_all_trials_pokes : np.ndarray
        Trial-wise histograms (n_trials, N_PORTS) for selected entries.
    arr_all_trials_pokes_polar : np.ndarray
        Flattened angular sequences per session (by_trial=False).
    """
    port_seqs = np.concatenate(
        [ALL_PORTS_REC[aa] for aa in range(len(ALL_PORTS_REC))]
    )
    lags_port_seqs = np.concatenate(
        [ALL_PORTS_LAGS_REC[aa] for aa in range(len(ALL_PORTS_LAGS_REC))]
    )

    # Mask: keep only positions with a valid lag (>0)
    mask = lags_port_seqs > 0

    arr_ports = port_seqs[mask]
    lags_arr_ports = lags_port_seqs[mask]

    # Average angular pokes
    hist_seqs = np.concatenate(
        [ALL_avgportsPoked_REC[aa] for aa in range(len(ALL_avgportsPoked_REC))]
    )
    arr_pokes = hist_seqs[mask]

    # Full histograms of pokes
    all_trials_hist = hist_all(ALL_PORTS_REC, ALL_portsPoked_REC, 56)
    arr_all_trials_pokes = np.concatenate(
        [all_trials_hist[aa] for aa in range(len(all_trials_hist))]
    )[mask]

    # Map ports to angles
    port_seq_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(arr_ports))
    lags_port_seq_rad = np.vectorize(ANG_RAD_DICT.get)(np.array(lags_arr_ports))

    # Distance between ports in port units
    distance_seq = (
        circular_distance(port_seq_rad, lags_port_seq_rad)
        * N_PORTS
        / (2 * np.pi)
    )

    Big_Hist_data = np.sum(arr_all_trials_pokes, axis=1)

    # Flattened angular pokes per session
    arr_all_trials_pokes_polar = convert_to_angular_poked(
        arr_all_trials_pokes, by_trial=False
    )

    return (
        Big_Hist_data,
        arr_pokes,
        arr_ports,
        lags_arr_ports,
        distance_seq,
        arr_all_trials_pokes,
        arr_all_trials_pokes_polar,
    )


def water_availability(datas):
    """
    Given a nested 'water availability' structure, return the index of the
    first trial where water==1 (for the first session), or last index if none.

    This matches your original logic exactly.

    Parameters
    ----------
    datas : array-like
        Typically a nested list/array of 0/1 per trial, per session.

    Returns
    -------
    first_idx : int
        Index of first occurrence of 1 in the *first* row. If no 1 is found,
        returns len(sub_list) - 1. If list is empty, returns raises IndexError.
    """
    new_list = np.array([[array.item() for array in inner_list] for inner_list in datas])

    first_occurrences = [
        (sub_list.tolist().index(1) if 1 in sub_list.tolist() else len(sub_list) - 1)
        if len(sub_list) > 0 else None
        for sub_list in new_list
    ]
    first_occurrences_ = np.array([[] if value is None else [value] for value in first_occurrences])
    return first_occurrences_[0][0]



