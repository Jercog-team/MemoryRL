import numpy as np
from pathlib import Path
import scipy.io

def circular_distance(angle1, angle2):
  angle_difference = np.abs(angle2 - angle1)
  wrapped_angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
  return wrapped_angle_difference


def CleaningRawDataset_trn(pathData,fileNames,DrugType='CONTROL', FlagforSessions='CONTROL'):
  #### This function aims to obtain the Performance, Reaction Time, Errors of all the groups of animals#
  S = [ scipy.io.loadmat (pathData + fileName) for fileName in fileNames]   # Load data from files
  # Parameters
  StaMin = 3
  StaMax = 3
  MinNumTrials = 1

  if DrugType=='CONTROL':
    DrugType=[0, 1, 3, 5, 8, 9, 7, 14, 15, 16] ##CONTROL or non infused
  if DrugType=='NMDA':
    DrugType=[6, 10]  ## anti_NMDAR infused animals
  if DrugType=='LGI1':
    DrugType=[11]   ## anti_LGI1 infused animals


  X = []
  ALL_FirstTrlWWater_trn=[]
  ALL_FirstTrlWaterAvai_trn=[]
  ALL_portsPoked_trn = []
  ALL_OnlyCorrectPortsPoked_trn = []
  ALL_IntTrialportsPoked_trn = []
  ALL_PORTS_trn=[]
  WATER = []
  CP = []

  # Loop over batches
  for file in fileNames:
    if '8NBatch' in file:
        bb=0
        if FlagforSessions=='CONTROL':
            YearStart, MonthStart, DayStart = 2019, 10, 21 # This is the real original start
            YearEnd, MonthEnd, DayEnd = 2020, 12, 31 # This is the real original end
            # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
            YearStart, MonthStart, DayStart = 2020, 1, 11 #24 days before the NMDA infusion period
            YearEnd, MonthEnd, DayEnd =   2020, 3, 8 # Day of NMDA end
                  # ## HIGH PERFORMERS
        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2020, 6, 12 #15 days before LGI1
          YearEnd, MonthEnd, DayEnd = 2020, 7, 17 #Day After LGI1
    elif '9' in file:
        bb=1
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2021, 1, 1 # This is the real original start
          YearEnd, MonthEnd, DayEnd = 2021, 7, 12 # This is the real original end
          ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2021, 3, 26# 2021, 4, 12
          YearEnd, MonthEnd, DayEnd = 2021, 5, 19 #2021, 6, 2

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2021, 5, 24
          YearEnd, MonthEnd, DayEnd = 2021, 6, 24
    elif '11' in file:
        bb=2
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2022, 1, 13# This is the real original start
          YearEnd, MonthEnd, DayEnd = 2022, 12, 31 # This is the real original end
        # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 5, 5

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 4, 25
    elif '12' in file:
        bb=3
        if FlagforSessions=='CONTROL':
           YearStart, MonthStart, DayStart = 2023, 10, 18
           YearEnd, MonthEnd, DayEnd = 2024, 3, 31

    elif '14' in file:
        bb=4
        if FlagforSessions=='CONTROL':
           YearStart, MonthStart, DayStart = 2024, 5, 27
           YearEnd, MonthEnd, DayEnd = 2024, 12, 31


    YearStart_str = str(YearStart)
    MonthStart_str = str(MonthStart).zfill(2)
    DayStart_str = str(DayStart).zfill(2)
    YearEnd_str = str(YearEnd)
    MonthEnd_str = str(MonthEnd).zfill(2)
    DayEnd_str = str(DayEnd).zfill(2)

    start_date = np.datetime64(f'{YearStart_str}-{MonthStart_str}-{DayStart_str}')
    end_date = np.datetime64(f'{YearEnd_str}-{MonthEnd_str}-{DayEnd_str}')

    if bb==1:
        Nsess=142
        Nanimals = S[bb]['CorrectPort'].shape[0]
    else:
        Nanimals = S[bb]['CorrectPort'].shape[0]
        Nsess = S[bb]['CorrectPort'].shape[1]

    anim_portsPoked = np.full((Nanimals,157,200) ,np.nan, dtype=object)
    anim_FirstTrlWWater=np.full((Nanimals,157),np.nan, dtype=object)
    anim_FirstTrlWaterAvai=np.full((Nanimals, 157),np.nan, dtype=object)
    anim_OnlyCorrectPortsPoked = []
    anim_IntTrialportsPoked = []
    anim_PORTS =np.full((Nanimals,157), 0)
    anim_WATER = []

    ExpDay_reshape = S[bb]['ExpDay'].reshape(-1, S[bb]['ExpDay'].shape[-1])
      #Find all the dates the experiment was run
    ExpDates = np.unique(ExpDay_reshape, axis=0)[1:]
    for aa in range(Nanimals):
          # print('Animal: ', aa, bb)
          sess_portsPoked = np.full((157,200) ,np.nan, dtype=object)
          sess_OnlyCorrectPortsPoked = []
          sess_IntTrialportsPoked = []
          sess_PORTS = np.full(157 ,0, dtype=object)
          sess_WATER = []
          FirstTrlWWater=np.full(157,np.nan, dtype=object)
          FirstTrlWaterAvai=np.full(157,np.nan, dtype=object)
          sess=0
          for count in range(Nsess):
            waterAvail = []
            portsPoked = np.full(200, np.nan,dtype=object)
            OnlyCorrectPortsPoked = []
            IntTrialportsPoked = []
            k = 0
            if np.array_equal(ExpDates[count] , S[bb]['ExpDay'][aa][sess]):
                # print(ExpDates[count],S[bb]['ExpDay'][aa][sess] )
                if not np.shape(S[bb]['CueTimes'][aa][sess]) == (1, 0):
                    vec_day = S[bb]['ExpDay'][aa, sess, :].squeeze()
                    VecDay = np.datetime64(f'{vec_day[0]:04d}-{vec_day[1]:02d}-{vec_day[2]:02d}', 'D')
                    if start_date <= VecDay <= end_date:
                        if S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess][0].all():
                            if MinNumTrials <= S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].shape[1] and \
                                        S[bb]['ExpParams'][aa, sess] is not None:
                                if StaMin <= S[bb]['ExpParams'][aa, sess][0][0] <= StaMax  and \
                                            any((S[bb]['ExpParams'][aa, sess][0][1] - DrugType) == 0):
                                    nw = 1
                                    for i in range(len(S[bb]['IncorrectLicksPerAnimalSessTrial'][aa, sess][0])):
                                        if S[bb]['IncorrectLicksPerAnimalSessTrial'][aa, sess][0][i][0].all():
                                            nw += 1
                                        #FirstTrlWWater[sess]=water_availability(S[bb]['WaterAvailabilityPerAnimalSess'][aa, sess])
                                    for t in range(len(S[bb]['WaterAvailabilityPerAnimalSess'][aa, sess][0])):
                                        if not np.isnan(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess][0][t]) and \
                                                    np.size(S[bb]['IncorrectLicksPerAnimalSessTrial'][aa, sess][0][t]!=0):
                                            portsPoked[t]=np.concatenate([S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess][0][t][0].astype(float),
                                                                            np.array([S[bb]['CorrectPort'][aa, sess].astype(float)])])
                                        else:
                                            if not np.isnan(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess][0][t]):
                                                portsPoked[t]=np.array([S[bb]['CorrectPort'][aa, sess]], dtype=float)
                                                OnlyCorrectPortsPoked.append(S[bb]['CorrectPort'][aa, sess].astype(float))
                                            if np.size(S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess][0][t]!=0):
                                                portsPoked[t]=np.array(S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess][0][t][0].astype(float))
                                            if S[bb]['ErrorLickPortIndPerAnimalSessTrial'][aa, sess][0][t][0].all():
                                                IntTrialportsPoked.extend(
                                                        S[bb]['ErrorLickPortIndPerAnimalSessTrial'][aa, sess][0][t])

                                    # ANIMALS VARIABLES APPENDING
                                    sess_portsPoked[count]=np.array(portsPoked)
                                    sess_OnlyCorrectPortsPoked.append(np.array(OnlyCorrectPortsPoked))
                                    sess_IntTrialportsPoked.append(IntTrialportsPoked)
                                    sess_PORTS[count]=np.array(S[bb]['CorrectPort'][aa, sess])
                sess=sess+1                  
            else:
                sess=sess

            anim_portsPoked[aa]=np.array(sess_portsPoked)
            anim_FirstTrlWWater[aa]=np.array(FirstTrlWWater)
            anim_FirstTrlWaterAvai[aa]=np.array(FirstTrlWaterAvai)
            anim_OnlyCorrectPortsPoked.append(sess_OnlyCorrectPortsPoked)
            anim_IntTrialportsPoked.append(sess_IntTrialportsPoked)
            anim_PORTS[aa]=np.array(sess_PORTS)


    ALL_portsPoked_trn.extend(anim_portsPoked)
    ALL_FirstTrlWaterAvai_trn.extend(np.array(anim_FirstTrlWaterAvai))
    ALL_FirstTrlWWater_trn.extend(np.array(anim_FirstTrlWWater))
    # ALL_OnlyCorrectPortsPoked_REC.append(np.array(anim_OnlyCorrectPortsPoked))
    # ALL_IntTrialportsPoked_REC.append(np.array(anim_IntTrialportsPoked))#np.array(anim_IntTrialportsPoked))
    ALL_PORTS_trn.extend(np.array(anim_PORTS))
  return ALL_portsPoked_trn, ALL_PORTS_trn, ALL_FirstTrlWWater_trn

def CorrectTimeLicksAllTreatmentsTRAINING(pathData, fileNames, DrugType='CONTROL',FlagforSessions='CONTROL'):
  #### This function aims to obtain the Performance, Reaction Time, Errors of all the groups of animals#
  S = [ scipy.io.loadmat (pathData + fileName) for fileName in fileNames]   # Load data from files

  if DrugType=='CONTROL':
    DrugType=[0, 1, 3, 5, 8, 9, 7, 14, 15, 16] ##CONTROL or non infused
  if DrugType=='NMDA':
    DrugType=[6, 10]  ## anti_NMDAR infused animals
  if DrugType=='LGI1':
    DrugType=[11]   ## anti_LGI1 infused animals
  StaMin = 3
  StaMax = 3
  # Window = np.ones(Nframes) / Nframes  ###Dont think this step is necessary
  MinNumTrials = 1
  ALL_CorrectTimeLicks=[]
  ALL_Cuetimes=[]
  ALL_ExpStartTime=[]
  bb_CorrectTimeLicks=[]
  bb_Cuetimes=[]
  bb_ExpStartTime=[]


  # Loop over batches
  for file in fileNames:
    if '8NBatch' in file:
        bb=0
        if FlagforSessions=='CONTROL':
            YearStart, MonthStart, DayStart = 2019, 10, 21 # This is the real original start
            YearEnd, MonthEnd, DayEnd = 2020, 12, 31 # This is the real original end
            # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
            YearStart, MonthStart, DayStart = 2020, 1, 11 #24 days before the NMDA infusion period
            YearEnd, MonthEnd, DayEnd =   2020, 3, 8 # Day of NMDA end
                  # ## HIGH PERFORMERS
        if FlagforSessions=='LGI1':
           YearStart, MonthStart, DayStart = 2020, 6, 12 #15 days before LGI1
           YearEnd, MonthEnd, DayEnd = 2020, 7, 17 #Day After LGI1
    elif '9' in file:
        bb=1
        if FlagforSessions=='CONTROL':
            YearStart, MonthStart, DayStart = 2021, 1, 1 # This is the real original start
            YearEnd, MonthEnd, DayEnd = 2021, 7, 12 # This is the real original end
          ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
            YearStart, MonthStart, DayStart = 2021, 3, 26# 2021, 4, 12
            YearEnd, MonthEnd, DayEnd = 2021, 5, 19 #2021, 6, 2

        if FlagforSessions=='LGI1':
            YearStart, MonthStart, DayStart = 2021, 5, 24
            YearEnd, MonthEnd, DayEnd = 2021, 6, 24
    elif '11' in file:
        bb=2
        if FlagforSessions=='CONTROL':
            YearStart, MonthStart, DayStart = 2022, 1, 13# This is the real original start
            YearEnd, MonthEnd, DayEnd = 2022, 12, 31 # This is the real original end
        # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
            YearStart, MonthStart, DayStart = 2022, 4, 8
            YearEnd, MonthEnd, DayEnd = 2022, 5, 5

        if FlagforSessions=='LGI1':
            YearStart, MonthStart, DayStart = 2022, 4, 8
            YearEnd, MonthEnd, DayEnd = 2022, 4, 25
    # elif '12' in file:
    #     bb=3
    #     if FlagforSessions=='CONTROL':
    #         YearStart, MonthStart, DayStart = 2023, 10, 18
    #         YearEnd, MonthEnd, DayEnd = 2024, 3, 31

    # elif '14' in file:
    #     bb=4
    #     if FlagforSessions=='CONTROL':
    #         YearStart, MonthStart, DayStart = 2024, 5, 27
    #         YearEnd, MonthEnd, DayEnd = 2024, 12, 31
    
    # elif '15' in file:
    #     bb=5
    #     if FlagforSessions=='CONTROL':
    #         YearStart, MonthStart, DayStart = 2024, 12, 1
    #         YearEnd, MonthEnd, DayEnd = 2024, 12, 31

    YearStart_str = str(YearStart)
    MonthStart_str = str(MonthStart).zfill(2)
    DayStart_str = str(DayStart).zfill(2)
    YearEnd_str = str(YearEnd)
    MonthEnd_str = str(MonthEnd).zfill(2)
    DayEnd_str = str(DayEnd).zfill(2)

    start_date = np.datetime64(f'{YearStart_str}-{MonthStart_str}-{DayStart_str}')
    end_date = np.datetime64(f'{YearEnd_str}-{MonthEnd_str}-{DayEnd_str}')


    Nanimals = S[bb]['CorrectTimeLicksPerAnimalSessTrial'].shape[0]
    Nsess = S[bb]['CorrectTimeLicksPerAnimalSessTrial'].shape[1]

    CorrectTimeLicks=np.full((Nanimals,157,200) ,np.nan, dtype=object)
    Cuetimes=np.full((Nanimals,157,200) ,np.nan, dtype=object)
    ExpStartTime=np.full((Nanimals,157) ,np.nan, dtype=object)

    ExpDay_reshape = S[bb]['ExpDay'].reshape(-1, S[bb]['ExpDay'].shape[-1])
    #Find all the dates the experiment was run
    ExpDates = np.unique(ExpDay_reshape, axis=0)[1:] #Avoiding the 1st array 
                      #since its the [0,0,0] default from MATLAB code when the animals did not run the task.

    for aa in range(Nanimals):
        sess=0
        for count in range(Nsess):
            # print('Anim: ', aa, 'SESS: ', sess) #if bb==0 or bb==1 or bb==2 :# and not any(np.array_equal(S[bb]['ExpDay'][aa][sess], monday) for monday in mondays_date_array12): ## Need to automatize this
            if np.array_equal(ExpDates[count] , S[bb]['ExpDay'][aa][sess]):
                if not np.shape(S[bb]['CueTimes'][aa][sess])==(1,0):
                    vec_day=S[bb]['ExpDay'][aa, sess, :].squeeze()
                    VecDay = np.datetime64(f'{vec_day[0]:04d}-{vec_day[1]:02d}-{vec_day[2]:02d}', 'D')
                    if start_date <= VecDay <= end_date:
                        if MinNumTrials <= S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].shape[1] and S[bb]['ExpParams'][aa, sess] is not None:
                            if StaMin <= S[bb]['ExpParams'][aa, sess][0][0] <= StaMax and any((S[bb]['ExpParams'][aa,sess][0][1] - DrugType) == 0):
                                CorrectTimeLicks=S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()
                                Cuetimes.append(S[bb]['CueTimes'][aa, sess].squeeze())
                                ExpStartTime.append(S[bb]['ExpStartTime'][aa, sess].squeeze())
                            else:
                                CorrectTimeLicks.append(np.nan)
                                Cuetimes.append(np.nan)
                                ExpStartTime.append(np.nan)
                        else:
                            CorrectTimeLicks.append(np.nan)
                            Cuetimes.append(np.nan)
                            ExpStartTime.append(np.nan)
                    else:
                        CorrectTimeLicks.append(np.nan)
                        Cuetimes.append(np.nan)
                        ExpStartTime.append(np.nan)  
                else:
                    CorrectTimeLicks.append(np.nan)
                    Cuetimes.append(np.nan)
                    ExpStartTime.append(np.nan)
                sess=sess+1
            else:
                CorrectTimeLicks.append(np.nan)
                Cuetimes.append(np.nan)
                ExpStartTime.append(np.nan)
                sess=sess

        # if CorrectTimeLicks:
        #     anim_CorrectTimeLicks.append(CorrectTimeLicks)
        #     anim_Cuetimes.append(Cuetimes)
        #     anim_ExpStartTime.append(ExpStartTime)
        # else:
        #     anim_CorrectTimeLicks.append(np.nan)
        #     anim_Cuetimes.append(np.nan)
        #     anim_ExpStartTime.append(np.nan)

    # bb_CorrectTimeLicks.extend(anim_CorrectTimeLicks)
    # bb_Cuetimes.extend(anim_Cuetimes)
    # bb_ExpStartTime.extend(anim_ExpStartTime)
  ALL_CorrectTimeLicks.extend(anim_CorrectTimeLicks)
  ALL_Cuetimes.extend(anim_Cuetimes)
  ALL_ExpStartTime.extend(anim_ExpStartTime)
    
  # ALL_CorrectTimeLicks=bb_CorrectTimeLicks
  # ALL_Cuetimes[FlagforSessions]=bb_Cuetimes
  # ALL_ExpStartTime[FlagforSessions]=bb_ExpStartTime
  return ALL_CorrectTimeLicks, ALL_Cuetimes, ALL_ExpStartTime

def PerfRTErrorsPerTrialMultiBatchAllTreatmentsTRAINING(pathData, fileNames, DrugType='CONTROL',FlagforSessions='CONTROL'):
    #### This function aims to obtain the Performance, Reaction Time, Errors of all the groups of animals#
    S = [ scipy.io.loadmat (pathData + fileName) for fileName in fileNames]   # Load data from files
    
    if DrugType=='CONTROL':
      DrugType=[0, 1, 3, 5, 8, 9, 7, 14, 15, 16] ##CONTROL or non infused
    if DrugType=='NMDA':
      DrugType=[6, 10]  ## anti_NMDAR infused animals
    if DrugType=='LGI1':
      DrugType=[11]   ## anti_LGI1 infused animalsDT=[0,1,3,5,14,15,16]   ## CONTROL
    StaMin = 3
    StaMax = 3

    # Window = np.ones(Nframes) / Nframes  ###Dont think this step is necessary
    Nfolds = 10
    MinNumTrials = 1
    # anim = 0
    NumMemDays = 1
    X = []
    # ALL_portsPoked={}
    # ALL_IntTrialportsPoked={}
    ALL_MeanRT={}
    ALL_MeanlastRT={}
    ALL_Mean1stRT={}
    ALL_Perf = {}
    ALL_PerfLast={}
    ALL_Perf1st={}
    ALL_Trials={}
    ALL_Inc={}
    ALL_MeanIncorr={}
    ALL_MeanError={}
    ALL_MeanTimeToReachTZ={}
    ALL_Time1stTrialWWater={}
    ALL_FirstTrialWithWater={}
    ALL_CorrectTimeLicks={}



    ALL_CorrectPort={}
    ALL_Date={}

    bb_Perf=[]
    bb_PerfLast=[]
    bb_Perf1st=[]
    bb_MeanRT=[]
    bb_Mean1stRT=[]
    bb_MeanlastRT=[]
    bb_Incorrect=[]
    bb_Trials=[]
    bb_MeanIncorr=[]
    bb_MeanError=[]
    bb_Time1stTrialWWater=[]
    bb_FirstTrialWithWater=[]
    bb_MeanTimeToReachTZ=[]
    bb_CorrectTimeLicks=[]


    bb_CorrectPort=[]
    bb_Date=[]

    # Loop over batches
    for file in fileNames:
      if '8NBatch' in file:
          bb=0
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2019, 10, 21 # This is the real original start
              YearEnd, MonthEnd, DayEnd = 2020, 12, 31 # This is the real original end
              # ## HIGH PERFORMERS
          if FlagforSessions=='NMDA':
              YearStart, MonthStart, DayStart = 2020, 1, 11 #24 days before the NMDA infusion period
              YearEnd, MonthEnd, DayEnd =   2020, 3, 8 # Day of NMDA end
                    # ## HIGH PERFORMERS
          if FlagforSessions=='LGI1':
            YearStart, MonthStart, DayStart = 2020, 6, 12 #15 days before LGI1
            YearEnd, MonthEnd, DayEnd = 2020, 7, 17 #Day After LGI1
      elif '9' in file:
          bb=1
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2021, 1, 1 # This is the real original start
              YearEnd, MonthEnd, DayEnd = 2021, 7, 12 # This is the real original end
            ## HIGH PERFORMERS
          if FlagforSessions=='NMDA':
              YearStart, MonthStart, DayStart = 2021, 3, 26# 2021, 4, 12
              YearEnd, MonthEnd, DayEnd = 2021, 5, 19 #2021, 6, 2

          if FlagforSessions=='LGI1':
              YearStart, MonthStart, DayStart = 2021, 5, 24
              YearEnd, MonthEnd, DayEnd = 2021, 6, 24
      elif '11' in file:
          bb=2
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2022, 1, 13# This is the real original start
              YearEnd, MonthEnd, DayEnd = 2022, 12, 31 # This is the real original end
          # ## HIGH PERFORMERS
          if FlagforSessions=='NMDA':
              YearStart, MonthStart, DayStart = 2022, 4, 8
              YearEnd, MonthEnd, DayEnd = 2022, 5, 5

          if FlagforSessions=='LGI1':
              YearStart, MonthStart, DayStart = 2022, 4, 8
              YearEnd, MonthEnd, DayEnd = 2022, 4, 25
      elif '12' in file:
          bb=3
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2023, 10, 18
              YearEnd, MonthEnd, DayEnd = 2024, 3, 31

      elif '14' in file:
          bb=4
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2024, 5, 27
              YearEnd, MonthEnd, DayEnd = 2024, 12, 31
      
      elif '15' in file:
          bb=1
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2024, 12, 1
              YearEnd, MonthEnd, DayEnd = 2024, 12, 31

      YearStart_str = str(YearStart)
      MonthStart_str = str(MonthStart).zfill(2)
      DayStart_str = str(DayStart).zfill(2)
      YearEnd_str = str(YearEnd)
      MonthEnd_str = str(MonthEnd).zfill(2)
      DayEnd_str = str(DayEnd).zfill(2)

      start_date = np.datetime64(f'{YearStart_str}-{MonthStart_str}-{DayStart_str}')
      end_date = np.datetime64(f'{YearEnd_str}-{MonthEnd_str}-{DayEnd_str}')


      Nanimals = S[bb]['CorrectTimeLicksPerAnimalSessTrial'].shape[0]
      Nsess = S[bb]['CorrectTimeLicksPerAnimalSessTrial'].shape[1]


      anim_Perf=[]
      anim_PerfLast=[]
      anim_Perf1st=[]
      anim_MeanRT=[]
      anim_MeanlastRT=[]
      anim_Mean1stRT=[]
      anim_Incorrect=[]
      anim_Trials=[]
      anim_MeanIncorr=[]
      anim_MeanError=[]
      anim_FirstTrialWithWater=[]
      anim_CorrectPort=[]
      anim_Date=[]
      anim_MeanTimeToReachTZ=[]
      anim_Time1stTrialWWater=[]
      anim_CorrectTimeLicks=[]

      ExpDay_reshape = S[bb]['ExpDay'].reshape(-1, S[bb]['ExpDay'].shape[-1])
      #Find all the dates the experiment was run
      ExpDates = np.unique(ExpDay_reshape, axis=0)[1:] #Avoiding the 1st array since its the [0,0,0] default from MATLAB code when the animals did not run the task.

      for aa in range(Nanimals):
          # print('batch:', bb, 'anim: ', aa)
          AveragePerf=[]
          AveragePerfLast=[]
          AveragePerf1st=[]
          MeanlastRT=[]
          Mean1stRT=[]
          MeanRT=[]
          IncorrPorts=[]
          Trials=[]
          MeanIncorr=[]
          MeanError=[]
          MeanTimeToReachTZ=[]
          Time1stTrialWWater=[]
          FirstTrialWithWater=[]
          CorrectPort=[]
          Date=[]
          CorrectTimeLicks=[]

          sess=0
          for count in range(Nsess):
            if bb==0 or bb==1 or bb==2: 
              if np.array_equal(ExpDates[count] , S[bb]['ExpDay'][aa][sess]):
                #if not any(np.array_equal(S[bb]['ExpDay'][aa][sess], monday) for monday in mondays_date_array12):
                # Initialize session containers
                if not np.shape(S[bb]['CueTimes'][aa][sess])==(1,0):
                    vec_day=S[bb]['ExpDay'][aa, sess, :].squeeze()
                    VecDay = np.datetime64(f'{vec_day[0]:04d}-{vec_day[1]:02d}-{vec_day[2]:02d}', 'D')
                    if start_date <= VecDay <= end_date:
                      if MinNumTrials <= S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].shape[1] and S[bb]['ExpParams'][aa, sess] is not None:
                        if StaMin <= S[bb]['ExpParams'][aa, sess][0][0] <= StaMax and any((S[bb]['ExpParams'][aa,sess][0][1] - DrugType) == 0):
                        #   print('Batch: ', bb,'Animal: ', aa,  'Sess: ', sess, 'DrugType: ', DrugType)
                          Date.append(S[bb]['ExpDay'][aa, sess, :].squeeze())
                          CorrectPort.append(S[bb]['CorrectPort'][aa,sess])
                          mask_nan=np.isnan(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze())  ##nan mask (tells me which of those are NaN values)
                          AverPrf=[len(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()[~mask_nan])/len(S[bb]['CueTimes'][aa, sess].squeeze())*100]  ##this is the final performance
                          AveragePerf.append(AverPrf[0])
                          num_trials = len(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze())
                          # Calculate the Nº of trials that are 30%
                          num_last_trials = int(np.ceil(num_trials * 0.3))
                          # print('#trials: ',num_trials, ' Num_last_trials: ', num_last_trials)
                          #obtain the last 30% of trials
                          last_trials = S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()[-num_last_trials:]
                          # Create the NaN values Mask
                          mask = np.isnan(last_trials)
                          # Calculate the performance for the last 30% of the trials
                          PerfLast = [(len(last_trials[~mask]) / num_last_trials)* 100
                                      if num_trials >= num_last_trials else 0]
                          AveragePerfLast.append(PerfLast[0])

                          CorrectTimeLicks.append(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze())

                          #obtain the 1st 30% of trials
                          first_trials = S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()[:num_last_trials]
                          # Create the NaN values Mask
                          mask_first = np.isnan(first_trials)
                          # Calculate the 1st 30%
                          AveragePerf1st.append([(len(first_trials[~mask_first]) / num_last_trials) * 100
                                      if num_trials >= num_last_trials else 0][0])
                          # AveragePerf1st.append(Perf1st[0])
                          # mask=np.isnan(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()[-5:])
                          # PerfLast =[(len(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()[-5:][~mask])/len(S[bb]['CueTimes'][aa, sess].squeeze()[-5:])*100) if len(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze())>=5 else AverPrf[0]]
                          MeanlastRT.append([np.nanmean(session_data[-num_last_trials:]) if len(session_data) >= num_last_trials else np.nanmean(session_data) for session_data in S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess]][0])
                          # MeanlastRT.append(mean_values[0])
                          Mean1stRT.append([np.nanmean(session_data[:num_last_trials]) if len(session_data) >= num_last_trials else np.nanmean(session_data) for session_data in S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess]][0])
                          # Mean1stRT.append(mean_values_1st[0])
                          MeanRT.append(np.nanmean(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].squeeze()))  ##Mean of the Reaction Time towards the Correct Port
                          Trials.append(len(S[bb]['CueTimes'][aa, sess].squeeze()))
                          IncorrPorts.append(S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess].squeeze())
                          INC= [np.shape(trl)[1] for trl in S[bb]['IncorrectLicksPerAnimalSessTrial'][aa, sess].squeeze()]
                          MeanIncorr.append(np.nanmean(INC))
                          MeanError_sess=np.mean([np.shape(trl)[1] for trl in S[bb]['ErrorLickPortIndPerAnimalSessTrial'][aa,sess][0]])
                          MeanError.append(MeanError_sess)
                          Time1stTrialWWater.append(S[bb]['TwS'][aa,sess]- S[bb]['ExpStartTime'][aa,sess])
                          water=[idx for idx, val in enumerate((S[bb]['CueTimes'][aa,sess]-S[bb]['TwS'][aa,sess])[0]) if val>0]
                          FirstTrialWithWater.append(water[0]+1 if np.size(water)>0 else len(S[bb]['CueTimes'][aa,sess][0]))
                          if S[bb]['ExpParams'][aa, sess][0][0] ==1:
                            arr=np.copy(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa,sess][0])
                            arr[np.isnan(arr)] = 120
                            time=np.roll(S[bb]['CueTimes'][aa, sess][0]-np.roll(S[bb]['CueTimes'][aa, sess][0],1),-1)-arr
                            MeanTimeToReachTZ.append(np.mean(time[:-1]))

                          if S[bb]['ExpParams'][aa, sess][0][0] ==1.2:
                            arr=np.copy(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa,sess][0])
                            arr[np.isnan(arr)] = 10
                            time=np.roll(S[bb]['CueTimes'][aa, sess][0]-np.roll(S[bb]['CueTimes'][aa, sess][0],1),-1)-arr
                            MeanTimeToReachTZ.append(np.mean(time[:-1]))

                          if S[bb]['ExpParams'][aa, sess][0][0] ==2.2:
                            arr=np.copy(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa,sess][0])
                            arr[np.isnan(arr)] = 10
                            time=np.roll(S[bb]['CueTimes'][aa, sess][0]-np.roll(S[bb]['CueTimes'][aa, sess][0],1),-1)-arr
                            MeanTimeToReachTZ.append(np.mean(time[:-1]))

                          if S[bb]['ExpParams'][aa, sess][0][0] ==3:
                            arr=np.copy(S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa,sess][0])
                            arr[np.isnan(arr)] = 4
                            time=np.roll(S[bb]['CueTimes'][aa, sess][0]-np.roll(S[bb]['CueTimes'][aa, sess][0],1),-1)-arr
                            MeanTimeToReachTZ.append(np.mean(time[:-1]))


                        else:
                          AveragePerf.append(np.nan)
                          CorrectPort.append(np.nan)
                          AveragePerfLast.append(np.nan)
                          AveragePerf1st.append(np.nan)
                          Mean1stRT.append(np.nan)
                          MeanRT.append(np.nan)
                          MeanlastRT.append(np.nan)
                          Trials.append(np.nan)
                          IncorrPorts.append(np.nan)
                          MeanIncorr.append(np.nan)
                          Date.append(np.nan)
                          MeanError.append(np.nan)
                          MeanTimeToReachTZ.append(np.nan)
                          Time1stTrialWWater.append(np.nan)
                          FirstTrialWithWater.append(np.nan)
                          CorrectTimeLicks.append(np.nan)

                      else:
                        AveragePerf.append(np.nan)
                        CorrectPort.append(np.nan)
                        AveragePerfLast.append(np.nan)
                        AveragePerf1st.append(np.nan)
                        Mean1stRT.append(np.nan)
                        MeanRT.append(np.nan)
                        MeanlastRT.append(np.nan)
                        Trials.append(np.nan)
                        IncorrPorts.append(np.nan)
                        MeanIncorr.append(np.nan)
                        Date.append(np.nan)
                        MeanError.append(np.nan)
                        MeanTimeToReachTZ.append(np.nan)
                        Time1stTrialWWater.append(np.nan)
                        CorrectTimeLicks.append(np.nan)
                    else:
                        AveragePerf.append(np.nan)
                        CorrectPort.append(np.nan)
                        AveragePerfLast.append(np.nan)
                        AveragePerf1st.append(np.nan)
                        Mean1stRT.append(np.nan)
                        MeanRT.append(np.nan)
                        MeanlastRT.append(np.nan)
                        Trials.append(np.nan)
                        IncorrPorts.append(np.nan)
                        MeanIncorr.append(np.nan)
                        Date.append(np.nan)
                        MeanError.append(np.nan)
                        MeanTimeToReachTZ.append(np.nan)
                        Time1stTrialWWater.append(np.nan)
                        CorrectTimeLicks.append(np.nan)
                else:
                  AveragePerf.append(np.nan)
                  CorrectPort.append(np.nan)
                  AveragePerfLast.append(np.nan)
                  AveragePerf1st.append(np.nan)
                  Mean1stRT.append(np.nan)
                  MeanRT.append(np.nan)
                  MeanlastRT.append(np.nan)
                  Trials.append(np.nan)
                  IncorrPorts.append(np.nan)
                  Date.append(np.nan)
                  MeanError.append(np.nan)
                  MeanIncorr.append(np.nan)
                  MeanTimeToReachTZ.append(np.nan)
                  Time1stTrialWWater.append(np.nan)
                  FirstTrialWithWater.append(np.nan)
                  CorrectTimeLicks.append(np.nan)
                sess=sess+1
            
              else:
                    AveragePerf.append(np.nan)
                    CorrectPort.append(np.nan)
                    AveragePerfLast.append(np.nan)
                    AveragePerf1st.append(np.nan)
                    Mean1stRT.append(np.nan)
                    MeanRT.append(np.nan)
                    MeanlastRT.append(np.nan)
                    Trials.append(np.nan)
                    IncorrPorts.append(np.nan)
                    Date.append(np.nan)
                    MeanError.append(np.nan)
                    MeanIncorr.append(np.nan)
                    MeanTimeToReachTZ.append(np.nan)
                    Time1stTrialWWater.append(np.nan)
                    FirstTrialWithWater.append(np.nan)
                    CorrectTimeLicks.append(np.nan)
                    sess=sess
                  # print(sess)

            else:
               continue
            #   AveragePerf.append(np.nan)
            #   AveragePerf1st.append(np.nan)
            #   AveragePerfLast.append(np.nan)
            #   MeanTimeToReachTZ.append(np.nan)
            #   MeanRT.append(np.nan)
            #   MeanlastRT.append(np.nan)
            #   Mean1stRT.append(np.nan)
            #   MeanError.append(np.nan)
            #   Trials.append(np.nan)
            #   IncorrPorts.append(np.nan)
            #   MeanIncorr.append(np.nan)
            #   Date.append(np.nan)
            #   FirstTrialWithWater.append(np.nan)
            #   CorrectTimeLicks.append(np.nan)
            # sess=sess+1

          ### ANIMALS VARIABLES APPENDING #####
          if AveragePerf:
              anim_Perf.append(AveragePerf)
              anim_PerfLast.append(AveragePerfLast)
              anim_Perf1st.append(AveragePerf1st)
              anim_MeanRT.append(MeanRT)
              anim_Mean1stRT.append(Mean1stRT)
              anim_MeanlastRT.append(MeanlastRT)
              anim_Trials.append(Trials)
              anim_Incorrect.append(IncorrPorts)
              anim_MeanIncorr.append(MeanIncorr)
              anim_CorrectPort.append(CorrectPort)
              anim_Date.append(Date)
              anim_MeanError.append(MeanError)
              anim_MeanTimeToReachTZ.append(MeanTimeToReachTZ)
              anim_Time1stTrialWWater.append(Time1stTrialWWater)
              anim_FirstTrialWithWater.append(FirstTrialWithWater)
              anim_CorrectTimeLicks.append(CorrectTimeLicks)
          else:
              anim_Perf.append(np.nan)
              anim_PerfLast.append(np.nan)
              anim_Perf1st.append(np.nan)
              anim_MeanRT.append(np.nan)
              anim_Mean1stRT.append(np.nan)
              anim_MeanlastRT.append(np.nan)
              anim_Trials.append(np.nan)
              anim_Incorrect.append(np.nan)
              anim_MeanIncorr.append(np.nan)
              anim_CorrectPort.append(np.nan)
              anim_Date.append(np.nan)
              anim_MeanError.append(np.nan)
              anim_MeanTimeToReachTZ.append(np.nan)
              anim_Time1stTrialWWater.append(np.nan)
              anim_FirstTrialWithWater.append(np.nan)
              anim_CorrectTimeLicks.append(np.nan)

      bb_Perf.append(anim_Perf)
      bb_PerfLast.append(anim_PerfLast)
      bb_Perf1st.append(anim_Perf1st)
      bb_MeanRT.append(anim_MeanRT)
      bb_Mean1stRT.append(anim_Mean1stRT)
      bb_MeanlastRT.append(anim_MeanlastRT)
      bb_Trials.append(anim_Trials)
      bb_Incorrect.append(anim_Incorrect)
      bb_MeanIncorr.append(anim_MeanIncorr)
      bb_CorrectPort.append(anim_CorrectPort)
      bb_Date.append(anim_Date)
      bb_MeanError.append(anim_MeanError)
      bb_MeanTimeToReachTZ.append(anim_MeanTimeToReachTZ)
      bb_Time1stTrialWWater.append(anim_Time1stTrialWWater)
      bb_FirstTrialWithWater.append(anim_FirstTrialWithWater)
      bb_CorrectTimeLicks.append(anim_CorrectTimeLicks)

    ALL_PerfLast[FlagforSessions]=bb_PerfLast
    ALL_Perf[FlagforSessions]=bb_Perf
    ALL_Trials[FlagforSessions]=bb_Trials
    ALL_MeanlastRT[FlagforSessions]=bb_MeanlastRT
    ALL_Mean1stRT[FlagforSessions]=bb_Mean1stRT
    ALL_Time1stTrialWWater[FlagforSessions]=bb_Time1stTrialWWater
    ALL_Perf1st[FlagforSessions]=bb_Perf1st
    ALL_FirstTrialWithWater[FlagforSessions]=bb_FirstTrialWithWater
    ALL_MeanRT[FlagforSessions]=bb_MeanRT
    ALL_Inc[FlagforSessions]=bb_Incorrect
    ALL_CorrectTimeLicks[FlagforSessions]=bb_CorrectTimeLicks
    ALL_MeanIncorr[FlagforSessions]=bb_MeanIncorr
    ALL_MeanError[FlagforSessions]=bb_MeanError
    ALL_MeanTimeToReachTZ[FlagforSessions]=bb_MeanTimeToReachTZ
    ALL_CorrectPort[FlagforSessions]=bb_CorrectPort
    ALL_Date=bb_Date

    return ALL_MeanRT, ALL_PerfLast, ALL_MeanlastRT ,ALL_Perf, ALL_Trials, ALL_Inc, ALL_MeanIncorr, ALL_CorrectPort, ALL_Date, ALL_MeanError, ALL_MeanTimeToReachTZ, ALL_Time1stTrialWWater,  ALL_Perf1st ,ALL_Mean1stRT, ALL_FirstTrialWithWater, ALL_CorrectTimeLicks

def CleaningRawDataset_TRAINING_TsA(DrugType='CONTROL', FlagforSessions='CONTROL'):
  pathData= 'C:\\Users\\User\\Documents\\cajal\\code\\data\\' #'/content/gdrive/MyDrive/Data8port/'
  # pathData='C:\\Users\\pau_1\\Documents\\DATA\\MaratoProjectData\\'
  fileName1='Output_8PortMazeAnalAVerRecallAllAnimals8NBatch.mat'
  fileName2='Output_8PortMazeAnalAVerRecallAllAnimals9NBatch.mat'
  fileName3='Output_8PortMazeAnalAVerRecallAllAnimals11NBatch.mat'
  fileName4='Output_8PortMazeAnalAVerRecallAllAnimals12Batch.mat'
  fileName5='Output_8PortMazeAnalAVerRecallAllAnimals14Batch.mat'
  # fileName6='Output_8PortMazeAnalAVerRecallAllAnimals15Batch.mat'
  # Load data from files
  S = [scipy.io.loadmat(pathData + fileName) for fileName in [fileName1, fileName2, fileName3, fileName4, fileName5]]#, fileName6]]
  # Parameters
  StaMin = 3
  StaMax = 3
  MinNumTrials = 1

  if DrugType=='CONTROL':
    DrugType=[0, 1, 3, 5, 8, 9, 7, 14, 15, 16] ##CONTROL or non infused
  if DrugType=='NMDA':
    DrugType=[6, 10]  ## anti_NMDAR infused animals
  if DrugType=='LGI1':
    DrugType=[11]   ## anti_LGI1 infused animals


  X = []
  ALL_FirstTrlWWater=[]
  ALL_FirstTrlWaterAvai=[]
  ALL_portsPoked = []
  ALL_OnlyCorrectPortsPoked = []
  ALL_IntTrialportsPoked = []
  ALL_PORTS=[]
  ALL_PORTS_YES=[]
  WATER = []
  CP = []

  # Loop over batches
  for bb in range(0,3):
      if bb == 0:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2019, 10, 21 # This is the real original start
          YearEnd, MonthEnd, DayEnd = 2020, 12, 31 # This is the real original end
          # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2020, 1, 11 #24 days before the NMDA infusion period
          YearEnd, MonthEnd, DayEnd =   2020, 3, 8 # Day of NMDA end
                  # ## HIGH PERFORMERS
        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2020, 6, 12 #15 days before LGI1
          YearEnd, MonthEnd, DayEnd = 2020, 7, 17 #Day After LGI1

      elif bb == 1:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2021, 1, 1 # This is the real original start
          YearEnd, MonthEnd, DayEnd = 2021, 7, 12 # This is the real original end
          ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2021, 3, 26# 2021, 4, 12
          YearEnd, MonthEnd, DayEnd = 2021, 5, 19 #2021, 6, 2

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2021, 5, 24
          YearEnd, MonthEnd, DayEnd = 2021, 6, 24

      elif bb == 2:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2022, 1, 13# This is the real original start
          YearEnd, MonthEnd, DayEnd = 2022, 12, 31 # This is the real original end
        # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 5, 5

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 4, 25

      elif bb == 3:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2023, 10, 18# This is the real original start
          YearEnd, MonthEnd, DayEnd = 2024, 5, 1 # This is the real original end

      elif bb==4:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2024, 5, 27
          YearEnd, MonthEnd, DayEnd = 2024, 12, 31

      elif bb==5:
          if FlagforSessions=='CONTROL':
              YearStart, MonthStart, DayStart = 2024, 12, 1
              YearEnd, MonthEnd, DayEnd = 2024, 12, 31



      YearStart_str = str(YearStart)
      MonthStart_str = str(MonthStart).zfill(2)
      DayStart_str = str(DayStart).zfill(2)
      YearEnd_str = str(YearEnd)
      MonthEnd_str = str(MonthEnd).zfill(2)
      DayEnd_str = str(DayEnd).zfill(2)

      start_date = np.datetime64(f'{YearStart_str}-{MonthStart_str}-{DayStart_str}')
      end_date = np.datetime64(f'{YearEnd_str}-{MonthEnd_str}-{DayEnd_str}')

      Nanimals = S[bb]['CorrectPort'].shape[0]
      Nsess = S[bb]['CorrectPort'].shape[1]

      anim_portsPoked = np.full((Nanimals,157,56) ,np.nan, dtype=object)
      anim_FirstTrlWWater=np.full((Nanimals,157),np.nan, dtype=object)
      anim_FirstTrlWaterAvai=np.full((Nanimals, 157),np.nan, dtype=object)
      anim_OnlyCorrectPortsPoked = []
      anim_IntTrialportsPoked = []
      anim_PORTS =np.full((Nanimals,157), 0)
      anim_PORTS_yes =np.full((Nanimals,157), 0)
      anim_WATER = []
      
      ExpDay_reshape = S[bb]['ExpDay'].reshape(-1, S[bb]['ExpDay'].shape[-1])
        #Find all the dates the experiment was run
      ExpDates = np.unique(ExpDay_reshape, axis=0)[1:]
    
      for aa in range(Nanimals):
          # print('aa: ', aa)
          sess_portsPoked = np.full((157,56) ,np.nan, dtype=object)

          sess_OnlyCorrectPortsPoked = []
          sess_IntTrialportsPoked = []
          sess_PORTS = np.full(157 ,0, dtype=object)
          sess_PORTS_yes = np.full(157 ,0, dtype=object)
          sess_WATER = []
          FirstTrlWWater=np.full(157,np.nan, dtype=object)
          FirstTrlWaterAvai=np.full(157,np.nan, dtype=object)
          FirstTrlWWater=np.full(157,np.nan, dtype=object)
          FirstTrlWaterAvai=np.full(157,np.nan, dtype=object)
          # if bb==1:
          #   Nsess=142
          #   waterAvailable=S[bb]['TaS'][aa]-S[bb]['ExpStartTime'][aa][:-1]
          #   cues=S[bb]['CueTimes'][aa][:-1]-S[bb]['ExpStartTime'][aa][:-1]
          #   correctTimes=S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa]+S[bb]['CueTimes'][aa][:-1]-S[bb]['ExpStartTime'][aa][:-1]
          #   incorrectTimes=S[bb]['IncorrectLicksPerAnimalSessTrial'][aa]+S[bb]['CueTimes'][aa][:-1]-S[bb]['ExpStartTime'][aa][:-1]
          # else: 
          #   waterAvailable=S[bb]['TaS'][aa]-S[bb]['ExpStartTime'][aa]
          #   cues=S[bb]['CueTimes'][aa]-S[bb]['ExpStartTime'][aa]
          #   correctTimes=S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa]+S[bb]['CueTimes'][aa]-S[bb]['ExpStartTime'][aa]
          #   incorrectTimes=S[bb]['IncorrectLicksPerAnimalSessTrial'][aa]+S[bb]['CueTimes'][aa]-S[bb]['ExpStartTime'][aa]
          # incorrectPorts=S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa]
          # correctPorts=S[bb]['CorrectPort'][aa]
          if bb==1:
            Nsess=142
        
          sess=0
          for count in range(Nsess):
              # Initialize session containers
              waterAvail = []
              portsPoked = np.full(56, np.nan,dtype=object)
              OnlyCorrectPortsPoked = []
              IntTrialportsPoked = []
              k = 0
              if np.array_equal(ExpDates[count] , S[bb]['ExpDay'][aa][sess]):
                  waterAvailable=S[bb]['TaS'][aa][sess]-S[bb]['ExpStartTime'][aa][sess]
                  cues=S[bb]['CueTimes'][aa][sess]-S[bb]['ExpStartTime'][aa][sess]
                  correctTimes=S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa][sess]+S[bb]['CueTimes'][aa][sess]-S[bb]['ExpStartTime'][aa][sess]
                  incorrectTimes=S[bb]['IncorrectLicksPerAnimalSessTrial'][aa][sess]+S[bb]['CueTimes'][aa][sess]-S[bb]['ExpStartTime'][aa][sess]
                  incorrectPorts=S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa][sess]
                  correctPorts=S[bb]['CorrectPort'][aa][sess]
                  
                  if not np.shape(S[bb]['WaterAvailabilityPerAnimalSess'][aa][sess]) == (1, 0):
                      vec_day = S[bb]['ExpDay'][aa, sess, :].squeeze()
                      VecDay = np.datetime64(f'{vec_day[0]:04d}-{vec_day[1]:02d}-{vec_day[2]:02d}', 'D')
                      IN = 0
                      if start_date <= VecDay <= end_date:
                         IN = 1

                  if S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess][0].all():
                      if not S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess][0].all():
                          IN = 0
                      if MinNumTrials <= S[bb]['CorrectTimeLicksPerAnimalSessTrial'][aa, sess].shape[1] and \
                              S[bb]['ExpParams'][aa, sess] is not None:
                          if StaMin <= S[bb]['ExpParams'][aa, sess][0][0] <= StaMax and IN == 1 and \
                                  any((S[bb]['ExpParams'][aa, sess][0][1] - DrugType) == 0):
                            sess_PORTS[sess]=np.array(S[bb]['CorrectPort'][aa, sess])
                            sess_PORTS_yes[sess]=np.array(S[bb]['CorrectPort'][aa, sess-1])
                            nw = 1
                            for i in range(len(S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess][0])):
                                if S[bb]['IncorrPortLickedPerAnimalSessTrial'][aa, sess][0][i][0].all():
                                    nw += 1
                            if cues.flatten()[0]>waterAvailable:
                                if not np.isnan(incorrectTimes.flatten()[0]).all() and not np.isnan(correctTimes.flatten()[0]):
                                    diccionario={}
                                    for i,poke in enumerate(incorrectPorts.flatten()[0][0]):
                                        diccionario[poke]=incorrectTimes.flatten()[0][0][i]
                                        diccionario[correctPorts]=correctTimes.flatten()[0]
                                        times=np.array([value for _, value in sorted(diccionario.items(), key=lambda item: item[1])])
                                        pokes = np.array([key for key, _ in sorted(diccionario.items(), key=lambda item: item[1])])
                                        portsPoked[0]=pokes[0]
                                if not np.isnan(incorrectTimes.flatten()[0]).all() and  np.isnan(correctTimes.flatten()[0]):
                                        portsPoked[0]=incorrectPorts.flatten()[0][0][0]
                                if np.isnan(incorrectTimes.flatten()[0]).all() and not np.isnan(correctTimes.flatten()[0]):
                                        portsPoked[0]=correctPorts
                                
                            else:
                                for trial in range(len(cues.flatten()[waterAvailable>cues.flatten()])):
                                    if trial==len(cues.flatten()[waterAvailable>cues.flatten()])-1:
                                        if not np.isnan(incorrectTimes.flatten()[trial]).all() and not np.isnan(correctTimes.flatten()[trial]):
                                            diccionario={}
                                            for i,poke in enumerate(incorrectPorts.flatten()[trial][0]):
                                                diccionario[poke]=incorrectTimes.flatten()[trial][0][i]
                                            diccionario[correctPorts]=correctTimes.flatten()[trial]
                                            times=np.array([value for _, value in sorted(diccionario.items(), key=lambda item: item[1])])
                                            pokes = np.array([key for key, _ in sorted(diccionario.items(), key=lambda item: item[1])])
                                            portsPoked[trial]=pokes[times<waterAvailable]
                                        if not np.isnan(incorrectTimes.flatten()[trial]).all() and np.isnan(correctTimes.flatten()[trial]):
                                            pokes=incorrectPorts.flatten()[trial][0]
                                            times=incorrectTimes.flatten()[trial][0]
                                            portsPoked[trial]=pokes[times<waterAvailable]
                                        if  np.isnan(incorrectTimes.flatten()[trial]).all() and not np.isnan(correctTimes.flatten()[trial]):
                                            pokes=correctPorts
                                            times=correctTimes.flatten()[trial]
                                            portsPoked[trial]=pokes[times<waterAvailable]
                                    else:
                                        if not np.isnan(incorrectTimes.flatten()[trial]).all() and not np.isnan(correctTimes.flatten()[trial]):
                                            portsPoked[trial]=np.concatenate([incorrectPorts.flatten()[trial][0],correctPorts.flatten()])
                                        if not np.isnan(incorrectTimes.flatten()[trial]).all() and np.isnan(correctTimes.flatten()[trial]):
                                            portsPoked[trial]=incorrectPorts.flatten()[trial][0]
                                        if np.isnan(incorrectTimes.flatten()[trial]).all() and not np.isnan(correctTimes.flatten()[trial]):
                                            portsPoked[trial]=correctPorts.flatten().astype(int)
                                        # ANIMALS VARIABLES APPENDING
                            sess_portsPoked[count]=np.array(portsPoked)
                            sess_OnlyCorrectPortsPoked.append(np.array(OnlyCorrectPortsPoked))
                            sess_IntTrialportsPoked.append(IntTrialportsPoked)
                            # sess_PORTS[sess]=np.array(S[bb]['CorrectPortPOSTG'][aa, sess])      
                  sess=sess+1

              else:
                sess=sess

          # Session VARIABLES APPENDING
          anim_portsPoked[aa]=np.array(sess_portsPoked)
          anim_FirstTrlWWater[aa]=np.array(FirstTrlWWater)
          anim_FirstTrlWaterAvai[aa]=np.array(FirstTrlWaterAvai)
          anim_OnlyCorrectPortsPoked.append(sess_OnlyCorrectPortsPoked)
          anim_IntTrialportsPoked.append(sess_IntTrialportsPoked)
          anim_PORTS[aa]=np.array(sess_PORTS)
          anim_PORTS_yes[aa]=np.array(sess_PORTS_yes)

      # ALL VARIABLES APPENDING
      ALL_portsPoked.extend(anim_portsPoked)
      ALL_PORTS.extend(np.array(anim_PORTS))
      ALL_PORTS_YES.extend(np.array(anim_PORTS_yes))
  return ALL_portsPoked, ALL_PORTS, ALL_PORTS_YES

def CleaningRawDataset_REC_TsA(DrugType='CONTROL', FlagforSessions='CONTROL'):
  path1 = Path(r'Z:\Raw_Data_8PortsMaze')
  path2 = Path(r'C:\Users\User\Documents\cajal\code\data')
    
    # Select the first path that exists
  pathData = path1 if path1.exists() else path2
  fileName1='Output_8PortMazeAnalAVerRecallAllAnimals8Batch.mat'
  fileName2='Output_8PortMazeAnalAVerRecallAllAnimals9Batch.mat'
  fileName3='Output_8PortMazeAnalAVerRecallAllAnimals11Batch.mat'

  # Load data from files
  S = [scipy.io.loadmat(pathData / fileName) for fileName in [fileName1, fileName2, fileName3]]
  AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}

  # Parameters
  StaMin = 3
  StaMax = 3
  MinNumTrials = 1

  if DrugType=='CONTROL':
    DrugType=[0, 1, 3, 5, 8, 9, 7, 14, 15, 16] ##CONTROL or non infused
  if DrugType=='NMDA':
    DrugType=[6, 10]  ## anti_NMDAR infused animals
  if DrugType=='LGI1':
    DrugType=[11]   ## anti_LGI1 infused animals


  X = []
  ALL_FirstTrlWWater_REC=[]
  ALL_FirstTrlWaterAvai_REC=[]
  ALL_portsPoked_REC = []
  ALL_avgportsPoked_REC = []
  ALL_OnlyCorrectPortsPoked_REC = []
  ALL_IntTrialportsPoked_REC = []
  ALL_PORTS_REC =[]
  ALL_PORTS_YES_REC=[]
  WATER = []
  CP = []

  # Loop over batches
  for bb in range(0,3):
      if bb == 0:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2019, 10, 21 # This is the real original start
          YearEnd, MonthEnd, DayEnd = 2020, 12, 31 # This is the real original end
          # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2020, 1, 11 #24 days before the NMDA infusion period
          YearEnd, MonthEnd, DayEnd =   2020, 3, 8 # Day of NMDA end
                  # ## HIGH PERFORMERS
        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2020, 6, 12 #15 days before LGI1
          YearEnd, MonthEnd, DayEnd = 2020, 7, 17 #Day After LGI1

      elif bb == 1:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2021, 1, 1 # This is the real original start
          YearEnd, MonthEnd, DayEnd = 2021, 7, 12 # This is the real original end
          ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2021, 3, 26# 2021, 4, 12
          YearEnd, MonthEnd, DayEnd = 2021, 5, 19 #2021, 6, 2

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2021, 5, 24
          YearEnd, MonthEnd, DayEnd = 2021, 6, 24


      elif bb == 2:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2022, 1, 13# This is the real original start
          YearEnd, MonthEnd, DayEnd = 2022, 12, 31 # This is the real original end
        # ## HIGH PERFORMERS
        if FlagforSessions=='NMDA':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 5, 5

        if FlagforSessions=='LGI1':
          YearStart, MonthStart, DayStart = 2022, 4, 8
          YearEnd, MonthEnd, DayEnd = 2022, 4, 25

      elif bb == 3:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2023, 10, 18# This is the real original start
          YearEnd, MonthEnd, DayEnd = 2024, 5, 1 # This is the real original end

      elif bb==4:
        if FlagforSessions=='CONTROL':
          YearStart, MonthStart, DayStart = 2024, 5, 27
          YearEnd, MonthEnd, DayEnd = 2024, 12, 31



      YearStart_str = str(YearStart)
      MonthStart_str = str(MonthStart).zfill(2)
      DayStart_str = str(DayStart).zfill(2)
      YearEnd_str = str(YearEnd)
      MonthEnd_str = str(MonthEnd).zfill(2)
      DayEnd_str = str(DayEnd).zfill(2)

      start_date = np.datetime64(f'{YearStart_str}-{MonthStart_str}-{DayStart_str}')
      end_date = np.datetime64(f'{YearEnd_str}-{MonthEnd_str}-{DayEnd_str}')

      Nanimals = S[bb]['CorrectPortPOSTG'].shape[0]
      Nsess = S[bb]['CorrectPortPOSTG'].shape[1]

      anim_portsPoked = np.full((Nanimals,157,56) ,np.nan, dtype=object)
      anim_avgportsPoked = np.full((Nanimals,157,56) ,np.nan, dtype=object)
      anim_FirstTrlWWater=np.full((Nanimals,157),np.nan, dtype=object)
      anim_FirstTrlWaterAvai=np.full((Nanimals, 157),np.nan, dtype=object)
      anim_OnlyCorrectPortsPoked = []
      anim_IntTrialportsPoked = []
      anim_PORTS =np.full((Nanimals,157), 0)
      anim_PORTS_yes =np.full((Nanimals,157), 0)
      anim_WATER = []
      
      ExpDay_reshape = S[bb]['ExpDay'].reshape(-1, S[bb]['ExpDay'].shape[-1])
        #Find all the dates the experiment was run
      ExpDates = np.unique(ExpDay_reshape, axis=0)[1:]
    
      for aa in range(Nanimals):
          sess_portsPoked = np.full((157,56) ,np.nan, dtype=object)
          sess_avgportsPoked=np.full((157,56) ,np.nan, dtype=object)
          sess_OnlyCorrectPortsPoked = []
          sess_IntTrialportsPoked = []
          sess_PORTS = np.full(157 ,0, dtype=object)
          sess_PORTS_yes = np.full(157 ,0, dtype=object)
          
          FirstTrlWWater=np.full(157,np.nan, dtype=object)
          FirstTrlWaterAvai=np.full(157,np.nan, dtype=object)
  
          waterAvailable=S[bb]['POSTaS'][aa]-S[bb]['POSTExpStartTime'][aa]
          cues=S[bb]['POSTCueTimes'][aa]-S[bb]['POSTExpStartTime'][aa]
          correctTimes=S[bb]['AllAnimalsSessTrialsPOSTCorrectTrialRT'][aa]+S[bb]['POSTCueTimes'][aa]-S[bb]['POSTExpStartTime'][aa]
          incorrectTimes=S[bb]['AllAnimalsSessTrialsPOSTIncorrectLicks'][aa]+S[bb]['POSTCueTimes'][aa]-S[bb]['POSTExpStartTime'][aa]
          incorrectPorts=S[bb]['AllAnimalsSessTrialsPOSTIncorrPortLicked'][aa]
          correctPorts=S[bb]['CorrectPortPOSTG'][aa]
          sess=0
          for count in range(Nsess):
              # print('animal: ', aa, 'Session: ', sess)
              # Initialize session containers
              avgportsPoked = np.full(56, np.nan,dtype=object)
              portsPoked = np.full(56, np.nan,dtype=object)
              OnlyCorrectPortsPoked = []
              IntTrialportsPoked = []
              
              k = 0
              if np.array_equal(ExpDates[count] , S[bb]['ExpDay'][aa][sess]):
                  if not np.shape(S[bb]['POSTWaterAvailabilityPerAnimalSess'][aa][sess]) == (1, 0):
                      vec_day = S[bb]['ExpDay'][aa, sess, :].squeeze()
                      VecDay = np.datetime64(f'{vec_day[0]:04d}-{vec_day[1]:02d}-{vec_day[2]:02d}', 'D')
                      IN = 0
                      if start_date <= VecDay <= end_date:
                         IN = 1

                  if S[bb]['AllAnimalsSessTrialsPOSTCorrectTrialRT'][aa, sess][0].all():
                      if not S[bb]['AllAnimalsSessTrialsPOSTCorrectTrialRT'][aa, sess][0].all():
                          IN = 0
                      if MinNumTrials <= S[bb]['AllAnimalsSessTrialsPOSTCorrectTrialRT'][aa, sess].shape[1] and \
                              S[bb]['ExpParams'][aa, sess] is not None:
                          if StaMin <= S[bb]['ExpParams'][aa, sess][0][0] <= StaMax and IN == 1 and \
                                  any((S[bb]['ExpParams'][aa, sess][0][1] - DrugType) == 0):
                            sess_PORTS[count]=np.array(S[bb]['CorrectPortPOSTG'][aa, sess])
                            sess_PORTS_yes[count]=np.array(S[bb]['CorrectPortPOSTG'][aa, sess-1])
                            nw = 1
                            for i in range(len(S[bb]['AllAnimalsSessTrialsPOSTIncorrPortLicked'][aa, sess][0])):
                                if S[bb]['AllAnimalsSessTrialsPOSTIncorrPortLicked'][aa, sess][0][i][0].all():
                                    nw += 1
                            if cues[sess].flatten()[0]>waterAvailable[sess]:
                                if not np.isnan(incorrectTimes[sess].flatten()[0]).all() and not np.isnan(correctTimes[sess].flatten()[0]):
                                    diccionario={}
                                    for i,poke in enumerate(incorrectPorts[sess].flatten()[0][0]):
                                        diccionario[poke]=incorrectTimes[sess].flatten()[0][0][i]
                                        diccionario[correctPorts[sess]]=correctTimes[sess].flatten()[0]
                                        times=np.array([value for _, value in sorted(diccionario.items(), key=lambda item: item[1])])
                                        pokes = np.array([key for key, _ in sorted(diccionario.items(), key=lambda item: item[1])])
                                        portsPoked[0]=np.array([pokes[0]])
                                        Poked= pokes[0]
                                        rad_pokes=AngRad_dict[Poked]
                                        avgportsPoked[0]=np.angle(np.exp(1j * rad_pokes).mean())
                                if not np.isnan(incorrectTimes[sess].flatten()[0]).all() and  np.isnan(correctTimes[sess].flatten()[0]):
                                        portsPoked[0]=np.array([incorrectPorts[sess].flatten()[0][0][0]])
                                        Poked= incorrectPorts[sess].flatten()[0][0][0]
                                        rad_pokes=AngRad_dict[Poked]
                                        avgportsPoked[0]=np.angle(np.exp(1j * rad_pokes).mean())
                                if np.isnan(incorrectTimes[sess].flatten()[0]).all() and not np.isnan(correctTimes[sess].flatten()[0]):
                                        portsPoked[0]=np.array([correctPorts[sess]])
                                        Poked= correctPorts[sess]
                                        rad_pokes=AngRad_dict[Poked]
                                        avgportsPoked[0]=np.angle(np.exp(1j * rad_pokes).mean())
  
                            else:
                                for trial in range(len(cues[sess].flatten()[waterAvailable[sess]>cues[sess].flatten()])):
                                    if trial==len(cues[sess].flatten()[waterAvailable[sess]>cues[sess].flatten()])-1:
                                        if not np.isnan(incorrectTimes[sess].flatten()[trial]).all() and not np.isnan(correctTimes[sess].flatten()[trial]):
                                            diccionario={}
                                            for i,poke in enumerate(incorrectPorts[sess].flatten()[trial][0]):
                                                diccionario[poke]=incorrectTimes[sess].flatten()[trial][0][i]
                                            diccionario[correctPorts[sess]]=correctTimes[sess].flatten()[trial]
                                            times=np.array([value for _, value in sorted(diccionario.items(), key=lambda item: item[1])])
                                            pokes = np.array([key for key, _ in sorted(diccionario.items(), key=lambda item: item[1])])
                                            portsPoked[trial]=pokes[times<waterAvailable[sess]]
                                        if not np.isnan(incorrectTimes[sess].flatten()[trial]).all() and np.isnan(correctTimes[sess].flatten()[trial]):
                                            pokes=incorrectPorts[sess].flatten()[trial][0]
                                            times=incorrectTimes[sess].flatten()[trial][0]
                                            portsPoked[trial]=pokes[times<waterAvailable[sess]]
                                            Poked= pokes[times<waterAvailable[sess]]
                                            rad_pokes=np.array([AngRad_dict[pokes] for pokes in Poked])
                                            avgportsPoked[trial]=np.angle(np.exp(1j * rad_pokes).mean())

                                        if  np.isnan(incorrectTimes[sess].flatten()[trial]).all() and not np.isnan(correctTimes[sess].flatten()[trial]):
                                            pokes=correctPorts[sess]
                                            times=correctTimes[sess].flatten()[trial]
                                            portsPoked[trial]=pokes[times<waterAvailable[sess]]
                                            Poked=pokes[times<waterAvailable[sess]]
                                            rad_pokes=np.array([AngRad_dict[pokes] for pokes in Poked])
                                            avgportsPoked[trial]=np.angle(np.exp(1j * rad_pokes).mean())

                                    else:
                                        if not np.isnan(incorrectTimes[sess].flatten()[trial]).all() and not np.isnan(correctTimes[sess].flatten()[trial]):
                                            portsPoked[trial]=np.concatenate([incorrectPorts[sess].flatten()[trial][0],correctPorts[sess].flatten()])
                                            Poked=np.concatenate([incorrectPorts[sess].flatten()[trial][0],correctPorts[sess].flatten()])
                                            rad_pokes=np.array([AngRad_dict[pokes] for pokes in Poked])
                                            avgportsPoked[trial]=np.angle(np.exp(1j * rad_pokes).mean())
                                        if not np.isnan(incorrectTimes[sess].flatten()[trial]).all() and np.isnan(correctTimes[sess].flatten()[trial]):
                                            portsPoked[trial]=incorrectPorts[sess].flatten()[trial][0]
                                            Poked=incorrectPorts[sess].flatten()[trial][0]
                                            rad_pokes=np.array([AngRad_dict[pokes] for pokes in Poked])
                                            avgportsPoked[trial]=np.angle(np.exp(1j * rad_pokes).mean())
                                        if np.isnan(incorrectTimes[sess].flatten()[trial]).all() and not np.isnan(correctTimes[sess].flatten()[trial]):
                                            portsPoked[trial]=correctPorts[sess].flatten().astype(int)
                                            Poked=correctPorts[sess].flatten().astype(int)
                                            rad_pokes=np.array([AngRad_dict[pokes] for pokes in Poked])
                                            avgportsPoked[trial]=np.angle(np.exp(1j * rad_pokes).mean())

                                        # ANIMALS VARIABLES APPENDING
                            sess_portsPoked[count]=np.array(portsPoked)#sess_portsPoked[sess]=np.array(portsPoked)
                            sess_avgportsPoked[count]=np.array(avgportsPoked)
                            sess_OnlyCorrectPortsPoked.append(np.array(OnlyCorrectPortsPoked))
                            sess_IntTrialportsPoked.append(IntTrialportsPoked)
                            FirstTrlWaterAvai[count]=len(cues[sess].flatten()[waterAvailable[sess]>cues[sess].flatten()])
                  sess=sess+1

              else:
                sess=sess

          # Session VARIABLES APPENDING
          anim_portsPoked[aa]=np.array(sess_portsPoked)
          anim_avgportsPoked[aa]=np.array(sess_avgportsPoked)
          anim_FirstTrlWWater[aa]=np.array(FirstTrlWWater)
          anim_FirstTrlWaterAvai[aa]=np.array(FirstTrlWaterAvai)
          anim_OnlyCorrectPortsPoked.append(sess_OnlyCorrectPortsPoked)
          anim_IntTrialportsPoked.append(sess_IntTrialportsPoked)
          anim_PORTS[aa]=np.array(sess_PORTS)
          anim_PORTS_yes[aa]=np.array(sess_PORTS_yes)

      # ALL VARIABLES APPENDING
      ALL_portsPoked_REC.extend(anim_portsPoked)
      ALL_PORTS_REC.extend(np.array(anim_PORTS))
      ALL_PORTS_YES_REC.extend(np.array(anim_PORTS_yes))
      ALL_avgportsPoked_REC.extend(anim_avgportsPoked)
      ALL_FirstTrlWaterAvai_REC.extend(anim_FirstTrlWaterAvai)
  return ALL_portsPoked_REC, ALL_avgportsPoked_REC,ALL_PORTS_REC, ALL_PORTS_YES_REC, ALL_FirstTrlWaterAvai_REC