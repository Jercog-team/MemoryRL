import random
import math
import numpy as np
from collections import Counter
from filtering_functions import circular_distance, hist_all

def MemoryIndex(hist_seq, port_seq, min_pokes):
  '''Computes the classical Memory Index: X projection of the vector strenght of the pokes relative to the correct port.
    
    Parameters: 
    hist_seq: numpy array of pokes of shape NeventsX 8 (8 possible ports) (Nevents: i.e.,N Sessions) - dtype=int
    port_seq: numpy array of the correct ports of all the events. 

    Ouput: 
    List of memory index per event (i.e; sess)
  '''
  cosAng = np.cos([ np.pi/4, 0, -np.pi/4, -np.pi/2, -3/4*np.pi, np.pi, 3/4*np.pi, np.pi/2])
  min_len=3
  MIsess=[]
  for ss in range(len(hist_seq)):
    if port_seq[ss] > 0:
      if np.nansum(hist_seq[ss])>min_pokes:
        Projected_hist_norm=(hist_seq[ss] / np.nansum(hist_seq[ss]))
        Projected_hist=np.roll(cosAng, int(port_seq[ss]) - 2)*Projected_hist_norm
        MIsess.append(np.nansum(Projected_hist))
      else:
        MIsess.append(np.nan)

    else:
      MIsess.append(np.nan)

  MIsess_arr=np.array(MIsess)
  return MIsess_arr

def MIsurrogatesLagsbyAnim(hist_seq, port_seq, yes_port_seq, sessFirst, sessLast, Nshuffles, lags, BigHist=False):
  """Inputs: the matrix of the histograms,  the list of ports"""
  arr_ports=port_seq[sessFirst:sessLast][port_seq[sessFirst:sessLast]!= 0] #Filter the values to only include the valid ones
  arr_pokes=hist_seq[sessFirst:sessLast][port_seq[sessFirst:sessLast]!= 0] #Filter the histogram sequence based on the port sequence 
  yes_arr_ports=yes_port_seq[sessFirst:sessLast][port_seq[sessFirst:sessLast]!= 0] #Filter the values to only include the valid ones
  # Nshuffles=np.min([maxnumberSurr(arr_ports), MaxNShuffles]) #Calculate the maximum number of permutations without repetition for this particular dataset
  Nsess=len(arr_ports) #Calculate the number of sessions are valid 
  print(Nshuffles)
  if BigHist==False:
    MI_Surr2h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 2h
    MI_Surr24h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 24h
    for i in range(Nshuffles):
        for lag in range(lags):
            if lag==0:
                random.shuffle(arr_ports)
                MI_Surr2h[i]=np.nanmean(MemoryIndex(arr_pokes, arr_ports))
            if lag==1:
                random.shuffle(yes_arr_ports)
                MI_Surr24h[i]=np.nanmean(MemoryIndex(arr_pokes, yes_arr_ports))
  else:
    MI_Surr2h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 2h
    MI_Surr24h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 24h
    for i in range(Nshuffles):
        for lag in range(lags):
            if lag==0:
                random.shuffle(arr_ports)
                big_hist=np.sum([np.roll(arr_pokes[ss],8-int(arr_ports[ss]), axis=0) for ss in range(len(arr_pokes))], axis=0)
                MI_Surr2h[i]=MemoryIndexbyTrl(big_hist, 8)
            if lag==1:
                random.shuffle(yes_arr_ports)
                big_hist24h=np.sum([np.roll(arr_pokes[ss],8-int(yes_arr_ports[ss]), axis=0) for ss in range(len(arr_pokes))], axis=0)
                MI_Surr24h[i]=MemoryIndexbyTrl(big_hist24h, 8)

  # MI_Surr_by_day=np.nanmean(MI_Surr, axis=0)
  return   MI_Surr2h, MI_Surr24h

def MIsurrogatesLagsbyDistance(hist_seq, port_seq, yes_port_seq, Nshuffles, lags):
    """Inputs: the matrix of the histograms,  the list of ports"""
    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}
    arr_ports=port_seq[port_seq>0]
    yes_arr_ports=yes_port_seq[port_seq>0]
    arr_pokes=hist_seq[port_seq>0]
    port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(arr_ports))
    yes_port_seq_rad =np.vectorize(AngRad_dict.get)(np.array(yes_arr_ports))
    distance_seq=circular_distance(port_seq_rad,yes_port_seq_rad)*8/(2*np.pi)
    MI_Surr2h_dist=np.full((5,Nshuffles), np.nan)
    MI_Surr24h_dist=np.full((5,Nshuffles), np.nan)
    for distance in range(5):
        for lag in range(lags):
            if lag==0:
                dist_hist_seq=arr_pokes[distance_seq==distance]
                dist_port_seq=arr_ports[distance_seq==distance]
                # Nshuffles=np.min([maxnumberSurr(arr_ports), MaxNShuffles]) #Calculate the maximum number of permutations without repetition for this particular dataset
                Nsess=len(arr_pokes[distance_seq==distance])
                MI_Surr2h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 2h
                for i in range(Nshuffles):
                    random.shuffle(dist_port_seq)
                    MI_Surr2h[i]=np.nanmean(MemoryIndex(dist_hist_seq, dist_port_seq, 1))
            if lag==1:
                Nsess=len(arr_pokes[distance_seq==distance])
                MI_Surr24h=np.full(Nshuffles, np.nan) #Iniciate container for the surrogates for 24h
                for i in range(Nshuffles):
                    random.shuffle(distance_seq)
                    dist_hist_seq=arr_pokes[distance_seq==distance]
                    dist_yes_port_seq=yes_arr_ports[distance_seq==distance]
                    MI_Surr24h[i]=np.nanmean(MemoryIndex(dist_hist_seq, dist_yes_port_seq))
        
        MI_Surr2h_dist[distance]=MI_Surr2h
        MI_Surr24h_dist[distance]=MI_Surr24h

    return   MI_Surr2h_dist, MI_Surr24h_dist

def MemoryIndexbyTrl(hist_seq, port_seq):
    cosAng = np.cos([np.pi/4, 0, -np.pi/4, -np.pi/2, -3/4*np.pi, np.pi, 3/4*np.pi, np.pi/2])

    if port_seq > 0:
        Projected_hist_seq_norm = hist_seq / np.nansum(hist_seq)
        Projected_hist_seq = np.roll(cosAng, int(port_seq) - 2) *Projected_hist_seq_norm
        MITrl = np.nansum(Projected_hist_seq)

    return MITrl

def maxnumberSurr(portseq):
    """
    Calculate the number of unique permutations for a port sequence list with potential repetitions.
    
    Parameters:
    portseq (list): List of ports (with or without duplicates).
    
    Returns:
    int: Number of unique permutations.
    """
    # Total number of items
    n = len(portseq)
    
    # Count the occurrences of each unique item
    counts = Counter(portseq)
    
    # Calculate the denominator as the product of factorials of the counts
    denominator = math.prod([math.factorial(count) for count in counts.values()])
    
    # Calculate the number of unique permutations using the formula
    unique_permutations = math.factorial(n) // denominator
    
    return unique_permutations

def output_MIs(pokes_seq ,port_seq,port_seq_lag,min_pokes):
    NumTrials=56
    hist_seq=np.sum(hist_all(port_seq, pokes_seq, NumTrials), axis=2)
    port_seq_arr=np.array(port_seq)
    port_seq_lag_arr=np.array(port_seq_lag)
    hist_all_total=np.concatenate([hist_seq[aa] for aa in range(len(hist_seq))])
    port_seq_total=np.concatenate([port_seq_arr[aa] for aa in range(len(port_seq_arr))])
    port_seq_lag_total=np.concatenate([port_seq_lag_arr[aa] for aa in range(len(port_seq_lag_arr))])
    MI=MemoryIndex(hist_all_total, port_seq_total,min_pokes)
    MILags=MemoryIndex(hist_all_total, port_seq_lag_total, min_pokes)
    MIdyAnim=np.array([MemoryIndex(hist_seq[aa], port_seq_arr[aa],min_pokes) for aa in range(len(port_seq_arr))])
    MIdyAnimLags=np.array([MemoryIndex(hist_seq[aa], port_seq_lag_arr[aa],min_pokes) for aa in range(len(port_seq_lag_arr))])

    return hist_all_total, port_seq_total, port_seq_lag_total, MI, MILags, MIdyAnim, MIdyAnimLags

def distance_MI_variables(port_seq, yes_port_seq,hist_seq,distance):

    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}
    port_seq_all=port_seq[port_seq>0]
    yes_port_seq_all=yes_port_seq[port_seq>0]
    hist_seq_all=hist_seq[port_seq>0]
    port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(port_seq_all))
    yes_port_seq_rad =np.vectorize(AngRad_dict.get)(np.array(yes_port_seq_all))
    distance_seq=circular_distance(port_seq_rad,yes_port_seq_rad)*8/(2*np.pi)
    dist_hist_seq=hist_seq_all[distance_seq==distance]
    dist_port_seq=port_seq_all[distance_seq==distance]
    dist_yes_port_seq=yes_port_seq_all[distance_seq==distance]

    MI=MemoryIndex(dist_hist_seq, dist_port_seq)
    MILags=MemoryIndex(dist_hist_seq, dist_yes_port_seq)
    return MI,MILags#, dist_hist_seq, dist_port_seq, dist_yes_port_seq#, MI, MILags,
