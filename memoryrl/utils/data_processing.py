import numpy.random as npr
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import i0
from filtering_functions import hist_all, circular_distance
import copy
from memoryIndex_functions import MemoryIndexbyTrl
# from HMM_EM import HMMPablo






new_AngRad = {1: -3*np.pi/4, 2: -np.pi/2, 3: -np.pi/4, 4: 0, 5: np.pi/4, 6: np.pi/2, 7: 3*np.pi/4, 8: np.pi}

def convert_to_angular_poked(hist):
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



def rad_avg_poked(hist):
    new_AngRad = {1: -3*np.pi/4, 2: -np.pi/2, 3: -np.pi/4, 4: 0, 5: np.pi/4, 6: np.pi/2, 7: 3*np.pi/4, 8: np.pi}
    avg_ang_sess=[]
    for sess in range(len(hist)):
        avg_ang=[]
        for i, row in enumerate(hist[sess]):
            # print(i)
            positions = np.where(row !=0)[0]  # columns where there is a 1
            if positions.size > 0:  # Only row with at least 1 val
                mapped_values = [new_AngRad.get(pos + 1, np.nan) for pos in positions]
                avg_ang.append(np.angle(np.exp(1j * np.array(mapped_values)).mean()))
        avg_ang_sess.extend(avg_ang)
    return avg_ang_sess



def hist_data_distance(Hist_data, port_seq, yes_port_seq, sess_distance):
  Hist_data_dist=np.full_like(Hist_data[sess_distance], np.nan)
  for ss in range(len(port_seq[sess_distance])):
    if (port_seq[sess_distance][ss]>yes_port_seq[sess_distance][ss] and (yes_port_seq[sess_distance][ss]>4 and (port_seq[sess_distance][ss]-yes_port_seq[sess_distance][ss])<=3)) or (port_seq[sess_distance][ss]<yes_port_seq[sess_distance][ss] and (yes_port_seq[sess_distance][ss]-port_seq[sess_distance][ss])>=5):
      Hist_data_dist[ss]=np.roll(np.flip(np.roll(Hist_data[sess_distance][ss],4-int(port_seq[sess_distance][ss]), axis=Hist_data.ndim-2), axis=Hist_data.ndim-2),-1,axis=Hist_data.ndim-2) #reversed
    else:
      Hist_data_dist[ss]=np.roll(Hist_data[sess_distance][ss],4-int(port_seq[sess_distance][ss]), axis=Hist_data.ndim-2)
  return Hist_data_dist



def circular_distance(angle1, angle2):
  angle_difference = np.abs(angle2 - angle1)
  wrapped_angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
  return wrapped_angle_difference



def water_availability(datas):
    new_list = np.array([[array.item() for array in inner_list] for inner_list in datas])
    first_occurrences = [
        (sub_list.tolist().index(1) if 1 in sub_list.tolist() else len(sub_list) - 1) if len(sub_list) > 0 else None
        for sub_list in new_list
    ]
    first_occurrences_ = np.array([[] if value is None else [value] for value in first_occurrences])
    return first_occurrences_[0][0]


def subsample_data(arr_pokes, distance_seq, port_seq, lag_port_seq, NSamples, fraction=0.8):
    """Takes a 80%  of the data maintaining the coherence between variables"""
    if fraction==False:
       n_samples=NSamples
    else:
       n_samples = int(len(arr_pokes) * fraction)  # 80% de los datos
    indexes = np.random.choice(len(arr_pokes), n_samples, replace=False)  # Índices aleatorios sin repetición
    
    return arr_pokes[indexes], distance_seq[indexes], port_seq[indexes], lag_port_seq[indexes]



def MemoryIndex_histogram(Big_Hist_data, port_seq, lag_port_seq):
    h=np.sum([np.roll(Big_Hist_data[ss],8-int(port_seq[ss]), axis=0) for ss in range(len(Big_Hist_data))], axis=0)
    hLags=np.sum([np.roll(Big_Hist_data[ss],8-int(lag_port_seq[ss]), axis=0) for ss in range(len(Big_Hist_data))], axis=0)
    MI_distances=MemoryIndexbyTrl(h, 8)
    MILags_distances=MemoryIndexbyTrl(hLags, 8)
    return MI_distances, MILags_distances



def figure_data_based_MI(MI,MILags,MI_Surr2h_dist, MI_Surr24h_dist, colors):
    CI_MI2h_Surr=np.percentile(MI_Surr2h_dist, [99,1], axis=0)
    CI_MI_SurrLags=np.percentile(MI_Surr24h_dist, [99,1], axis=0)
    
    x_plot=np.array([0,1,2,3,4])
    fig, axs = plt.subplots(1,1, figsize=np.array([6.4, 6.4]) * .7)
    axs.scatter([x_plot[0]], [MI[0]],  color=colors[0], alpha=0.4,s=7, zorder=3)
    axs.plot(x_plot[:2], MI[:2], '--',  alpha=0.5, color=colors[0])
    axs.plot(x_plot[1:], MI[1:], '.-', alpha=0.5, color=colors[0], label='2h MI')
    axs.scatter([x_plot[0]], [MILags[0]],  color=colors[1], alpha=0.4,s=7, zorder=3)
    axs.plot(x_plot[:2], MILags[:2], '--',  alpha=0.5, color=colors[1])
    axs.plot(x_plot[1:], MILags[1:], '.-',  alpha=0.5, color=colors[1], label='24h MI')
    axs.fill_between(x_plot, CI_MI2h_Surr[0], CI_MI2h_Surr[1], color=colors[0], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI2h_Surr, axis=0),'--' ,color=colors[0], alpha=0.1)
    axs.fill_between(x_plot, CI_MI_SurrLags[0], CI_MI_SurrLags[1], color=colors[1], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI_SurrLags, axis=0),'--' ,color=colors[1], alpha=0.1)

    axs.axhline(y=0, color='grey', alpha=0.2)
    axs.spines[['top', 'right']].set_visible(False)
    axs.legend(loc= 'upper left',bbox_to_anchor=(1.2, 4.2), fontsize=8)
    axs.set_ylabel('Memory Index')
    axs.set_xlabel('Distance to Yesterday Port')
    axs.set_title('Model Free Memory Index \n Real Data')
    axs.set_ylim(-.2,0.4)
    axs.legend(loc= 'upper right',bbox_to_anchor=(1.3,1), fontsize=8)
    # plt.tight_layout()



def figure_model_based_MI(models, port_seq, lag_port_seq, distance_seq,MI_Surr2h_dist, MI_Surr24h_dist, colors, Transitions,SurrogateMode, DrugType, save=False):
    MI_m, MILags_m=model_based_MI(models, port_seq, lag_port_seq, distance_seq)[:2]
    Kappas = [np.exp(models[distance]['model'].observations.log_kappas) for distance in range(5)]
    kappas=Kappas[0][1]

    CI_MI2h_Surr=np.percentile(MI_Surr2h_dist, [99,1], axis=0)
    CI_MI_SurrLags=np.percentile(MI_Surr24h_dist, [99,1], axis=0)
    
    x_plot=np.array([0,1,2,3,4])
    fig, axs = plt.subplots(1,1, figsize=np.array([6.4, 6.4]) * .7)
    axs.scatter([x_plot[0]], [MI_m[0]],  color=colors[0], alpha=0.4,s=7, zorder=3)
    axs.plot(x_plot[:2], MI_m[:2], '--',  alpha=0.5, color=colors[0])
    axs.plot(x_plot[1:], MI_m[1:], '.-', alpha=0.5, color=colors[0], label='Model Pred. 2h MI')
    axs.scatter([x_plot[0]], [MI_m[0]],  color=colors[1], alpha=0.4,s=7, zorder=3)
    axs.plot(x_plot[:2], MILags_m[:2], '--',  alpha=0.5, color=colors[1])
    axs.plot(x_plot[1:], MILags_m[1:], '.-',  alpha=0.5, color=colors[1], label='Model Pred. 24h MI')
    axs.fill_between(x_plot, CI_MI2h_Surr[0], CI_MI2h_Surr[1], color=colors[0], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI2h_Surr, axis=0),'--' ,color=colors[0], alpha=0.1)
    axs.fill_between(x_plot, CI_MI_SurrLags[0], CI_MI_SurrLags[1], color=colors[1], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI_SurrLags, axis=0),'--' ,color=colors[1], alpha=0.1)

    axs.axhline(y=0, color='grey', alpha=0.2)
    axs.spines[['top', 'right']].set_visible(False)
    axs.legend(loc= 'upper left',bbox_to_anchor=(1.2, 4.2), fontsize=8)
    axs.set_ylabel('Memory Index')
    axs.set_xlabel('Distance to Yesterday Port')
    axs.set_title(f'Model Based Memory Index ({Transitions}) \n kappas={np.round(kappas[0])}')
    axs.set_ylim(-.2,0.4)
    axs.legend(loc= 'upper right',bbox_to_anchor=(1.3,1), fontsize=8)
    if save: 
        plt.savefig(f'D:\AutoSynaptopatiesData\Figures\HMM\Model_Based_MI_{DrugType}_{SurrogateMode}_{Transitions}.png', dpi=300, bbox_inches="tight")
    # plt.tight_layout()



def train_hmm_models(arr_pokes, distance_seq, port_seq, lag_port_seq, transitions, kappas,seed):
    """
    Trains HMM models for different distances and returns a dictionary with the models.

    Parameters:
    - arr_pokes: numpy array with the poke data.
    - distance_seq: numpy array with the corresponding distances.
    - port_seq_rad: numpy array with the port angles in radians.
    - lags_port_seq_rad: numpy array with the lagged port angles in radians.
    - transitions:str i.e.; (iidstationary, standard)

    Returns:
    - models: dictionary with the HMM models for each distance.
    """

    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}

    port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(port_seq))
    lags_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(lag_port_seq))

    npr.seed(seed)  # Fijar la semilla para reproducibilidad
    K = 3  # States of the HMM (exploration, lag, today)
    D = 1  # Dimensions of the explorations (angles in radians)
    
    models = {}
    for i in range(5):  # Iterate over the 5 possible distances
        datas = arr_pokes[distance_seq == i]
        datas = [d[~np.isnan(d.astype(float))] for d in datas]
        datas = [d[:, None].astype('float64') if len(d) > 0 else np.array([[100.]]) for d in datas]

        masks = [~np.isnan(d) for d in datas]
        ports = port_seq_rad[distance_seq == i]
        ports_yes = lags_port_seq_rad[distance_seq == i]

        inputs = [np.array([0, ports_yes[sess], ports[sess]])[:, None] for sess in range(len(ports))]

        test_hmm = HMMPablo(K, D, observations='vonmisespablo', transitions=transitions)
        test_hmm.observations.log_kappas = copy.deepcopy(np.log(kappas[i]))
        # test_hmm.transitions.log_pis = copy.deepcopy(np.log(init_probabilities))
        test_hmm.fit(datas, inputs=inputs, masks=masks, num_iters=1000, tolerance=10**-5)

        models[i] = {
            'model': test_hmm,
            'datas': datas,
            'inputs': inputs,
            'masks': masks
        }

    return models




def model_based_MI(models, port_seq, lag_port_seq, distance_seq):
    Kappas = [np.exp(models[distance]['model'].observations.log_kappas) for distance in range(5)]
    Expectations=[[np.nanmean(models[distance]['model'].expected_states(models[distance]['datas'][ss], input=models[distance]['inputs'][ss], mask=models[distance]['masks'][ss])[0], axis=0) 
        for ss in range(len(models[distance]['datas']))] for distance in range(5)]
    # Estract the part I wanna modify (i=0)
    mod_data = np.array(Expectations[0])  
    # Compute the mean of the columns 1 and 2 at distance 0
    col_mean = np.mean(mod_data[:, 1:3], axis=1)

    # Replace columns 1 and 2 with the mean
    mod_data[:, 1] = col_mean
    mod_data[:, 2] = col_mean

    # Asigna los valores modificados de vuelta a Probabilities
    Expectations[0] = mod_data

    MI_distances=np.full(5, np.nan)
    MILags_distances=np.full(5, np.nan)
    Big_Hist_data=[]
    port_seq_sorted=[]
    lag_port_seq_sorted=[]
    distance_seq_sorted=[]

    for i in range(5):
        Big_Hist_data_model=np.full((np.shape(Expectations[i])[0],8), np.nan)
        ports=np.array(models[i]['inputs'])[:,-1]
        ports_yes=np.array(models[i]['inputs'])[:,1]
        kappas=Kappas[i]
        expectations=Expectations[i]
        for ss in range(len(expectations)):
            x = np.array([np.pi/4, 0, -np.pi/4, -np.pi/2, -3*np.pi/4, np.pi, 3*np.pi/4, np.pi/2])
            mu, kappa_val = ports[ss], kappas[-1]
            muE, kappaE = 0, kappas[0]
            muY, kappaY = ports_yes[ss], kappas[1]

            y = (expectations[ss][0] * np.exp(kappaE * np.cos(x - muE)) / (2 * np.pi * i0(kappaE)) +
                    expectations[ss][1] * np.exp(kappaY * np.cos(x - muY)) / (2 * np.pi * i0(kappaY)) +
                    expectations[ss][-1] * np.exp(kappa_val * np.cos(x - mu)) / (2 * np.pi * i0(kappa_val)))
            Big_Hist_data_model[ss]=y
        Big_Hist_data.extend(Big_Hist_data_model)

        new_port_seq=port_seq[distance_seq==i]
        port_seq_sorted.extend(new_port_seq)
        new_lag_port_seq=lag_port_seq[distance_seq==i]
        lag_port_seq_sorted.extend(new_lag_port_seq)
        new_distance_seq=distance_seq[distance_seq==i]
        distance_seq_sorted.extend(new_distance_seq)
        
        MI_distances[i], MILags_distances[i]=MemoryIndex_histogram(Big_Hist_data_model,new_port_seq,new_lag_port_seq  )

    return MI_distances, MILags_distances, np.array(Big_Hist_data), np.array(port_seq_sorted), np.array(lag_port_seq_sorted), np.array(distance_seq_sorted)




def distance_surrogates(Big_Hist_data,port_seq, lag_port_seq,distance_seq,mode,Nshuffles):
  AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}

  port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(port_seq))
  yes_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(lag_port_seq))

  if mode=='Distance':
    MI_Surr2h_dist=np.full((Nshuffles,5), np.nan)
    MI_Surr24h_dist=np.full((Nshuffles,5), np.nan)
    for i in range(Nshuffles):
        random.shuffle(distance_seq)
        MI_Surr2h=np.full(5, np.nan)
        MI_Surr24h=np.full(5, np.nan)
        for distance in range(5):
            histogram=[np.sum([np.roll(Big_Hist_data[distance_seq==distance][ss],8-int(port_seq[distance_seq==distance][ss]), axis=0) for ss in range(len(Big_Hist_data[distance_seq==distance]))], axis=0) ]
            histogram_Lags=[np.sum([np.roll(Big_Hist_data[distance_seq==distance][ss],8-int(lag_port_seq[distance_seq==distance][ss]), axis=0) for ss in range(len(Big_Hist_data[distance_seq==distance]))], axis=0) ]
            MI_Surr2h[distance]=MemoryIndexbyTrl(histogram, 8)
            MI_Surr24h[distance]=MemoryIndexbyTrl(histogram_Lags, 8)

        MI_Surr2h_dist[i]=MI_Surr2h
        MI_Surr24h_dist[i]=MI_Surr24h
    return MI_Surr2h_dist, MI_Surr24h_dist
  
  if mode=='Shuffle':
    MI_Surr2h_dist=np.full((Nshuffles,5), np.nan)
    MI_Surr24h_dist=np.full((Nshuffles,5), np.nan)
    for lags in range(2):
      if lags==0:
        for i in range(Nshuffles):
          shuffled_seq = port_seq.copy()
          random.shuffle(shuffled_seq)
          new_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(shuffled_seq))
          Distance_seq=circular_distance(new_port_seq_rad,yes_port_seq_rad)*8/(2*np.pi)
          MI_Surr2h=np.full(5, np.nan)
          for distance in range(5):
              histogram_Lags=[np.sum([np.roll(Big_Hist_data[Distance_seq==distance][ss],8-int(shuffled_seq[Distance_seq==distance][ss]), axis=0) for ss in range(len(Big_Hist_data[Distance_seq==distance]))], axis=0) ]
              MI_Surr2h[distance]=MemoryIndexbyTrl(histogram_Lags, 8)
          MI_Surr2h_dist[i]=MI_Surr2h
      if lags==1:
        for i in range(Nshuffles):
          shuffled_seq = lag_port_seq.copy()
          random.shuffle(shuffled_seq)
          new_yes_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(shuffled_seq))
          Distance_seq=circular_distance(port_seq_rad,new_yes_port_seq_rad)*8/(2*np.pi)
          MI_Surr24h=np.full(5, np.nan)
          for distance in range(5):
              histogram_Lags=[np.sum([np.roll(Big_Hist_data[Distance_seq==distance][ss],8-int(shuffled_seq[Distance_seq==distance][ss]), axis=0) for ss in range(len(Big_Hist_data[Distance_seq==distance]))], axis=0) ]
              MI_Surr24h[distance]=MemoryIndexbyTrl(histogram_Lags, 8)
          MI_Surr24h_dist[i]=MI_Surr24h

        MI_Surr2h_dist[i]=MI_Surr2h
    return MI_Surr2h_dist, MI_Surr24h_dist
  if mode=='Group':
    MI_Surr2h_dist=np.full((5,Nshuffles), np.nan)
    MI_Surr24h_dist=np.full((5,Nshuffles), np.nan)
    for distance in range(5):
      for lag in range(2):
          if lag==0:
              MI_Surr2h=np.full(Nshuffles, np.nan)
              new_port_seq=port_seq[distance_seq==distance]
              for i in range(Nshuffles):
                  random.shuffle(new_port_seq)
                  histogram=[np.sum([np.roll(Big_Hist_data[distance_seq==distance][ss],8-int(new_port_seq[ss]), axis=0) for ss in range(len(Big_Hist_data[distance_seq==distance]))], axis=0) ]
                  MI_Surr2h[i]=MemoryIndexbyTrl(histogram, 8)
              MI_Surr2h_dist[distance]=MI_Surr2h
          if lag==1:
              new_yes_port_seq=lag_port_seq[distance_seq==distance]
              MI_Surr24h=np.full(Nshuffles, np.nan)
              for i in range(Nshuffles):
                  random.shuffle(new_yes_port_seq)
                  histogram_Lags=[np.sum([np.roll(Big_Hist_data[distance_seq==distance][ss],8-int(new_yes_port_seq[ss]), axis=0) for ss in range(len(Big_Hist_data[distance_seq==distance]))], axis=0) ]
                  MI_Surr24h[i]=MemoryIndexbyTrl(histogram_Lags, 8) 
              MI_Surr24h_dist[distance]=MI_Surr24h
     
    return MI_Surr2h_dist.T, MI_Surr24h_dist.T
  


def dataset_MI(ALL_portsPoked_REC,ALL_PORTS_REC,ALL_PORTS_LAGS_REC, ALL_avgportsPoked_REC ):
    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}

    port_seqs=np.concatenate( [ALL_PORTS_REC[aa] for aa in range(len(ALL_PORTS_REC))])
    lags_port_seqs=np.concatenate( [ALL_PORTS_LAGS_REC[aa] for aa in range(len(ALL_PORTS_LAGS_REC))])
    arr_ports=port_seqs[lags_port_seqs>0]
    lags_arr_ports=lags_port_seqs[lags_port_seqs>0]
    hist_seqs=np.concatenate( [ALL_avgportsPoked_REC[aa] for aa in range(len(ALL_avgportsPoked_REC))])
    arr_pokes=hist_seqs[lags_port_seqs>0]
    all_trials_hist=hist_all(ALL_PORTS_REC,ALL_portsPoked_REC,56)
    arr_all_trials_pokes=np.concatenate([all_trials_hist[aa] for aa in range(len(all_trials_hist))])[lags_port_seqs>0]
    port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(arr_ports))
    lags_port_seq_rad =np.vectorize(AngRad_dict.get)(np.array(lags_arr_ports))
    # np.random.shuffle(lags_port_seq_rad)
    distance_seq=circular_distance(port_seq_rad,lags_port_seq_rad)*8/(2*np.pi)
    Big_Hist_data=np.sum(arr_all_trials_pokes,axis=1)
    arr_all_trials_pokes_polar=convert_to_angular_poked(arr_all_trials_pokes)
    return Big_Hist_data, arr_pokes, arr_ports, lags_arr_ports, distance_seq, arr_all_trials_pokes,arr_all_trials_pokes_polar



def fig_only_probability_model_based(models, transitions, colors):
    x=range(0,5)
    fig, axs = plt.subplots(1,1, figsize=np.array([6.4, 6.4]) * .7)
    plt.rcParams['font.family'] = 'Arial'  # Usa Arial para toda la figura

    # kappas = [np.exp(models[distance]['model'].observations.log_kappas) for distance in range(5)]
    if isinstance(models, list):
        expectations=[[[np.nanmean(models[m][distance]['model'].expected_states(models[m][distance]['datas'][ss], input=models[m][distance]['inputs'][ss], mask=models[m][distance]['masks'][ss])[0], axis=0) 
        for ss in range(len(models[m][distance]['datas']))] for distance in range(5)] for m in range(len(models))]
        Probabilities_mean_arr= np.array([[np.nanmean(expectations[m][i], axis=0) for i in range(5)] for m in range(len(models))])
        CIs=np.array([np.percentile(Probabilities_mean_arr[:,i],[99,1],axis=0) for i in range(5)])

        CI_tod=CIs[:,:,-1]
        CI_yes=CIs[:,:,1]
        
        plt.plot(np.nanmean(Probabilities_mean_arr, axis=0)[:,1], '.-',  label= '24h Strategy', color=colors[1], alpha=0.4)
        plt.fill_between(x, CI_yes[:,0], CI_yes[:,1], color=colors[1], alpha=0.1)
        plt.plot(np.nanmean(Probabilities_mean_arr, axis=0)[:,2], '.-',  label= '2h Strategy', color=colors[0],alpha=0.4)
        plt.fill_between(x, CI_tod[:,0], CI_tod[:,1], color=colors[0], alpha=0.1)
    else:
        expectations=[[np.nanmean(models[distance]['model'].expected_states(models[distance]['datas'][ss], input=models[distance]['inputs'][ss], mask=models[distance]['masks'][ss])[0], axis=0) 
        for ss in range(len(models[distance]['datas']))] for distance in range(5)]
        Probabilities= [np.array(expectations[i]) for i in range(5)]
        # Estract the part I wanna modify (i=0)
        mod_data = Probabilities[0]  
        # Compute the mean of the columns 1 and 2 at distance 0
        col_mean = np.mean(mod_data[:, 1:3], axis=1)

        # Replace columns 1 and 2 with the mean
        mod_data[:, 1] = col_mean
        mod_data[:, 2] = col_mean

        # Asigna los valores modificados de vuelta a Probabilities
        Probabilities[0] = mod_data

        Probabilities_mean_arr= np.array([np.nanmean(Probabilities[i], axis=0) for i in range(5)])
    
        plt.scatter([x[0]], [Probabilities_mean_arr[0,1]],  color=colors[1], alpha=0.4,s=7, zorder=3)
        plt.plot(x[:2], Probabilities_mean_arr[:2,1], '--',  color=colors[1], alpha=0.4)
        plt.plot(x[1:], Probabilities_mean_arr[1:,1], '.-',  label= '24h Strategy', color=colors[1], alpha=0.4)
        plt.scatter([x[0]], [Probabilities_mean_arr[0,2]],  color=colors[0], alpha=0.4, s=7, zorder=3)
        plt.plot(x[:2], Probabilities_mean_arr[:2,2], '--',  label= '2h Strategy', color=colors[0],alpha=0.4)
        plt.plot(x[1:], Probabilities_mean_arr[1:,2], '.-',  label= '2h Strategy', color=colors[0],alpha=0.4)
    axs.spines[['top', 'right']].set_visible(False)
    axs.axhline(y=0, color='grey', alpha=0.2)
    axs.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    axs.set_ylabel('Probability')
    axs.set_xlabel('Distance to yesterday Port')
    axs.set_title(f'Model Based Probability \n{transitions}')
    axs.set_ylim(-.2,0.4)
    axs.legend(loc= 'upper right',bbox_to_anchor=(1.3,1), fontsize=8)
    # plt.tight_layout()




def figure_expectations_model_based(models, transitions):
    kappas = [np.exp(models[distance]['model'].observations.log_kappas) for distance in range(5)]
    expectations=[[np.nanmean(models[distance]['model'].expected_states(models[distance]['datas'][ss], input=models[distance]['inputs'][ss], mask=models[distance]['masks'][ss])[0], axis=0) 
    for ss in range(len(models[distance]['datas']))] for distance in range(5)]
    Probabilities_mean_arr= np.array([np.nanmean(expectations[i], axis=0) for i in range(5)])
    CIs=np.array([np.percentile(expectations[i],[99,1],axis=0) for i in range(5)])

    CI_tod=CIs[:,:,-1]
    CI_yes=CIs[:,:,1]
    CI_expl=CIs[:,:,0]

    fig, axs = plt.subplots(2,3, figsize=np.array([6.4*3, 4.8*2]) * .7)
    axs[0,0].spines[['top', 'right']].set_visible(False)
    axs[0,0].plot(expectations[0], linewidth=0.5, label=['Exploration', 'Yesterday', 'Today'])
    axs[0,0].set_ylim(0,1)
    axs[0,0].set_ylabel('Probability')
    axs[0,0].set_xlabel('Trial Number')
    axs[0,0].set_title('Distance0')


    axs[0,1].spines[['top', 'right']].set_visible(False)
    axs[0,1].plot(expectations[1], linewidth=0.5, label=['Exploration', 'Yesterday', 'Today'])
    axs[0,1].set_ylim(0,1)
    axs[0,1].set_ylabel('Probability')
    axs[0,1].set_xlabel('Trial Number')
    axs[0,1].set_title('Distance1')

    axs[0,2].spines[['top', 'right']].set_visible(False)
    axs[0,2].plot(expectations[2], linewidth=0.5, label=['Exploration', 'Yesterday', 'Today'])
    axs[0,2].legend(loc='upper right', bbox_to_anchor=(1.4, 1.05))
    axs[0,2].set_ylim(0,1)
    axs[0,2].set_ylabel('Probability')
    axs[0,2].set_xlabel('Trial Number')
    axs[0,2].set_title('Distance2')

    axs[1,0].spines[['top', 'right']].set_visible(False)
    axs[1,0].plot(expectations[3], linewidth=0.5, label=['Exploration', 'Yesterday', 'Today'])
    axs[1,0].set_ylim(0,1)
    # axs[1,1].legend(loc='upper right', bbox_to_anchor=(1.4, 1.05))
    axs[1,0].set_ylabel('Probability')
    axs[1,0].set_xlabel('Trial Number')
    axs[1,0].set_title('Distance3')

    axs[1,1].spines[['top', 'right']].set_visible(False)
    axs[1,1].plot(expectations[4], linewidth=0.5, label=['Exploration', 'Yesterday', 'Today'])
    axs[1,1].set_ylim(0,1)
    # axs[1,1].legend(loc='upper right', bbox_to_anchor=(1.4, 1.05))
    axs[1,1].set_ylabel('Probability')
    axs[1,1].set_xlabel('Trial Number')
    axs[1,1].set_title('Distance4')

    x=range(0,5)
    axs[1,2].plot(Probabilities_mean_arr[:,0], '.-',  label= 'Explotarion',alpha=0.1)
    axs[1,2].fill_between(x, CI_expl[:,0], CI_expl[:,1], color='tab:blue', alpha=0.05)
    axs[1,2].plot(Probabilities_mean_arr[:,1], '.-',  label= 'Yesterday',alpha=0.4)
    axs[1,2].fill_between(x, CI_yes[:,0], CI_yes[:,1], color='tab:orange', alpha=0.1)
    axs[1,2].plot(Probabilities_mean_arr[:,2], '.-',  label= 'Today',alpha=0.4)
    axs[1,2].fill_between(x, CI_tod[:,0], CI_tod[:,1], color='tab:green', alpha=0.1)
    axs[1,2].spines[['top', 'right']].set_visible(False)
    axs[1,2].set_ylim(0,1)
    axs[1,2].legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    axs[1,2].set_ylabel('Probability')
    axs[1,2].set_xlabel('Distance to yesterday Port')
    axs[1,2].set_title('Probability over Distance')

    fig.suptitle(f'OUPUT FIT {transitions} : RECALL \nKappas= {np.round(kappas[0][1][0])}')

    plt.tight_layout()




def fig_probability_model_free(Hist_data, port_seq, lag_port_seq,Surr_Prob_2h, Surr_Prob_24h,colors, DrugType,SurrogateMode, save=False):  
    Probability=probability_model_free(Hist_data, port_seq, lag_port_seq)
    CIProb_2hSurr=np.percentile(Surr_Prob_2h, [99,1], axis=0)
    CIProb_24hSurr=np.percentile(Surr_Prob_24h, [99,1], axis=0)

    x=range(0,5)
    fig, axs = plt.subplots(1,1, figsize=np.array([6.4, 6.4]) * .7)
    plt.rcParams['font.family'] = 'Arial'  # Usa Arial para toda la figura
    plt.scatter([x[0]], [Probability[0,1]],  color=colors[1], alpha=0.4,s=7, zorder=3)
    plt.plot(x[:2], Probability[:2,1], '--',  color=colors[1], alpha=0.4)
    plt.fill_between(x[:2], CIProb_24hSurr[0][:2], CIProb_24hSurr[1][:2], color=colors[1], alpha=0.05)
    plt.plot(x[1:], Probability[1:,1], '.-',  label= '24h Strategy', color=colors[1], alpha=0.4)
    plt.fill_between(x[1:], CIProb_24hSurr[0][1:], CIProb_24hSurr[1][1:], color=colors[1], alpha=0.05)

    plt.scatter([x[0]], [Probability[0,2]],  color=colors[0], alpha=0.4, s=7, zorder=3)
    plt.plot(x[:2], Probability[:2,2], '--',   color=colors[0],alpha=0.4)
    plt.fill_between(x[:2], CIProb_2hSurr[0][:2], CIProb_2hSurr[1][:2], color=colors[0], alpha=0.05)
    plt.plot(x[1:], Probability[1:,2], '.-',  label= '2h Strategy', color=colors[0],alpha=0.4)
    plt.fill_between(x[1:], CIProb_2hSurr[0][1:], CIProb_2hSurr[1][1:], color=colors[0], alpha=0.05)

    axs.spines[['top', 'right']].set_visible(False)
    axs.axhline(y=0, color='grey', alpha=0.2)
    axs.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    axs.set_ylabel('Probability')
    axs.set_xlabel('Distance to yesterday Port')
    axs.set_title(f'Model Free Probability')
    axs.set_ylim(-0.01,0.3)
    axs.legend(loc= 'upper right',bbox_to_anchor=(1.3,1), fontsize=8)
    if save:
        plt.savefig(f'D:\AutoSynaptopatiesData\Figures\Model_free\Probabilities_model_free{DrugType}_{SurrogateMode}.png', dpi=300, bbox_inches="tight")
def probability_model_free(Hist_data, port_seq, lag_port_seq, distance_flag=True, distance_seq=None):
    AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}
    if distance_flag==True:
        port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(port_seq))
        yes_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(lag_port_seq))
        distance_seq=circular_distance(port_seq_rad,yes_port_seq_rad)*8/(2*np.pi)
    elif distance_seq is None:
        raise ValueError("If 'distance_flag' is False, 'distance_seq' must be fed.")

    sess0=np.array(distance_seq)==0.0
    sess1=np.array(distance_seq)==1.0
    sess2=np.array(distance_seq)==2.0
    sess3=np.array(distance_seq)==3.0
    sess4=np.array(distance_seq)==4.0
    # Generate histograms for all distances
    hist0 = hist_data_distance(Hist_data, port_seq, lag_port_seq, sess0)
    hist1 = hist_data_distance(Hist_data, port_seq, lag_port_seq, sess1)
    hist2 = hist_data_distance(Hist_data, port_seq, lag_port_seq, sess2)
    hist3 = hist_data_distance(Hist_data, port_seq, lag_port_seq, sess3)
    hist4 = hist_data_distance(Hist_data, port_seq, lag_port_seq, sess4)

    # Combine the histograms in a list
    histograms = [hist0,hist1,hist2,hist3,hist4]

    bin_centers = np.linspace(-np.pi, np.pi, num=16, endpoint=False) + np.pi / 8
    bin_width = bin_centers[1] - bin_centers[0]
    bins = np.concatenate([bin_centers - bin_width / 2, [bin_centers[-1] + bin_width / 2]])
    Probability=np.full((5,3), np.nan)
    for i in range(5): 
        counts, bin_edges=np.histogram(rad_avg_poked(histograms[i]),bins=bins, density=False)
        if i==0:
            MeanExplor = (np.sum(counts)-counts[7])/15
            P2 = ((counts[7]-MeanExplor)/np.sum(counts))/2
            P1 = ((counts[7]-MeanExplor)/np.sum(counts))/2
            P0 = 1-(P2+P1)
        else:
            MeanExplor = (np.sum(counts)-(counts[7])+counts[7+i*2])/14
            P1 = (counts[7+i*2]-MeanExplor)/np.sum(counts)
            P2 = (counts[7]-MeanExplor)/np.sum(counts)
            P0 = 1-(P2+P1)
        P0, P1, P2 = max(P0, 0), max(P1, 0), max(P2, 0)
        Probability[i]=np.array([P0, P1, P2])
    return(Probability)


def compute_surrogate_mode_free_probability(Hist_data, port_seq, lag_port_seq, mode ):
    """ Computes the Probabilities & Lag Probabilities with shuffled ports """
    if mode=='distance':
        AngRad_dict = {1: np.pi/4, 2: 0, 3: -np.pi/4, 4: -np.pi/2, 5: -3*np.pi/4, 6: np.pi, 7: 3*np.pi/4, 8: np.pi/2}
        port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(port_seq))
        yes_port_seq_rad=np.vectorize(AngRad_dict.get)(np.array(lag_port_seq))
        distance_seq=circular_distance(port_seq_rad,yes_port_seq_rad)*8/(2*np.pi)
        shuffled_distances = distance_seq.copy()  # Copia para evitar modificar el original
        np.random.shuffle(shuffled_distances)
        ProbT = probability_model_free(Hist_data, port_seq, lag_port_seq,distance_flag=False,distance_seq=shuffled_distances )[:,2]
        ProbY = probability_model_free(Hist_data, port_seq, lag_port_seq,distance_flag=False,distance_seq=shuffled_distances)[:,1]
    if mode=='Shuffle':
        shuffled_ports = port_seq.copy()  # Copia para evitar modificar el original
        np.random.shuffle(shuffled_ports)
        ProbT = probability_model_free(Hist_data, shuffled_ports, lag_port_seq, distance_flag=True)[:,2]
        lag_shuffled_ports = lag_port_seq.copy()
        np.random.shuffle(lag_shuffled_ports)
        ProbY = probability_model_free(Hist_data, port_seq, lag_shuffled_ports,distance_flag=True)[:,1]
    return ProbT, ProbY