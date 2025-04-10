import numpy.random as npr
import ssm.observations as obs
import ssm.hierarchical as hier
import ssm.stats as stats
from ssm.observations import VonMisesObservations, Observations
from ssm.transitions import StationaryTransitions
from ssm.hmm import HMM
import ssm.transitions as trans
from scipy.special import logsumexp
# from ssm.messages import hmm_expected_states, hmm_filter, hmm_sample, viterbi

class IIDTransitions(StationaryTransitions):
  def __init__(self, K, D, M=0):
    super(StationaryTransitions, self).__init__(K, D, M=M)
    Ps =np.ones(K) * 1 / K #np.array([0.01,0.49, 0.5])# 
    # dirichlet_alpha = np.array([0.5, 0.3, 10])
    # Ps = np.random.dirichlet(alpha=dirichlet_alpha)
    self.log_pis = np.log(Ps)

  @property
  def params(self):
    return (self.log_pis,)

  @params.setter
  def params(self, value):
    self.log_pis = value[0]

  @property
  def log_Ps(self):
    return np.tile(self.log_pis, (self.K, 1))

  def permute(self, perm):
    """
        Permute the discrete latent states.
        """
    # self.log_Ps = self.log_Ps[perm]
    self.log_pis=self.log_pis[perm]

  @property
  def transition_matrix(self):
    return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

  def log_transition_matrices(self, data, input, mask, tag):
      log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
      return log_Ps[None, :, :]


  def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
    K = self.K
    # Ez is the 1st element of expectations,
    # yes, we are getting an array of 3 values summing over all the trials (one for eack K )
    pi = sum([np.sum(Ez, axis=0) for Ez, _, _ in expectations]) + 1e-32  #sum of the expected probabilities of each state for all observations

    # the axis=-1 it is over the of length K, corresponding to the K (=3 for us) different states/strategies
    pi = np.nan_to_num(pi / pi.sum(axis=-1, keepdims=True))

    log_pi = np.log(pi)
    self.log_pis = log_pi - logsumexp(log_pi, axis=-1, keepdims=True)

    
  def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

# a new VonMises class with fixed mu's (means) and with (approximately) 0 kappa (VM precision) for the 0-th HMM state (k=0 below)
class VonMisesObservationsPablo(VonMisesObservations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        # self.mus = npr.randn(K, D)
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

    @property
    def params(self):
        # return self.mus, self.log_kappas
        self.log_kappas

    @params.setter
    # def params(self, value):
    #     # self.mus, self.log_kappas = value
    #     self.log_kappas = value
    def params(self, values):
      # Ensure that the length of the provided list matches the length of self.log_kappas
      if len(values) != self.log_kappas.shape[0]:
          raise ValueError("Length of provided values does not match the length of self.log_kappas")

      # Assign the values to self.log_kappas
      self.log_kappas = values

    def permute(self, perm):
        # self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        # mus, kappas = self.mus, np.exp(self.log_kappas)
        mus, kappas = input, np.exp(self.log_kappas)

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        # print(f"data = {data.shape}, mus = {mus.shape}, kappas = {kappas.shape}, mask = {mask.shape}")
        return stats.vonmises_logpdf(data[:, None, :], mus, kappas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        # D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        D, mus, kappas = self.D, input, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)


    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)  # T x D
        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x K #Probability of each observation to belong to a k state
        assert x.shape[0] == weights.shape[0]
        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D
        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D  #normalized angular mean
        mask = (r_norm.sum(1) == 0)
        r_bar[mask] = 0

        # # Approximation
        # kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K, D1 #precision

        # kappa0[kappa0 == 0] += 1e-6
        # Establecer kappas como constantes (no cambiamos estas en el fit)
        self.log_kappas[0] = np.log(0.001)  # kappa small to create a Connstant
        self.log_kappas[1] = np.log(30)     # kappa for the previous day bump
        self.log_kappas[2] = np.log(30)     #  kappa for today bump
        # for k in range(self.K):
            # if k==0:
            #     self.log_kappas[0] = -20  # K, D
            # else:
            #     self.log_kappas[k] = #np.log(kappa0[k])  # K, D  #This kappas for a model adjusting also kappas.
    

    def smooth(self, expectations, data, input, tag):
        # mus = self.mus
        mus = input
        return expectations.dot(mus)

class PoissonObservations(Observations):

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas))

# a Poisson VonMises observation class with fittable rates, fixed mu's (means) and with (approximately) 0 kappa (VM precision) for the 0-th HMM state (k=0 below)
class PoissonVonMisesObservations(VonMisesObservations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        self.log_lambdas = np.zeros((K, D))
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

    @property
    def params(self):
        return self.log_lambdas, self.log_kappas

    @params.setter
    def params(self, value):
        self.log_lambdas, self.log_kappas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]
        self.log_kappas = self.log_kappas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        # mus, kappas = self.mus, np.exp(self.log_kappas)
        mus, kappas = input, np.exp(self.log_kappas)
        lambdas = np.exp(self.log_lambdas)

        mask = np.ones_like(data, dtype=bool) if mask is None else mask

        # n_data = np.array([[len(d_o) for d_o in d_t] for d_t in data]) # good for the old format
        n_data = np.array([[np.sum(~np.isnan(d_o)) for d_o in d_t] for d_t in data])
        ll_poiss = stats.poisson_logpdf(n_data[:, None, :], lambdas) #, mask=mask[:, None, :])
        if data.shape[-1] == 1: # D == 1
            data = np.stack(data.flatten()) # shape = (T, nmax)
            data = np.nan_to_num(data)
            mask = np.stack(mask.flatten()) # shape = (T, nmax)
            nmax = data.shape[-1] # maximum number of pokes across trials
            ll_vm = stats.vonmises_logpdf(data[:, None, :], np.tile(mus, nmax), np.tile(kappas, nmax), mask=mask[:, None, :])
        else:
            raise NotImplementedError

        return ll_poiss + ll_vm

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, input, np.exp(self.log_kappas)
        lambdas = np.exp(self.log_lambdas)
        n_pokes = npr.poisson(lambdas[z])
        return np.array([npr.vonmises(self.mus[z], kappas[z], D) for i in range(n_pokes)])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas) # T x D
        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x K
        assert x.shape[0] == weights.shape[0]

        ns = np.array([[np.sum(~np.isnan(d_o)) for d_o in d_t] for d_t in x]) # T x D
        for k in range(self.K):
            self.log_lambdas[k] = np.log(np.average(ns, axis=0, weights=weights[:,k]) + 1e-16)

        if x.shape[-1] == 1: # D == 1
            x = np.stack(x.flatten()) # T x nmax
            m = np.concatenate(masks) # T x D
            m = np.stack(m.flatten()) # T x nmax
            ws = np.tile(weights.T[..., None], m.shape[-1]) # K x T x nmax
            weights = ws[np.tile(m, (ws.shape[0],1,1))].reshape((ws.shape[0],-1)) # K x T1
            weights = weights.T # T1 x K
            x = x[m] # (T1,) where T1 is the total number of non-nan elements in x
            x = x[:,None] # (T1, 1)
        else:
            raise NotImplementedError
        # henceforth D1 == 1

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T1 x 2 x D1

        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D1
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D1
        # r_norm = np.atleast_2d(r_norm).T # added by Yashar to avoid error for D1 == 1

        # mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D1
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D1

        mask = (r_norm.sum(1) == 0)
        # mus_k[mask] = 0
        r_bar[mask] = 0

        # Approximation
        kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K, D1

        kappa0[kappa0 == 0] += 1e-6

        for k in range(self.K):
            #self.mus[k] = np.arctan2(*mus_k[k])  #
            if k==0:
                self.log_kappas[0] = -20  # K, D
            else:
                self.log_kappas[k] = np.log(kappa0[k])  # K, D

    def smooth(self, expectations, data, input, tag):
        # TODO: didn't really think much about this... but makes sense(?)
        mus = input
        return expectations.dot(mus)

class HMMPablo(HMM):
    """
    Base class for hidden Markov models.
    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M: dimensionality of inputs
    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """
    def __init__(self, K, D, M=0, init_state_distn=None,
                 transitions='standard',
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 observations="gaussian", observation_kwargs=None,
                 hierarchical_observation_tags=None,**kwargs): # max_iter=1000,

        super(HMMPablo, self).__init__(K, D, M=M, init_state_distn=init_state_distn,
                 transitions='standard',
                 transition_kwargs=transition_kwargs,
                 hierarchical_transition_tags=hierarchical_transition_tags,
                 observations="gaussian", observation_kwargs=observation_kwargs,
                 hierarchical_observation_tags=hierarchical_observation_tags, **kwargs)

        #self.max_iter = max_iter

       # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            constrained=trans.ConstrainedStationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions,
            iidstationary=IIDTransitions
            )

        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = \
                hier.HierarchicalTransitions(transition_classes[transitions], K, D, M=M,
                                        tags=hierarchical_transition_tags,
                                        **transition_kwargs) \
                if hierarchical_transition_tags is not None \
                else transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        self.transitions = transitions


        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            exponential=obs.ExponentialObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            input_driven_obs=obs.InputDrivenObservations,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            vonmisespablo=VonMisesObservationsPablo,
            poissonvonmises=PoissonVonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            no_input_ar=obs.AutoRegressiveObservationsNoInput,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            no_input_robust_ar=obs.RobustAutoRegressiveObservationsNoInput,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = \
                hier.HierarchicalObservations(observation_classes[observations], K, D, M=M,
                                        tags=hierarchical_observation_tags,
                                        **observation_kwargs) \
                if hierarchical_observation_tags is not None \
                else observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.observations = observations
