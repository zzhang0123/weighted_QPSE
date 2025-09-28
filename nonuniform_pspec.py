import numpy as np
import uvtools.dspec as dspec


def trace_AB(A, B):
    """
    Compute the trace of the matrix product A @ B.T efficiently.
    """
    return np.sum(A*B.T)

def example_beam():
    """
    Load example beam pattern data from saved file.
    
    Loads normalized beam response pattern and solid angle element
    from 'fromKaifeng/beam_info.npz'. The beam is normalized by the
    primary beam solid angle Omega_p.
    
    Returns
    -------
    beam : ndarray
        Normalized beam response of shape (N_freqs, N_directions)
        Units: dimensionless (normalized by Omega_p)
    d_Omega : float
        Solid angle element for numerical integration over directions
        Units: steradians
        
    Notes
    -----
    The beam file should contain:
    - arr_0: beam response array
    - arr_1: beam_omega (complex, real part used)
    - arr_2: N parameter for solid angle calculation
    """
    _ = np.load("fromKaifeng/beam_info.npz")
    beam, beam_omega, N = (_['arr_0'], _['arr_1'], _['arr_2'])

    omega_p = beam_omega.real

    beam = beam/omega_p[:, None]
    d_Omega = np.pi/(3.*N*N)

    return beam, d_Omega

test_beam, test_d_Omega = example_beam()

def generate_freq_weights(flag1, flag2, return_matrix=False, method_func=None):
    '''
    method_func: function to generate weights given flag arrays.
    '''
    if method_func is not None:
        weight1 = method_func(flag1)
        weight2 = method_func(flag2)
    else:
        weight1 = flag1
        weight2 = flag2
    if return_matrix:
        return np.einsum('i,j->ij', weight1, weight2.conj())
    return weight1, weight2

def get_quad_Fourier_list(Ndlys, d_eta):

    Q_alt_li = []
    Nfreqs = Ndlys

    for mode in range(Ndlys):

        if Ndlys % 2 == 0:
            start_idx = -Ndlys/2
        else:
            start_idx = -(Ndlys - 1)/2
        m = (start_idx + mode) * np.arange(Nfreqs)
        m = np.exp(-2j * np.pi * m / Ndlys) * d_eta
        Q_alt = np.einsum('i,j', m.conj(), m) # dot it with its conjugate
        Q_alt_li.append(Q_alt)

    return Q_alt_li

def generate_quad_Beam(normalised_beam, d_Omega=1.0, isotropic_signal=False):
    """
    normalised_beam: shape (N_freqs, N_directions) - Beam response normalized by Omega_p
        dType: np.ndarray (real or complex with zero imaginary part)
    d_Omega: float, solid angle element for each direction
    isotropic_signal: bool, if True, assumes isotropic signal and integrates over directions first
        This is just for testing purposes

    Returns:
    quad_Beam: shape (N_freqs, N_freqs)
    """
    if not isotropic_signal:
        # Vectorized computation: sum over directions
        quad_Beam = np.einsum('if,jf->ij', normalised_beam, normalised_beam) * d_Omega
        return quad_Beam
    else:
        normalised_beam_int = np.sum(normalised_beam, axis=1) * d_Omega
        quad_Beam = np.einsum('i,j->ij', normalised_beam_int, normalised_beam_int)
        return quad_Beam

def generate_response_matrix(quad_Fourier_list, quad_Weights, quad_Beam):
    """
    Vectorized version - eliminates the loop entirely
    """
    # Convert list to 3D array: (n_modes, n_freqs, n_freqs)
    quad_Fourier_array = np.array(quad_Fourier_list)
    
    # Broadcast multiplication: aux is (n_freqs, n_freqs), quad_Fourier_array is (n_modes, n_freqs, n_freqs)
    aux = quad_Weights * quad_Beam
    resp_mat_array = aux[None, :, :] * quad_Fourier_array

    return resp_mat_array

def generate_window_func(quad_form_mat, resp_mat_list):
    window = np.array([ trace_AB(quad_form_mat, R) for R in resp_mat_list ])
    norm = np.sum(window)
    return window/norm, norm

def evaluate_bias(quad_form_mat, quad_Weights, norm, noise_cov=None):
    if noise_cov is None:
        return 0.0
    weighted_Ncov = noise_cov * quad_Weights
    bias = trace_AB(quad_form_mat, weighted_Ncov) / norm
    return bias

class QE_window:
    def __init__(self, 
                 freqs, 
                 normalised_beam=test_beam, 
                 d_Omega=test_d_Omega, 
                 ):
    
        df = np.median(np.diff(freqs))
        self.Ndlys = freqs.size
        delays = np.fft.fftshift(np.fft.fftfreq(self.Ndlys, d=df)) # Delay coordinates, in sec
        d_eta = np.median(np.diff(delays))

        self.quad_Fourier_list =get_quad_Fourier_list(self.Ndlys, d_eta)
        self.quad_Beam = generate_quad_Beam(normalised_beam, d_Omega=d_Omega)

    def generate_window_coeffs(self, weight1, weight2, quad_form="Q"):
        """
        weight1, weight2: shape (N_freqs,)
        quad_form: "Q" or "Q_alt"
        """
        quad_Weights = np.einsum('i,j->ij', weight1, weight2.conj())
        resp_mat_arr = generate_response_matrix(self.quad_Fourier_list, quad_Weights, self.quad_Beam)
        
        if quad_form == "Q":
            quad_form_matrices = resp_mat_arr
        elif quad_form == "Q_alt":
            quad_form_matrices = self.quad_Fourier_list
        wind_coeffs_li = []
        for i in range(self.Ndlys):
            wind_coeffs, norm = generate_window_func(quad_form_matrices[i], resp_mat_arr)
            wind_coeffs_li.append(wind_coeffs)
        return np.array(wind_coeffs_li)

def apodize_around_flags(flag_arr, ramp_width):
    """
    flag_arr : boolean array (True = flagged, False = good)
    ramp_width : integer number of samples for the half-cosine ramp on each side
    returns apod : float array in [0,1] same length as flag_arr
    """
    mask = ~flag_arr  # Convert to True=good, False=flagged
    n = len(mask)
    apod = mask.astype(float).copy()

    # find contiguous flagged segments
    diffs = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.where(diffs == -1)[0]  # start index of flagged seg
    ends   = np.where(diffs == 1)[0]   # end index (exclusive) of flagged seg

    for s, e in zip(starts, ends):
        # left ramp: from s-ramp_width .. s-1 rises from 1 -> 0
        L = ramp_width
        if L > 0:
            left0 = max(0, s - L)
            left_len = s - left0
            if left_len > 0:
                t = np.arange(left_len) / left_len  # 0..1
                apod[left0:s] *= 0.5*(1 + np.cos(np.pi * (t)))  # half-cosine taper

            # right ramp: from e .. e+L-1 falls from 0 -> 1
            right1 = min(n, e + L)
            right_len = right1 - e
            if right_len > 0:
                t = np.arange(right_len) / right_len  # 0..1
                apod[e:right1] *= 0.5*(1 - np.cos(np.pi * (t)))  # inverted half-cosine

        # set flagged region to 0
        apod[s:e] = 0.0

    # small numerical safety
    apod = np.clip(apod, 0.0, 1.0)
    return apod

def taper_apodised_flags(flag_arr, ramp_width, taper_coeffs=None):
    """
    Combines apodization around flags with a global tapering window.
    flag_arr : boolean array (True = flagged, False = good)
    ramp_width : integer number of samples for the half-cosine ramp on each side
    taper_coeffs : optional float array in [0,1] same length as flag_arr
                    if None, uses Blackman-Harris window
    returns apod_taper : float array in [0,1] same length as flag_arr
    """
    n = len(flag_arr)
    if taper_coeffs is None:
        import uvtools.dspec as dspec
        taper_coeffs = dspec.gen_window('blackman-harris', n)

    apod = apodize_around_flags(flag_arr, ramp_width)

    apod_taper = apod * taper_coeffs

    # small numerical safety
    apod_taper = np.clip(apod_taper, 0.0, 1.0)
    return apod_taper


class nonuniform_pspec:
    def __init__(self, freqs, normalised_beam=test_beam,
                 d_Omega=test_d_Omega, taper='blackman-harris',
                 test_with_isotropic_signal=True
                 ):
        self.Nfreqs = freqs.size
        self.Ndlys = freqs.size

        self.taper = taper
        if self.taper == 'none':
            self.taper_func = np.ones(self.Nfreqs)
        else:
            self.taper_func = dspec.gen_window(self.taper, self.Nfreqs)

        df = np.median(np.diff(freqs))
        
        delays = np.fft.fftshift(np.fft.fftfreq(self.Ndlys, d=df)) # Delay coordinates, in sec
        self.d_eta = np.median(np.diff(delays))

        self.quad_Fourier_list =get_quad_Fourier_list(self.Ndlys, self.d_eta)
        if normalised_beam is None or d_Omega is None:
            self.quad_Beam = np.ones((self.Nfreqs, self.Nfreqs))
        else:
            self.quad_Beam = generate_quad_Beam(normalised_beam, d_Omega=d_Omega, isotropic_signal=test_with_isotropic_signal)

    def p_hat(self, vis1, vis2, flag1, flag2, N_cov=None, weight1=None, weight2=None, ramp_width=20, quad_form="Q"):
        """
        weight1, weight2: shape (N_freqs,)
        quad_form: "Q" or "Q_alt" or provided matrices
        """
        N_cov = N_cov

        if weight1 is None:
            weight1 = taper_apodised_flags(flag1, ramp_width, taper_coeffs=self.taper_func)
        if weight2 is None:
            weight2 = taper_apodised_flags(flag2, ramp_width, taper_coeffs=self.taper_func)

        weighted_vis1 = vis1 * weight1
        weighted_vis2 = vis2 * weight2
        Dmat = np.einsum('i,j->ij', weighted_vis1, weighted_vis2.conj())

        quad_Weights = np.einsum('i,j->ij', weight1, weight2.conj())
        self.resp_mat_arr = generate_response_matrix(self.quad_Fourier_list, quad_Weights, self.quad_Beam)
        
        
        if quad_form == "Q":
            quad_form_matrices = self.resp_mat_arr
        elif quad_form == "Q_alt":
            quad_form_matrices = self.quad_Fourier_list
        else: 
            quad_form_matrices = quad_form # Assume user provided valid matrices
        wind_coeffs_li = []
        norm_li = []
        bias_li = []
        p_hat_li = []
        for i in range(self.Ndlys):
            wind_coeffs, norm_factor = generate_window_func(quad_form_matrices[i], self.resp_mat_arr)
            wind_coeffs_li.append(wind_coeffs)
            bias = evaluate_bias(quad_form_matrices[i], quad_Weights, norm_factor, noise_cov=N_cov)
            p_hat_i = trace_AB(quad_form_matrices[i], Dmat) / norm_factor - bias
            p_hat_li.append(p_hat_i)
            bias_li.append(bias)
            norm_li.append(norm_factor)

        return np.array(wind_coeffs_li), np.array(bias_li), np.array(norm_li), np.array(p_hat_li)


def calculate_p_hat_covariance(bps, resp_mat_arr, norm_li, weighted_noise_cov=None):
    """

    bps: array of bandpowers, shape (Ndlys,)
    resp_mat_arr: array of response matrices, shape (Ndlys, Nfreqs, Nfreqs)
    norm
    weighted_noise_cov: optional weighted noise covariance matrix, shape (Nfreqs, Nfreqs)
    returns var_p: covariance matrix of bandpower estimates, shape (Ndlys, Ndlys)
    """
    Ndlys = resp_mat_arr.shape[0]
    if weighted_noise_cov is not None:
        # Initialize data_cov as deep copy of weighted_noise_cov
        data_cov = weighted_noise_cov.copy()
    else:
        data_cov = np.zeros((Ndlys, Ndlys), dtype=np.complex128)
    for i in range(Ndlys):
        bandpower = bps[i].real
        data_cov += bandpower * resp_mat_arr[i]
    E_arr = resp_mat_arr / norm_li[:, None, None]
    C_E_arr = np.einsum('ajk,kl->ajl', E_arr, data_cov) 
    Tr_C_E_C_E = np.einsum('ajl,blj->ab', C_E_arr, C_E_arr)
    var_p = 2 * Tr_C_E_C_E 
    # Var(p_alpha, p_beta) = 2 * tr(C E_alpha C E_beta)
    return var_p