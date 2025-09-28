import hera_pspec as hp
# beam_hera = hp.PSpecBeamUV("/lustre/aoc/projects/hera/H4C/beams/NF_HERA_Vivaldi_efield_beam_healpix_pstokes.fits")
import numpy as np
from scipy import special, integrate
import uvtools.dspec as dspec
from hera_pspec import uvwindow, conversions
from astropy import constants
import copy

class simple_pspec:
    def __init__(self, vis1, vis2, nsample1, nsample2, 
                 freqs, beamfunc, beamtype="healpix",
                 cosmo=None, little_h=True,
                 vis_unit='mK', taper='blackman-harris'):
        self.spw_Nfreqs = freqs.size
        self.spw_Ndlys  = freqs.size
        self.x1 = copy.deepcopy(vis1)
        self.x2 = copy.deepcopy(vis2)
        self.w1 = nsample1
        self.w2 = nsample2
        self.freqs = freqs  
        df = np.median(np.diff(self.freqs))
        self.delays = np.fft.fftshift(np.fft.fftfreq(self.spw_Ndlys, d=df)) # Delay coordinates, in sec
        
        self.taper = taper
        self.Q_alt = {}
        if self.taper == 'none':
            self.taper_func = np.ones(self.spw_Nfreqs)
        else:
            self.taper_func = dspec.gen_window(self.taper, self.spw_Nfreqs)
            
        if beamtype == "healpix":
            raise NotImplementedError
            
        elif beamtype == "pspec_beam":
            if type(beamfunc) == hp.pspecbeam.PSpecBeamUV:
                _beam, beam_omega, N = \
                beamfunc.beam_normalized_response(pol='pI', freq=self.freqs)
            elif type(beamfunc) == tuple:
                assert len(beamfunc) == 3, "Invalid beam function"
                _beam, beam_omega, N = beamfunc
            else:
                raise ValueError("Invalid beam function")
            
            self.omega_p = beam_omega.real
            print("Using beam with omega_p = ", self.omega_p)
            print("Using beam with norm = ", np.sum(_beam**2, axis=-1).real)


            self.omega_pp = np.sum(_beam**2, axis=-1).real*np.pi/(3.*N*N)

            _beam = _beam/self.omega_p[:, None]
            print("Using beam with omega_pp = ", self.omega_pp)
            
            self.qnorm_exact = np.pi/(3.*N*N) * np.dot(_beam, _beam.T)
            self.qnorm_exact *= np.median(np.diff(self.delays))
            
        elif beamtype == "azimuthal":
            _th = np.linspace(0, np.pi/2, 500) #Only above the ground
            _beam = beamfunc(_th, self.freqs)
            self.omega_p = 2*np.pi*integrate.simpson(_beam*np.sin(_th), x=_th, axis=-1)
            self.omega_pp = 2*np.pi*integrate.simpson(_beam**2*np.sin(_th), x=_th, axis=-1)

            _beam = _beam/self.omega_p[:, None]
            self.qnorm_exact = 2*np.pi*integrate.simpson(_beam[:, None, :]*_beam[None, :, :]*np.sin(_th), x=_th, axis=-1)
            self.qnorm_exact *= np.median(np.diff(self.delays))
        
        else:
            raise NotImplementedError

        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()
        df = np.median(np.diff(self.freqs))
        integration_freqs = np.linspace(self.freqs.min(),
                                        self.freqs.min() + df*self.spw_Nfreqs,
                                        5000, endpoint=True, dtype=float)
        integration_freqs_MHz = integration_freqs / 1e6

        # Get redshifts and cosmological functions
        redshifts = self.cosmo.f2z(integration_freqs).flatten()
        X2Y = np.array([self.cosmo.X2Y(z, little_h=little_h) for z in redshifts])
        self.scalar = integrate.trapezoid(X2Y, x=integration_freqs)/(np.abs(integration_freqs[-1]-integration_freqs[0]))
        
        if vis_unit == 'Jy':
            c =  constants.c.cgs.value
            k_b =  constants.k_B.cgs.value
            self.Jy2mK = 1e3 * 1e-23 * c**2 / (2 * k_b * self.freqs**2 * self.omega_p)
            self.x1 *= self.Jy2mK
            self.x2 *= self.Jy2mK
        
    def get_R(self):
        ''' Return the R matrix for flagging, tapering, non-uniform weighting, etc.
        '''
        return np.diag(self.taper_func)
        
    def get_Q_alt(self, mode):
        try:
            Q_alt = self.Q_alt[mode]
        except KeyError:
            if self.spw_Ndlys % 2 == 0:
                start_idx = -self.spw_Ndlys/2
            else:
                start_idx = -(self.spw_Ndlys - 1)/2
            m = (start_idx + mode) * np.arange(self.spw_Nfreqs)
            m = np.exp(-2j * np.pi * m / self.spw_Ndlys)
            Q_alt = np.einsum('i,j', m.conj(), m) # dot it with its conjugate
            self.Q_alt[mode] = Q_alt
            
        return Q_alt 
        
    def get_GH(self, operator=None):
        G = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=complex)
        H = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=complex)
        R = self.get_R()
        
        sinc_matrix = np.zeros((self.spw_Nfreqs, self.spw_Nfreqs))
        for i in range(self.spw_Nfreqs):
            for j in range(self.spw_Nfreqs):
                sinc_matrix[i,j] = float(i - j)
        sinc_matrix = np.sinc(sinc_matrix / float(self.spw_Nfreqs))
        
        iR1Q1, iR2Q2, iR1Q1_win, iR2Q2_win = {}, {}, {}, {}
        for ch in range(self.spw_Ndlys):
            Q_alt = self.get_Q_alt(ch)
            if operator is not None:
                iR1Q1[ch] = np.conj(operator).T@np.conj(R).T@(Q_alt) # O R_1 Q_alt
                iR2Q2[ch] = R@operator@(Q_alt* self.qnorm_exact) # R_2 OQ_true
                iR1Q1_win[ch] = iR1Q1[ch] #np.conj(operator).T@np.conj(R).T@(Q_alt) # O R_1 Q_alt
                iR2Q2_win[ch] = R@operator@(Q_alt* self.qnorm_exact * sinc_matrix) # R_2 O Q_true
            else:
                iR1Q1[ch] = np.conj(R).T@(Q_alt) # R_1 Q_alt
                iR2Q2[ch] = R@(Q_alt * self.qnorm_exact) # R_2 Q_true                
                iR1Q1_win[ch] = iR1Q1[ch] # R_1 Q_alt
                iR2Q2_win[ch] = R@(Q_alt* self.qnorm_exact * sinc_matrix) # R_2 Q_true
            
        for i in range(self.spw_Ndlys):
            for j in range(self.spw_Ndlys):
                # tr(R_2 Q_i R_1 Q_j)
                G[i,j] = np.einsum('ab,ba', iR1Q1[i], iR2Q2[j])  
                H[i,j] = np.einsum('ab,ba', iR1Q1_win[i], iR2Q2_win[j])
        if np.count_nonzero(G) == 0:
            G = np.eye(self.spw_Ndlys)
        if np.count_nonzero(H) == 0:
            H = np.eye(self.spw_Ndlys)            
        self.G = G/2.
        self.H = H/2.
        return G/2., H/2. 

    def get_MW(self, GH=None, operator=None):
        if GH is None:
            if hasattr(self, 'G'):
                G, H = self.G, self.H
            else:
                G, H = self.get_GH(operator)
        else:
            G, H = GH
        M = np.diag(1. / np.sum(G, axis=1)) 
        W_norm = np.diag(1. / np.sum(H, axis=1))
        W = np.dot(W_norm, H)
        return M, W
    
    def p_hat(self, calc_cov=False, Cnos1=None, Cnos2=None, SN=False, s_model=None):
        R = self.get_R()
        Rx1 = np.dot(R, self.x1.T)
        Rx2 = np.dot(R, self.x2.T)
        Q_alt_tensor = np.array([self.get_Q_alt(mode) for mode in range(self.spw_Ndlys)])
        QRx2 = np.dot(Q_alt_tensor, Rx2)
        q = 1/2*np.einsum('i...,ji...->j...', Rx1.conj(), QRx2)
        
        M, W = self.get_MW()
        M *= self.scalar
        p = M@q
        
        if calc_cov:
            if hasattr(self, 'E_matrices'):
                E_matrices = self.E_matrices
            else:
                E_matrices = 1/2*np.diag(np.conj(R).T)[None, :, None]*Q_alt_tensor*np.diag(R)[None, None, :]
                E_matrices = np.einsum('ab, bij->aij', M, E_matrices)
                self.E_matrices = E_matrices

            Cnos1 = Cnos1.astype(np.complex128)
            if Cnos2 is None:
                Cnos2 = Cnos1.astype(np.complex128)
            else:
                Cnos2 = Cnos2.astype(np.complex128)
            
            if hasattr(self, 'einstein_path'):
                einstein_path_0, einstein_path_1, einstein_path_2 = self.einstein_path
            else:
                einstein_path_0 =  np.einsum_path('bij, cji->bc', E_matrices, E_matrices, optimize='optimal')[0]
                einstein_path_1 = np.einsum_path('bi, cj,ij->bc', 
                                                 E_matrices[:,:,0], E_matrices[:,:,0], E_matrices[0,:,:], 
                                                 optimize='optimal')[0]
                einstein_path_2 =  np.einsum_path('ab,cd,bd->ac', M, M, M, optimize='optimal')[0]
                self.einstein_path = (einstein_path_0, einstein_path_1, einstein_path_2)
            
            if SN:
                raise NotImplementedError("P_SN currently not supported")
                '''
                if s_model is None:
                    Csig = 1/2*(self.x1[:, None]*self.x2[None, :].conj() + self.x2[:, None]*self.x1[None, :].conj())
                else:
                    Csig = s_model[:, None]*s_model[None, :].conj()
                E_Csn = np.einsum("aij, jk->aik", E_matrices, (Csig+Cnos))
                E_Csig = np.einsum("aij, jk->aik", E_matrices, Csig)
                p_pdagger = np.einsum('bij, cji->bc', E_Csn, E_Csn, optimize=einstein_path_0)
                p_pdagger -= np.einsum('bij, cji->bc', E_Csig, E_Csig, optimize=einstein_path_0)
                '''
            else:
                E_Cnos1 = np.einsum("aij, jk->aik", E_matrices, Cnos1)
                E_Cnos2 = np.einsum("aij, jk->aik", E_matrices, Cnos2)
                p_pdagger = 1/2*np.einsum('bij, cji->bc', E_Cnos1, E_Cnos2, optimize=einstein_path_0)
            '''
            if SN:
                if s1 is None:
                    s1 = self.x1
                if s2 is None:
                    s2 = self.x2                   
                E12_x1 = np.dot(E_matrices, s1)
                E12_x2 = np.dot(E_matrices, s2)
                x2star_E21 = E12_x2.conj()
                x1star_E21 = E12_x1.conj()
                x1star_E12 = np.dot(np.transpose(E_matrices,(0,2,1)), s1.conj())
                x2star_E12 = np.dot(np.transpose(E_matrices,(0,2,1)), s2.conj())
                E21_x1 = x1star_E12.conj()
                E21_x2 = x2star_E12.conj()
                SN_cov = (np.einsum('bi,cj,ij->bc', E12_x1, x2star_E21, C11, optimize=einstein_path_1)/2. + 
                          np.einsum('bi,cj,ij->bc', E12_x2, x1star_E21, C11, optimize=einstein_path_1)/2. +
                          np.einsum('bi,cj,ij->bc', x2star_E12, E21_x1, C22, optimize=einstein_path_1)/2. + 
                          np.einsum('bi,cj,ij->bc', x1star_E12, E21_x2, C22, optimize=einstein_path_1)/2.
                         )
                # Apply zero clipping on the columns and rows containing negative diagonal elements
                SN_cov[np.real(np.diag(SN_cov))<=0., :] = 0. + 1.j*0
                SN_cov[:, np.real(np.diag(SN_cov))<=0.,] = 0. + 1.j*0
                q_qdagger += SN_cov  
            '''
            return p, p_pdagger
            
        else:
            return p
                
def airy_beam(theta, freqs, a=6):
    
    k = 2*np.pi * freqs / constants.c.value
    k = k[:, np.newaxis]

    arg = k*6*np.sin(theta)

    beam = (2*special.jv(1,arg)/arg)**2
    beam = np.where(np.isnan(beam), 1, beam)

    return beam

