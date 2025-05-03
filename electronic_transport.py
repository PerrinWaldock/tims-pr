import os
import numpy as np
import pandas as pd
import zipfile
# from PyAstronomy.pyasl import generalizedESD
import scipy
from scipy.constants import mu_0, epsilon_0, pi, c
import mpmath as mp
# from scipy.optimize import curve_fit

## The full skin effect response functions are from the paper by Hein, Ormeno, and Gough:
## "High-frequency electrodynamic response of strongly anisotropic clean normal and superconducting metals"
## doi: 10.1103/PhysRevB.64.024529

## Define the constants for the anomalous skin effect regime beta: boundary scattering
## From Graham Baker's PhD thesis, eqn. 1.45, pg. 21.
global beta_specular, beta_diffuse
beta_specular: float = 4/(3*np.sqrt(3))
"""
Constant for the anomalous skin effect regime with specular boundary scattering.

From Graham Baker's PhD thesis, equation 1.45, page 21.

Notes
-----
.. math:: 
    \beta_{specular} = \frac{4}{3\sqrt{3}} \approx 0.7698
"""
beta_diffuse: float = np.sqrt(3)/2
"""
Constant for the anomalous skin effect regime with diffuse boundary scattering.

From Graham Baker's PhD thesis, equation 1.45, page 21.

Notes
-----
.. math::
    \beta_{diffuse} = \frac{\sqrt{3}}{2} \approx 0.8660
"""

# @np.vectorize
def q_prime_func(q: np.ndarray, omega: np.ndarray, mean_free_path_MR: float, rate_scattering_MR: float) -> np.ndarray:
    """
    Calculate the dimensionless wavevector product, that include relaxation effects.

    Used for calculating the conductivity spectrum.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    q : ndarray[n]
        Wavevector [1/m]
    omega : ndarray[m]
        Angular frequency [rad/s]
    mean_free_path_MR : float
        Mean free path l_MR [m]
    rate_scattering_MR : float
        Scattering rate gamma_MR [rad/s]

    Returns
    -------
    ndarray[n, m]
        q * mean_free_path_MR / (1 - i * omega / rate_scattering_MR) [1/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        q' = \frac{q l_{MR}}{1 - i \omega / \gamma_{MR}}
    """
    return np.outer((q*mean_free_path_MR), 1/(1-1j*omega/rate_scattering_MR))

@np.vectorize
def conductivity_relaxation(cond_DC: float, omega: np.ndarray, rate_scattering_MR: float) -> np.ndarray:
    """
    Calculate the frequency-dependent Drude response conductivity of a metal, using a momentum-relaxing scattering rate.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    cond_DC : float
        Conductivity at zero frequency [S/m]
    omega : ndarray(float)[n]
        Angular frequency [rad/s]
    rate_scattering_MR : float
        Momentum-relaxing scattering rate [rad/s]

    Returns
    -------
    ndarray(float)[n]
        Drude conductivity including relaxation effects [S/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        \sigma(\omega) = \frac{\sigma_{DC}}{1 - i \omega / \gamma_{MR}}
    """
    return cond_DC/(1 - 1j*omega/rate_scattering_MR)

@np.vectorize
def nonlocality_term_3D_isotropic_func(q_prime: np.ndarray) -> np.ndarray:
    """
    Term that accounts for nonlocality of a 3D isotropic Fermi surface.

    Used for calculating the conductivity spectrum.

    Equation (3) in Hein et al. PRB 64 024529 (2001).

    Parameters
    ----------
    q_prime : ndarray[n,m]
        q * l_MR / (1 - i * omega / gamma_MR) [1]

    Returns
    -------
    ndarray[n,m]
        Nonlocality term. [1]

    Notes
    -----
    Returns the following expression:
    .. math::
        \frac{3}{2} q'^{-3} \left( (1 + q'^2) \arctan(q') - q' \right)
    """

    ## I'm not sure why it works this way 
    @np.vectorize
    def nonlocality_term_func(q_prime):
        q_function = (3/2) * q_prime**-3 * ((1 + q_prime**2) * mp.atan(q_prime) - q_prime)
        nonlocality_term = complex(q_function)
        return nonlocality_term

    # q_function = (3/2) * q_prime**-3 * ((1 + q_prime**2) * mp.atan(q_prime) - q_prime)
    # nonlocality_term = complex(q_function)
    nonlocality_term = np.array(nonlocality_term_func(q_prime).tolist(), dtype=complex)
    return nonlocality_term

def cond_spectrum_3D_isotropic(q: np.ndarray, omega: np.ndarray, mean_free_path_MR: float, rate_scattering_MR: float, resistivity_residual_DC: float) -> np.ndarray:
    """
    Calculate the conductivity spectrum for a 3D isotropic Fermi surface.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    q : ndarray[n]
        Wavevector [1/m]
    omega : ndarray[m]
        Angular frequency [rad/s]
    mean_free_path_MR : float
        Mean free path l_MR [m]
    rate_scattering_MR : float
        Scattering rate gamma_MR [rad/s]
    resistivity_residual_DC : float
        Residual resistivity [Ohm m]

    Returns
    -------
    cond_spectrum : ndarray[n,m]
        Conductivity spectrum [S/m]
    """
    cond_spectrum_relaxation = conductivity_relaxation(resistivity_residual_DC**-1, omega, rate_scattering_MR)
    q_prime = q_prime_func(q, omega, mean_free_path_MR, rate_scattering_MR)
    nonlocality_term = nonlocality_term_3D_isotropic_func(q_prime)
    return np.multiply(cond_spectrum_relaxation, nonlocality_term)

## Analytic limits

def Z_CSE_limit(freq: np.ndarray, resistivity_residual: float) -> np.ndarray:
    """
    Surface impedance in the classical skin effect regime.

    Has a negative complex phase, and negative surface reactance.
    
    From Jake Bobowski's PhD thesis, eqn. 3.5, pg. 50 (2010).

    Parameters
    ----------
    freq : ndarray(float)[n]
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]

    Returns
    -------
    Z_CSE : ndarray(complex)[n]
        Surface impedance in the classical skin effect regime [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{CSE} = \sqrt{-1j \mu_0 \omega \rho_{residual}}
    """
    omega = 2*pi*freq

    return np.sqrt(-1j*mu_0*omega*resistivity_residual)


def Z_ASE_limit(freq: np.ndarray, freq_plasma: float, velocity_Fermi: float, surface_scattering_type: str) -> np.ndarray:
    """
    Surface impedance in the anomalous skin effect regime, for either specular or diffuse surface scattering.

    Has a negative complex phase, and negative surface reactance.
    
    From Graham Baker's PhD thesis, eqn. 1.44, pg. 21 (2022).

    Parameters
    ----------
    freq : ndarray(float)[n]
        Frequency [Hz]
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity [m/s]
    surface_scattering_type : str('specular' or 'diffuse')
        Surface scattering type

    Returns
    -------
    Z_ASE : ndarray(complex)[n]
        Surface impedance in the anomalous skin effect regime [Ohm]

    Raises
    ------
    ValueError
        If the surface scattering type is not 'specular' or 'diffuse'.
        
    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{ASE} = \beta \mu_0 \left( \frac{4 \lambda_{London}^2 velocity_Fermi}{3 \pi} \right)^{1/3} \omega^{2/3} e^{-i \pi/3}
    """

    omega = 2*pi*freq

    skin_depth_London = c/freq_plasma
    
    if surface_scattering_type == 'specular':
        beta = beta_specular
    elif surface_scattering_type == 'diffuse':
        beta = beta_diffuse
    else:
        raise ValueError("Invalid surface scattering type")

    return beta*mu_0*((4*skin_depth_London**2*velocity_Fermi)/(3*pi))**(1/3)*omega**(2/3)*np.exp(-1j*pi/3)

def Z_relaxation_limit(freq: np.ndarray, freq_plasma: float) -> np.ndarray:
    """
    Surface impedance in the relaxation regime, to the leading order., so has no surface resistance.

    Has a negative complex phase, and negative surface reactance.
    
    From Graham Baker's PhD thesis, eqn. 1.44, pg. 21.

    Parameters
    ----------
    freq : ndarray(float)[n]
        Frequency [Hz]
    freq_plasma : float
        Plasma frequency [Hz]

    Returns
    -------
    Z_relaxation : ndarray(complex)[n]
        Surface impedance in the relaxation regime limit [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{relaxation} = \mu_0 \lambda_{London} \omega e^{-i \pi/2}
    """
    omega = 2*pi*freq

    skin_depth_London = c/freq_plasma

    return mu_0*skin_depth_London*omega*np.exp(-1j*pi/2)

def Z_relaxation_local(freq: np.ndarray, resistivity_residual: float, rate_scattering_MR: float) -> np.ndarray:
    """ 
    Surface impedance in the local transport regime. Captures the transition from the classical skin effect regime to the relaxation regime.

    Has a negative complex phase, and negative surface reactance.

    Parameters
    ----------
    freq : ndarray(float)[n]
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]
    rate_scattering_MR : float
        Analytic scattering rate [rad/s]

    Returns
    -------
    ndarray(complex)[n]
        Surface impedance in the local transport regime [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{local} = \sqrt{-1j \mu_0 \omega / \sigma_{local}}
    """

    omega = 2*pi*freq

    return np.sqrt(-1j*mu_0*omega/conductivity_relaxation(resistivity_residual**-1, omega, rate_scattering_MR))


def get_skin_effect_regime_limits(freqs: np.ndarray, resistivity_residual: float, freq_plasma: float, velocity_Fermi: float, rate_scattering_MR: float) -> dict:
    """
    Provides the limits of the surface impedance for each skin effect regime.

    Included: CSE, ASE (specular and diffuse), Relaxation, Local transport (Drude response).

    Parameters
    ----------
    freqs : ndarray(float)[n]
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity in the direction of the current [m/s]
    rate_scattering_MR : float
        Momentum-relaxing scattering rate [rad/s]

    Returns
    -------
    limits : dict[str, ndarray(complex)[n]]
        Dictionary with keys as regime names and values as surface impedance [Ohm]
        keys include:
            "CSE": Classical Skin Effect
            "ASE (specular surface scattering)": Anomalous Skin Effect, specular surface scattering
            "ASE (diffuse surface scattering)": Anomalous Skin Effect, diffuse surface scattering
            "Relaxation": Relaxation
            "Local transport (Drude response)": Local transport (Drude response)
    """

    limits = {
        "CSE": Z_CSE_limit(freqs, resistivity_residual),
        "ASE (specular surface scattering)": Z_ASE_limit(freqs, freq_plasma, velocity_Fermi, 'specular'),
        "ASE (diffuse surface scattering)": Z_ASE_limit(freqs, freq_plasma, velocity_Fermi, 'diffuse'),
        "Relaxation": Z_relaxation_limit(freqs, freq_plasma),
        "Local transport (Drude response)": Z_relaxation_local(freqs, resistivity_residual, rate_scattering_MR)
    }
    return limits


def frequency_boundary_CSE_to_relaxation(rate_scattering_MR):
    """
    Frequency at which the CSE to relaxation boundary occurs.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Parameters
    ----------
    rate_scattering_MR : float
        Momentum-relaxing scattering rate [rad/s]

    Returns
    -------
    float
        Frequency [Hz]

    Notes
    -----
    Returns the following expression:
    .. math::
        f_{CSE, relaxation} = \frac{\gamma_{MR}}{2\pi}
    """

    return rate_scattering_MR/(2*pi)

def frequency_boundary_CSE_to_ASE_length_scale(freq_plasma, velocity_Fermi, mean_free_path_MR):
    """
    Frequency at which the CSE to ASE boundary occurs, based on 
    the momentum-relaxing mean free path being equal to the classical skin depth.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Slight adjustment made, adding a factor of 2 from the table.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity v_F [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path l_MR [m]

    Returns
    -------
    float
        Frequency boundary between CSE and ASE regimes [Hz]
        
    """
    skin_depth_London = c/freq_plasma

    return (1/pi)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_CSE_to_ASE_Rs_specular(freq_plasma, velocity_Fermi, mean_free_path_MR):
    """
    Frequency at which the CSE to ASE boundary occurs, based on the surface resistance being equal
    in the CSE limit and the specular surface scattering ASE limit.

    Calculated from the limits in eqn 1.44 in Graham Baker's PhD thesis.
    
    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    float
        Frequency [Hz]
    """

    skin_depth_London = c/freq_plasma

    return (1/(2*pi))*((2/beta_specular**2)**3*(3*pi/4)**2)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_CSE_to_ASE_Rs_diffuse(freq_plasma, velocity_Fermi, mean_free_path_MR):
    """
    Frequency at which the CSE to ASE boundary occurs, based on the surface resistance being equal
    in the CSE limit and the diffuse surface scattering ASE limit.

    Calculated from the limits in eqn 1.44 in Graham Baker's PhD thesis.
    
    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    float
        Frequency [Hz]
    """

    skin_depth_London = c/freq_plasma

    return (1/(2*pi))*((2/beta_diffuse**2)**3*(3*pi/4)**2)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_ASE_to_Anomalous_Reflection_rate_scale(freq_plasma, velocity_Fermi, mean_free_path_MR):
    """
    Frequency at which the ASE to the anomalous reflection regime occurs,
    based on scattering rates.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    float
        Frequency [Hz]

    Notes
    -----
    Returns the following expression:
    .. math::
        f_{ASE, anomalous reflection} = \frac{v_F}{2\pi \lambda_{London}}
    """

    skin_depth_London = c/freq_plasma

    return (velocity_Fermi/skin_depth_London)*(1/(2*pi))

def get_skin_effect_regime_frequency_boundaries(freq_plasma, velocity_Fermi, mean_free_path_MR):
    """
    Returns all frequency boundaries in a dictionary.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F)[m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    dict
        Dictionary with keys as boundary names and values as frequencies [Hz]
        keys include:
            "CSE, Relaxation (scattering rate)": CSE to relaxation boundary
            "CSE, ASE (l_MR = delta)": CSE to ASE boundary based on l_MR = delta
            "CSE, ASE (Rs_CSE = Rs_ASE_diffuse)": CSE to ASE boundary based on surface resistance
            "CSE, ASE (Rs_CSE = Rs_ASE_specular)": CSE to ASE boundary based on surface resistance
            "ASE, Anomalous Reflection (scattering rate)": ASE to anomalous reflection boundary
    """

    gamma_MR = velocity_Fermi/mean_free_path_MR
    boundaries = {
        "CSE, Relaxation (scattering rate)": frequency_boundary_CSE_to_relaxation(gamma_MR),
        "CSE, ASE (l_MR = delta)": frequency_boundary_CSE_to_ASE_length_scale(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "CSE, ASE (Rs_CSE = Rs_ASE_diffuse)": frequency_boundary_CSE_to_ASE_Rs_diffuse(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "CSE, ASE (Rs_CSE = Rs_ASE_specular)": frequency_boundary_CSE_to_ASE_Rs_specular(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "ASE, Anomalous Reflection (scattering rate)": frequency_boundary_ASE_to_Anomalous_Reflection_rate_scale(freq_plasma, velocity_Fermi, mean_free_path_MR)
    }
    return boundaries



def conductivity_spectrum_Fermi_surface_properties(q: np.ndarray, omega: np.ndarray, conductivity_spectrum: np.ndarray) -> tuple:
    """
    Calculate the Fermi velocity, plasma frequency, and other properties from the conductivity spectrum.

    Only valid in the limit of high wavevector magnitude, q l_MR >> 1.

    Parameters
    ----------
    q : ndarray(float)[n]
        Wavevector [1/m]
    omega : ndarray(float)[m]
        Angular frequency [rad/s]
    conductivity_spectrum : ndarray(complex)[n,m]
        Conductivity spectrum [S/m]

    Returns
    -------
    tuple
        Fermi velocity, plasma frequency, and other properties (See Notes)

    Notes
    -----
    Returns the following expressions, for each q_0, omega_0, and sigma(q_0, omega_0):
    .. math::
        v_F = A/B
        \omega_{plasma} = A/v_F
        A = \epsilon_0 \omega_{p}^2/v_F = (4/(3*\pi))*\Re(\sigma(q_0,\omega_0))*q_0
        B = \epsilon_0*\omega_{p}^2/v_F^2 = \Im(\sigma(q_0,\omega_0))*q_0^2/(3*\omega_0)
    """
    epsilon_freq_plasma_squared_over_vF = (4/(3*pi))*conductivity_spectrum.real*q
    epsilon_freq_plasma_squared_over_vF_squared = conductivity_spectrum.imag*q**2/(3*omega)

    velocity_Fermi_cond_calc = epsilon_freq_plasma_squared_over_vF/epsilon_freq_plasma_squared_over_vF_squared
    epsilon_freq_plasma_squared_cond_calc = epsilon_freq_plasma_squared_over_vF*velocity_Fermi_cond_calc

    freq_plasma_cond_calc = np.sqrt(epsilon_freq_plasma_squared_cond_calc/epsilon_0)

    return velocity_Fermi_cond_calc, freq_plasma_cond_calc, epsilon_freq_plasma_squared_over_vF, epsilon_freq_plasma_squared_over_vF_squared 

## Conductivity spectrum high-q limits.

def cond_spectra_3D_isotropic_high_q_complex(ql: np.ndarray, freq: float, resistivity_residual: float, freq_plasma: float) -> np.ndarray:
    """
    Conductivity spectrum limit for a 3D isotropic Fermi surface at high q.

    Parameters
    ----------
    ql : ndarray(float)[n]
        Wavevector times mean free path [1]
    freq : float
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity (sigma_DC) [Ohm m]
    freq_plasma : float
        Plasma frequency [Hz]

    Returns
    -------
    cond_spectra_high_q_limit : ndarray(complex)[n]
        Conductivity spectrum [S/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        \Re(\sigma) = \sigma_{DC} \frac{3\pi}{4\epsilon_0 (q \ell_{MR})}
        \Im(\sigma) = 3 \sigma_{DC} \left( \frac{\gamma_{MR}}{\omega} \right) \frac{1}{(q \ell_{MR})^2}
    """

    gamma_MR = epsilon_0*freq_plasma**2*resistivity_residual
    omega = 2*pi*freq

    real_part = np.multiply(resistivity_residual**-1,(3*pi/(4*ql)))
    imag_part = np.multiply(resistivity_residual**-1*3*(omega/gamma_MR),ql**-2)

    return real_part + 1j*imag_part
