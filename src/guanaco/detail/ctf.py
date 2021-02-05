#
# This file is part of guanaco-ctf.
# Copyright 2021 Diamond Light Source
# Copyright 2021 Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# guanaco-ctf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# guanaco-ctf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with guanaco-ctf. If not, see <http:// www.gnu.org/licenses/>.
#
import numpy
from cmath import exp
from math import sqrt, pi

__all__ = ["ctf1d", "ctf2d"]


def get_electron_wavelength(V):
    """
    Compute the electron wavelength

    Args:
        V (float): energy in electron volts

    Returns:
        float: Wavelength (A)

    """
    from scipy.constants import c
    from scipy.constants import e
    from scipy.constants import h
    from scipy.constants import m_e

    return h * c / sqrt((e * V) ** 2 + 2 * m_e * e * V * c ** 2) * 1e10


def get_defocus_spread(Cc, dEE, dII, dVV):
    """
    From equation 3.41 in Kirkland: Advanced Computing in Electron Microscopy

    The dE, dI, dV are the 1/e half widths or E, I and V respectively

    Args:
        Cc (float): The chromatic aberration (A)
        dEE (float): dE/E, the fluctuation in the electron energy
        dII (float): dI/I, the fluctuation in the lens current
        dVV (float): dV/V, the fluctuation in the acceleration voltage

    Returns:
        float: The defocus spread

    """
    return Cc * sqrt((dEE) ** 2 + (2 * dII) ** 2 + (dVV) ** 2)


def spatial_incoherence_envelope(
    q,
    defocus=0,
    spherical_aberration=0,
    wavelength=1,
    defocus_spread=0,
    source_spread=0,
):
    """
    Compute the spatial incoherence as equation 10.53 in De Graef

    Args:
        q (array): The array of spatial frequencies to evaluate the CTF (1/A)
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (rad)

    Returns:
        array: The spatial incoherence envelope evaluated at q


    """
    l = wavelength  # Wavelength (A)
    df = defocus  # Defocus (A)
    Cs = spherical_aberration  # Cs (A)
    theta_c = source_spread  # Source spread (rad)
    d = defocus_spread  # Defocus spread (A)
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q ** 2
    return numpy.exp(
        -((pi * theta_c) ** 2) / (l ** 2 * u) * (Cs * l ** 3 * q ** 3 + df * l * q) ** 2
    )


def temporal_incoherence_envelope(q, wavelength=1, defocus_spread=0, source_spread=0):
    """
    Compute the temporal incoherence envelope as equation 10.53 in De Graef

    Args:
        q (array): The array of spatial frequencies to evaluate the CTF (1/A)
        wavelength (float): The electron wavelength (A)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (rad)

    Returns:
        array: The temporal incoherence envelope evaluated at q


    """
    l = wavelength  # Wavelength (A)
    theta_c = source_spread  # Source spread (rad)
    d = defocus_spread  # Defocus spread (A)
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q ** 2
    return numpy.exp(-((pi * l * d) ** 2) * q ** 4 / (2 * u))


def chi(q, defocus=0, spherical_aberration=0, wavelength=1):
    """
    Compute chi as in Equation 10.9 in De Graef

    Args:
        q (array): The array of spatial frequencies to evaluate the CTF (1/A)
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)

    Returns:
        array: The CTF evaluated at q


    """
    l = wavelength  # Wavelength (A)
    df = defocus  # Defocus (A)
    Cs = spherical_aberration  # Cs (A)
    return pi * (Cs * l ** 3 * q ** 4 / 2 - l * df * q ** 2)


def q_to_Q(q, spherical_aberration=0, wavelength=1):
    """
    Convert spatial frequencies to dimensionless quantities

    From Equation 10.10 in DeGraef

    Args:
        q (array): The array of spatial frequencies  (1/A)
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)

    Returns:
        array: The dimensionless spatial frequencies

    """
    l = wavelength  # Wavelength (A)
    Cs = spherical_aberration  # Cs (A)
    return q * (Cs * l ** 3) ** (1.0 / 4.0)


def Q_to_q(Q, spherical_aberration=0, wavelength=1):
    """
    Get spatial frequencies from dimensionless quantities

    From Equation 10.10 in DeGraef

    Args:
        Q (array): The array of dimensionless spatial frequencies
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)

    Returns:
        array: The spatial frequencies (1/A)

    """
    l = wavelength  # Wavelength (A)
    Cs = spherical_aberration  # Cs (A)
    return Q / (Cs * l ** 3) ** (1.0 / 4.0)


def df_to_D(df, spherical_aberration=0, wavelength=1):
    """
    Convert defocus to dimensionless quantities

    From Equation 10.11 in DeGraef

    Args:
        df (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)

    Returns:
        array: The dimensionless spatial frequencies

    """
    l = wavelength  # Wavelength (A)
    Cs = spherical_aberration  # Cs (A)
    return df / sqrt(Cs * l)


def D_to_df(D, spherical_aberration=0, wavelength=1):
    """
    Get defocus from dimensionless quantities

    From Equation 10.11 in DeGraef

    Args:
        D (float): The dimensionless defocus
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)

    Returns:
        array: The spatial frequencies

    """
    l = wavelength  # Wavelength (A)
    Cs = spherical_aberration  # Cs (A)
    return D * sqrt(Cs * l)


def ctf1d(
    q,
    defocus=0,
    spherical_aberration=0,
    wavelength=1,
    defocus_spread=0,
    source_spread=0,
):
    """
    Compute the CTF

    The defocus is positive for underfocus

    Args:
        q (array): The array of spatial frequencies  (1/A)
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in A)
        wavelength (float): The electron wavelength (A)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (rad)

    Returns:
        array: The CTF evaluated at q


    """

    # Get the things we need to compute the CTF
    l = wavelength  # Wavelength (A)
    df = defocus  # Defocus (A)
    Cs = spherical_aberration  # Cs (A)
    theta_c = source_spread  # Source spread (rad)
    d = defocus_spread  # Defocus spread (A)

    # Compute chi as Equation 10.9 in De Graef
    chi = pi * (Cs * l ** 3 * q ** 4 / 2 - l * df * q ** 2)

    # Compute the spatial and temporal coherence envelopes as in equation 10.53
    # in De Graef
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q ** 2
    Et = numpy.exp(-((pi * l * d) ** 2) * q ** 4 / (2 * u))
    Es = numpy.exp(
        -((pi * theta_c) ** 2) / (l ** 2 * u) * (Cs * l ** 3 * q ** 3 + df * l * q) ** 2
    )
    A = 1 / numpy.sqrt(u)

    # Compute the CTF
    return A * Es * Et * numpy.exp(-1j * chi)


def ctf2d(
    shape=None,
    pixel_size=1,
    defocus=0,
    spherical_aberration=2.7,
    wavelength=None,
    energy=300,
    defocus_spread=None,
    source_spread=0.02,
    chromatic_abberation=2.7,
    energy_spread=0.33e-6,
    current_spread=0.33e-6,
    acceleration_voltage_spread=0.8e-6,
    phase_shift=0,
    centre=False,
):
    """
    Simulate the CTF

    Args:
        shape (tuple): The size of the image
        pixel_size (float): The size of a pixel
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in mm)
        wavelength (float): The electron wavelength (A)
        energy (float): The electron energy (keV)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (mrad)
        Cc (float): The chromatic abberationa (mm)
        dEE (float): dE/E, the fluctuation in the electron energy
        dII (float): dI/I, the fluctuation in the lens current
        dVV (float): dV/V, the fluctuation in the acceleration voltage
        phase_shift (float): The phase shift (rad) - 0 without phase plate, pi/2 with phase plate

    """
    # Convert some quantities
    source_spread = source_spread / 1e3  # mrad -> rad
    spherical_aberration = spherical_aberration * 1e7  # mm -> A
    chromatic_abberation = chromatic_abberation * 1e7  # mm -> A
    energy = energy * 1e3  # keV -> eV

    # Compute the defocus spread
    if defocus_spread is None:
        defocus_spread = get_defocus_spread(
            Cc=chromatic_abberation,
            dEE=energy_spread,
            dII=current_spread,
            dVV=acceleration_voltage_spread,
        )

    # Compute the wavelength
    if wavelength is None:
        assert energy is not None
        wavelength = get_electron_wavelength(energy)

    # Generate the spatial frequencies (1/A)
    assert len(shape) == 2
    Y, X = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    Y = (1 / pixel_size) * (Y - shape[0] // 2) / shape[0]
    X = (1 / pixel_size) * (X - shape[1] // 2) / shape[1]
    q = numpy.sqrt(X ** 2 + Y ** 2)

    # Evaluate the ctf
    ctf = ctf1d(
        q,
        defocus=defocus,
        spherical_aberration=spherical_aberration,
        wavelength=wavelength,
        defocus_spread=defocus_spread,
        source_spread=source_spread,
    )

    # Add a phase shift
    ctf *= exp(1j * phase_shift)

    # Shift
    if not centre:
        ctf = numpy.fft.fftshift(ctf)

    # Return the ctf
    return ctf
