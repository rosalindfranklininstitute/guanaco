import numpy
import pytest
import guanaco.detail
from cmath import exp
from math import sqrt, pi


def get_electron_wavelength_py(V):
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

    return h * c / sqrt((e * V) ** 2 + 2 * m_e * e * V * c**2) * 1e10


def get_defocus_spread_py(Cc, dEE, dII, dVV):
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
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q**2
    return numpy.exp(
        -((pi * theta_c) ** 2) / (l**2 * u) * (Cs * l**3 * q**3 + df * l * q) ** 2
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
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q**2
    return numpy.exp(-((pi * l * d) ** 2) * q**4 / (2 * u))


def get_chi_py(q, defocus=0, spherical_aberration=0, wavelength=1):
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
    return pi * (Cs * l**3 * q**4 / 2 - l * df * q**2)


def q_to_Q_py(q, spherical_aberration=0, wavelength=1):
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
    return q * (Cs * l**3) ** (1.0 / 4.0)


def Q_to_q_py(Q, spherical_aberration=0, wavelength=1):
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
    return Q / (Cs * l**3) ** (1.0 / 4.0)


def df_to_D_py(df, spherical_aberration=0, wavelength=1):
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


def D_to_df_py(D, spherical_aberration=0, wavelength=1):
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
    chi = pi * (Cs * l**3 * q**4 / 2 - l * df * q**2)

    # Compute the spatial and temporal coherence envelopes as in equation 10.53
    # in De Graef
    u = 1 + 2 * (pi * theta_c * d) ** 2 * q**2
    Et = numpy.exp(-((pi * l * d) ** 2) * q**4 / (2 * u))
    Es = numpy.exp(
        -((pi * theta_c) ** 2) / (l**2 * u) * (Cs * l**3 * q**3 - df * l * q) ** 2
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
        defocus_spread = get_defocus_spread_py(
            Cc=chromatic_abberation,
            dEE=energy_spread,
            dII=current_spread,
            dVV=acceleration_voltage_spread,
        )

    # Compute the wavelength
    if wavelength is None:
        assert energy is not None
        wavelength = get_electron_wavelength_py(energy)

    # Generate the spatial frequencies (1/A)
    assert len(shape) == 2
    Y, X = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    Y = (1 / pixel_size) * (Y - shape[0] // 2) / shape[0]
    X = (1 / pixel_size) * (X - shape[1] // 2) / shape[1]
    q = numpy.sqrt(X**2 + Y**2)

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


def test_get_electron_wavelength():

    l1 = guanaco.detail.get_electron_wavelength(300000)
    l2 = get_electron_wavelength_py(300000)
    assert l1 == pytest.approx(l2)


def test_get_defocus_spread():

    dd1 = guanaco.detail.get_defocus_spread(2.7 * 1e7, 1e-6, 1e-6, 1e-6)
    dd2 = get_defocus_spread_py(2.7 * 1e7, 1e-6, 1e-6, 1e-6)
    assert dd1 == pytest.approx(dd2)


def test_q_to_Q():
    l = guanaco.detail.get_electron_wavelength(300000)
    q1 = 10
    Q1 = guanaco.detail.q_to_Q(q1, l, 2.7 * 1e7)
    Q2 = q_to_Q_py(q1, 2.7 * 1e7, l)
    assert Q1 == pytest.approx(Q2)
    q2 = guanaco.detail.Q_to_q(Q1, l, 2.7 * 1e7)
    assert q1 == pytest.approx(q2)


def test_df_to_D():
    l = guanaco.detail.get_electron_wavelength(300000)
    df1 = 20000
    D1 = guanaco.detail.df_to_D(df1, l, 2.7 * 1e7)
    D2 = df_to_D_py(df1, 2.7 * 1e7, l)
    assert D1 == pytest.approx(D2)
    df2 = guanaco.detail.D_to_df(D1, l, 2.7 * 1e7)
    assert df1 == pytest.approx(df2)


def test_get_Es():
    l = guanaco.detail.get_electron_wavelength(300000)
    q = 10
    config = guanaco.detail.CTF(l, 20000, 2.7 * 1e7, 0, 0, 1, 1e-3, 0)
    Es1 = config.get_Es([q], [0])[0]
    Es2 = spatial_incoherence_envelope(q, 20000, 2.7 * 1e7, l, 1, 1e-3)
    assert Es1 == pytest.approx(Es2)


def test_get_Et():
    l = guanaco.detail.get_electron_wavelength(300000)
    q = 10
    config = guanaco.detail.CTF(l, 20000, 2.7 * 1e7, 0, 0, 1, 1e-3, 0)
    Et1 = config.get_Et([q])[0]
    Et2 = temporal_incoherence_envelope(q, l, 1, 1e-3)
    assert Et1 == pytest.approx(Et2)


def test_get_chi():
    l = guanaco.detail.get_electron_wavelength(300000)
    q = 10
    config = guanaco.detail.CTF(l, 20000, 2.7 * 1e7, 0, 0, 0, 0, 0)
    chi1 = config.get_chi(q, 0)
    chi2 = get_chi_py(q, 20000, 2.7 * 1e7, l)
    assert chi1 == pytest.approx(chi2)


def test_get_q_and_theta():
    q1, a1 = guanaco.detail.get_q_and_theta(100, 100, 2)
    y, x = numpy.mgrid[0:100, 0:100]
    y -= 50
    x -= 50
    y = (y / 100) / 2
    x = (x / 100) / 2
    q2 = numpy.fft.fftshift(numpy.sqrt(x * x + y * y))
    a2 = numpy.fft.fftshift(numpy.arctan2(y, x))
    assert pytest.approx(numpy.abs(q1 - q2).max(), abs=1e-5) == 0
    assert pytest.approx(numpy.abs(a1 - a2).max(), abs=1e-5) == 0


def test_get_ctf():

    w = 100
    h = 100
    ps = 2
    l = guanaco.detail.get_electron_wavelength(300000)
    df = 20000
    Cs = 2.7 * 1e7
    dd = 1
    theta_c = 0.1
    config = guanaco.detail.CTF(l, df, Cs, 0, 0, dd, theta_c / 1e3, 0.1)
    ctf1 = config.get_ctf(w, h, ps)
    ctf2 = ctf2d((h, w), ps, df, Cs / 1e7, l, 300, dd, theta_c, phase_shift=0.1)
    assert pytest.approx(numpy.abs(ctf1 - ctf2).max(), abs=1e-5) == 0


def test_get_ctf_simple():
    w = 100
    h = 100
    ps = 2
    l = guanaco.detail.get_electron_wavelength(300000)
    df = 20000
    Cs = 2.7 * 1e7
    config = guanaco.detail.CTF(l, df, Cs, 0, 0, 0, 0, 0.1)
    ctf1 = config.get_ctf_simple(w, h, ps)
    ctf2 = ctf2d((h, w), ps, df, Cs / 1e7, l, 300, 0, 0, phase_shift=0.1)
    assert pytest.approx(numpy.abs(ctf1 - ctf2).max(), abs=1e-5) == 0


def test_get_ctf_simple_real():
    w = 100
    h = 100
    ps = 2
    l = guanaco.detail.get_electron_wavelength(300000)
    df = 20000
    Cs = 2.7 * 1e7
    config = guanaco.detail.CTF(l, df, Cs, 0, 0, 0, 0, 0.1)
    ctf1 = config.get_ctf_simple_real(w, h, ps)
    ctf2 = numpy.real(ctf2d((h, w), ps, df, Cs / 1e7, l, 300, 0, 0, phase_shift=0.1))
    assert pytest.approx(numpy.abs(ctf1 - ctf2).max(), abs=1e-5) == 0


def test_get_ctf_simple_imag():
    w = 100
    h = 100
    ps = 2
    l = guanaco.detail.get_electron_wavelength(300000)
    df = 20000
    Cs = 2.7 * 1e7
    config = guanaco.detail.CTF(l, df, Cs, 0, 0, 0, 0, 0.1)
    ctf1 = config.get_ctf_simple_imag(w, h, ps)
    ctf2 = numpy.imag(ctf2d((h, w), ps, df, Cs / 1e7, l, 300, 0, 0, phase_shift=0.1))
    assert pytest.approx(numpy.abs(ctf1 - ctf2).max(), abs=1e-5) == 0
