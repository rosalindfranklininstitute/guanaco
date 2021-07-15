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
import mrcfile
import numpy
import guanaco
from matplotlib import pylab
from math import pi


def plot_ctf(
    filename=None,
    q_max=0.5,
    defocus=0,
    spherical_aberration=2.7,
    astigmatism=0,
    astigmatism_angle=0,
    energy=300,
    defocus_spread=None,
    source_spread=0.02,
    chromatic_aberration=2.7,
    energy_spread=0.33e-6,
    current_spread=0.33e-6,
    acceleration_voltage_spread=0.8e-6,
    phase_shift=0,
    component="imag",
    envelope=None,
    ax=None,
    label=None,
    Et_label=None,
    Es_label=None,
    show=False,
):
    """
    Plot the CTF

    Args:
        filename (str): The output filename
        q_max (float): The maximum spatial frequency  (1/A)
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in mm)
        astigmatism (float): The astigmatism (A)
        astigmatism_angle (float): The astigmatism angle (deg)
        energy (float): The electron energy (keV)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (mrad)
        Cc (float): The chromatic aberrationa (mm)
        dEE (float): dE/E, the fluctuation in the electron energy
        dII (float): dI/I, the fluctuation in the lens current
        dVV (float): dV/V, the fluctuation in the acceleration voltage
        phase_shift (float): The phase shift (rad) - 0 without phase plate, pi/2 with phase plate
        component (str): The bit to plot (real, imag, abs)
        envelope (bool): Plot the envelope function
        ax (object): The axis object (if none create figure)
        label (str): The label for the plot
        Et_label (str): The label for the plot
        Es_label (str): The label for the plot

    """
    # Convert some quantities
    source_spread = source_spread / 1e3  # mrad -> rad
    spherical_aberration = spherical_aberration * 1e7  # mm -> A
    chromatic_aberration = chromatic_aberration * 1e7  # mm -> A
    astigmatism_angle = astigmatism_angle * pi / 190  # deg -> rad
    energy = energy * 1e3  # keV -> eV

    # Compute the defocus spread
    if defocus_spread is None:
        defocus_spread = guanaco.detail.get_defocus_spread(
            Cc=chromatic_aberration,
            dEE=energy_spread,
            dII=current_spread,
            dVV=acceleration_voltage_spread,
        )

    # Check the defocus
    if defocus is None:
        defocus = 0

    # Compute the wavelength
    wavelength = guanaco.detail.get_electron_wavelength(energy)

    # Setup the CTF calculation
    ctf_calculator = guanaco.detail.CTF(
        l=wavelength,
        df=defocus,
        Cs=spherical_aberration,
        Ca=astigmatism,
        Pa=astigmatism_angle,
        dd=defocus_spread,
        theta_c=source_spread,
        phi=phase_shift,
    )

    # Generate the spatial frequencies (1/A)
    q = numpy.arange(0, q_max, q_max / 1000.0)
    theta = numpy.zeros(q.size)

    # Evaluate the ctf
    ctf = ctf_calculator.get_ctf(q, theta)

    # Create the figure
    if ax is None:
        fig, ax = pylab.subplots(figsize=(12, 8))
    else:
        fig = None
    ax.set_xlabel("Spatial frequency (1/A)")
    ax.set_ylabel("CTF")

    # Plot the CTF
    if component is None:
        pass
    elif component == "real":
        if label is None:
            label = "CTF (real)"
        ax.plot(q, numpy.real(ctf), label=label)
    elif component == "imag":
        if label is None:
            label = "CTF (imag)"
        ax.plot(q, numpy.imag(ctf), label=label)
    elif component == "abs":
        if label is None:
            label = "CTF (abs)"
        ax.plot(q, numpy.abs(ctf), label=label)
    else:
        raise ValueError(
            "Expected component in ['real', 'imag', 'abs'], got %s" % component
        )

    # Plot the envelope
    if envelope is not None:

        # Compute the spatial incoherence envelope
        if envelope in ["spatial", "all"]:
            if Es_label is None:
                Es_label = "Es"
            Es = ctf_calculator.get_Es(q, theta)
            ax.plot(q, Es, label=Es_label)

        # Compute the temporal incoherence envelope
        if envelope in ["temporal", "all"]:
            if Et_label is None:
                Et_label = "Et"
            Et = ctf_calculator.get_Et(q)
            ax.plot(q, Et, label=Es_label)

        # Plot the envelope
        ax.legend()

    # Save the figure
    if fig is not None:
        if filename is not None:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
        elif show:
            pylab.show()
        pylab.close(fig)


def generate_ctf(
    filename=None,
    image_size=None,
    pixel_size=1,
    defocus=0,
    num_defocus=None,
    step_defocus=None,
    spherical_aberration=2.7,
    astigmatism=0,
    astigmatism_angle=0,
    energy=300,
    defocus_spread=None,
    source_spread=0.02,
    chromatic_aberration=2.7,
    energy_spread=0.33e-6,
    current_spread=0.33e-6,
    acceleration_voltage_spread=0.8e-6,
    phase_shift=0,
    component="imag",
    envelope=True,
    recentre=False,
):
    """
    Generate a CTF image

    Args:
        filename (str): The output filename
        image_size (tuple): The image size in pixels
        pixel_size (float): The pixel size (A)
        defocus (float): The defocus (df in A)
        spherical_aberration (float): The spherical aberration (Cs in mm)
        astigmatism (float): The astigmatism (A)
        astigmatism_angle (float): The astigmatism angle (deg)
        energy (float): The electron energy (keV)
        defocus_spread (float): The defocus spread (A)
        source_spread (float): The source spread (mrad)
        Cc (float): The chromatic aberrationa (mm)
        dEE (float): dE/E, the fluctuation in the electron energy
        dII (float): dI/I, the fluctuation in the lens current
        dVV (float): dV/V, the fluctuation in the acceleration voltage
        phase_shift (float): The phase shift (rad) - 0 without phase plate, pi/2 with phase plate
        component (str): The bit to plot (real, imag, abs)
        envelope (bool): Plot the envelope function

    """
    # Convert some quantities
    source_spread = source_spread / 1e3  # mrad -> rad
    spherical_aberration = spherical_aberration * 1e7  # mm -> A
    chromatic_aberration = chromatic_aberration * 1e7  # mm -> A
    astigmatism_angle = astigmatism_angle * pi / 190  # deg -> rad
    energy = energy * 1e3  # keV -> eV

    # Compute the defocus spread
    if defocus_spread is None:
        defocus_spread = guanaco.detail.get_defocus_spread(
            Cc=chromatic_aberration,
            dEE=energy_spread,
            dII=current_spread,
            dVV=acceleration_voltage_spread,
        )

    # Check the defocus
    if defocus is None:
        defocus = 0
    if num_defocus is None or num_defocus == 0:
        num_defocus = 1
    if step_defocus is None:
        step_defocus = 0

    # Compute the wavelength
    wavelength = guanaco.detail.get_electron_wavelength(energy)

    # Init output ctf file
    handle = mrcfile.new_mmap(
        filename,
        shape=(num_defocus, image_size[0], image_size[1]),
        mrc_mode=mrcfile.utils.mode_from_dtype(numpy.dtype(numpy.complex64)),
        overwrite=True,
    )

    # Set the voxel size
    handle.voxel_size = pixel_size

    # Loop through the defoci
    for i in range(num_defocus):

        # Set the defocus
        df = defocus + (i - 0.5 * (num_defocus - 1)) * step_defocus
        print("Computing CTF for defocus = %d A" % df)

        # Setup the CTF calculation
        ctf_calculator = guanaco.detail.CTF(
            l=wavelength,
            df=df,
            Cs=spherical_aberration,
            Ca=astigmatism,
            Pa=astigmatism_angle,
            dd=defocus_spread,
            theta_c=source_spread,
            phi=phase_shift,
        )

        # Evaluate the ctf
        ctf = ctf_calculator.get_ctf(image_size[1], image_size[0], pixel_size)

        # Recentre
        if recentre:
            ctf = numpy.fft.fftshift(ctf)

        # Set the CTF
        handle.data[i, :, :] = ctf
