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
import argparse
import guanaco
import guanaco.plot


def add_argument(parser, name):
    """
    Add common command line arguments

    """
    {
        "energy": lambda: (
            parser.add_argument(
                "--E",
                dest="energy",
                default=300,
                type=float,
                help="The energy of the incident electrons (keV)",
            )
        ),
        "defocus": lambda: (
            parser.add_argument(
                "--df",
                dest="defocus",
                default=None,
                type=float,
                help="The defocus at the rotation axis (A)",
            )
        ),
        "num_defocus": lambda: (
            parser.add_argument(
                "--ndf",
                dest="num_defocus",
                default=None,
                type=int,
                help="The number of defocus steps to use",
            )
        ),
        "step_defocus": lambda: (
            parser.add_argument(
                "--sdf",
                dest="step_defocus",
                default=None,
                type=int,
                help="The defocus step to use",
            )
        ),
        "spherical_aberration": lambda: (
            parser.add_argument(
                "--Cs",
                dest="spherical_aberration",
                default=0,
                type=float,
                help="The spherical aberration (mm)",
            )
        ),
        "astigmatism": lambda: (
            parser.add_argument(
                "--Ca",
                dest="astigmatism",
                default=0,
                type=float,
                help="The 2-fold astigmatism (A)",
            )
        ),
        "astigmatism_angle": lambda: (
            parser.add_argument(
                "--Pa",
                dest="astigmatism_angle",
                default=0,
                type=float,
                help="The angle for the 2-fold astigmatism (deg)",
            )
        ),
        "phase_shift": lambda: (
            parser.add_argument(
                "--phi",
                dest="phase_shift",
                default=0,
                type=float,
                help="The phase shift to apply to the CTF (deg)",
            )
        ),
        "defocus_spread": lambda: (
            parser.add_argument(
                "--d",
                "--defocus_spread",
                dest="defocus_spread",
                type=float,
                default=None,
                help="The defocus spread (A)",
            )
        ),
        "source_spread": lambda: (
            parser.add_argument(
                "--theta_c",
                "--source_spread",
                dest="source_spread",
                type=float,
                default=0.02,
                help="The source spread (mrad)",
            )
        ),
        "chromatic_aberration": lambda: (
            parser.add_argument(
                "--Cc",
                "--chromatic_aberration",
                dest="chromatic_aberration",
                type=float,
                default=2.7,
                help="The chromatic aberration (mm)",
            )
        ),
        "energy_spread": lambda: (
            parser.add_argument(
                "--dEE",
                "--energy_spread",
                dest="energy_spread",
                type=float,
                default=0.33e-6,
                help="The energy spread dE / E",
            )
        ),
        "acceleration_voltage_spread": lambda: (
            parser.add_argument(
                "--dVV",
                "--acceleration_voltage_spread",
                dest="acceleration_voltage_spread",
                type=float,
                default=0.8e-6,
                help="The acceleration voltage spread dV / V",
            )
        ),
        "current_spread": lambda: (
            parser.add_argument(
                "--dII",
                "--current_spread",
                dest="current_spread",
                type=float,
                default=0.33e-6,
                help="The current spread dI / I",
            )
        ),
    }[name]()


def main(args=None):
    """
    Do the reconstruction

    """

    # Create the command line parser
    parser = argparse.ArgumentParser(description="Reconstruct")

    parser.add_argument(
        "-o",
        dest="output",
        default="rec.mrc",
        help="The output file containing the reconstructed projections.",
    )

    parser.add_argument(
        "-i",
        dest="input",
        default=None,
        required=True,
        help="The input file containing a stack of projection images.",
    )

    parser.add_argument(
        "--corrected-filename",
        dest="corrected_filename",
        default="GUANACO_CORRECTED.dat",
        help="The intermediate file containing the CTF corrected projections",
    )

    parser.add_argument(
        "-d,--device",
        dest="device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Use either the CPU or GPU",
    )

    parser.add_argument(
        "--transform",
        dest="transform",
        default=None,
        choices=["none", "minus"],
        help="Set the transform to use on the corrected projections",
    )

    parser.add_argument(
        "--angular_weights",
        dest="angular_weights",
        default=False,
        type=bool,
        help="Set whether or not to use angular weights",
    )

    parser.add_argument(
        "-n,--ncore",
        dest="ncore",
        default=None,
        type=int,
        help="Set the number of cores to use",
    )

    parser.add_argument(
        "--chunk_size",
        dest="chunk_size",
        default=None,
        type=int,
        help="Set the number of rows to reconstruct",
    )

    parser.add_argument(
        "--start_angle",
        dest="start_angle",
        default=None,
        type=float,
        help="The starting angle in degrees.",
    )

    parser.add_argument(
        "--step_angle",
        dest="step_angle",
        default=None,
        type=float,
        help="The step angle in degrees.",
    )

    parser.add_argument(
        "--pixel_size",
        dest="pixel_size",
        default=None,
        type=float,
        help="The pixel size in A",
    )

    parser.add_argument(
        "--centre",
        dest="centre",
        default=None,
        type=float,
        help="The rotation centre in pixels.",
    )

    parser.add_argument(
        "--method",
        dest="method",
        default="FBP_CTF",
        choices=["FBP_CTF", "FBP", "SIRT", "SART", "CGLS", "EM"],
        help="""
            Select the reconstruction algorithm. The FBP_CTF algorithm uses 3D
            CTF correction with FBP, other algorithms require tomopy to be
            installed""",
    )

    parser.add_argument(
        "--num_iter",
        dest="num_iter",
        default=50,
        type=int,
        help="The number of iterations to use in SIRT/SART/CGLS/EM",
    )

    # Add some common arguments
    add_argument(parser, "energy")
    add_argument(parser, "defocus")
    add_argument(parser, "num_defocus")
    add_argument(parser, "step_defocus")
    add_argument(parser, "spherical_aberration")
    add_argument(parser, "astigmatism")
    add_argument(parser, "astigmatism_angle")
    add_argument(parser, "phase_shift")

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Do the reconstruction
    guanaco.reconstruct_file(
        input_filename=args.input,
        output_filename=args.output,
        corrected_filename=args.corrected_filename,
        start_angle=args.start_angle,
        step_angle=args.step_angle,
        pixel_size=args.pixel_size,
        centre=args.centre,
        energy=args.energy,
        defocus=args.defocus,
        num_defocus=args.num_defocus,
        step_defocus=args.step_defocus,
        spherical_aberration=args.spherical_aberration,
        astigmatism=args.astigmatism,
        astigmatism_angle=args.astigmatism_angle,
        phase_shift=args.phase_shift,
        device=args.device,
        ncore=args.ncore,
        transform=args.transform,
        angular_weights=args.angular_weights,
        chunk_size=args.chunk_size,
        method=args.method,
        num_iter=args.num_iter,
    )


def plot_ctf(args=None):
    """
    Plot the CTF

    """

    # Create the command line parser
    parser = argparse.ArgumentParser(description="Plot the CTF")

    parser.add_argument(
        "-o",
        dest="output",
        default="ctf.png",
        help="The output image file for the CTF plot",
    )

    parser.add_argument(
        "--q_max",
        dest="q_max",
        type=float,
        default=0.5,
        help="The maximum spatial frequency (1/A)",
    )

    parser.add_argument(
        "--component",
        dest="component",
        type=str,
        choices=["real", "imag", "abs"],
        default="imag",
        help="The component to plot",
    )

    parser.add_argument(
        "--envelope",
        dest="envelope",
        type=str,
        choices=["spatial", "temporal", "all"],
        default=None,
        help="Plot the CTF envelope",
    )

    # Add some common arguments
    add_argument(parser, "energy")
    add_argument(parser, "defocus")
    add_argument(parser, "spherical_aberration")
    add_argument(parser, "astigmatism")
    add_argument(parser, "astigmatism_angle")
    add_argument(parser, "phase_shift")
    add_argument(parser, "defocus_spread")
    add_argument(parser, "source_spread")
    add_argument(parser, "chromatic_aberration")
    add_argument(parser, "energy_spread")
    add_argument(parser, "acceleration_voltage_spread")
    add_argument(parser, "current_spread")

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Do the reconstruction
    guanaco.plot.plot_ctf(
        filename=args.output,
        q_max=args.q_max,
        defocus=args.defocus,
        spherical_aberration=args.spherical_aberration,
        astigmatism=args.astigmatism,
        astigmatism_angle=args.astigmatism_angle,
        energy=args.energy,
        defocus_spread=args.defocus_spread,
        source_spread=args.source_spread,
        chromatic_aberration=args.chromatic_aberration,
        energy_spread=args.energy_spread,
        current_spread=args.current_spread,
        acceleration_voltage_spread=args.acceleration_voltage_spread,
        phase_shift=args.phase_shift,
        component=args.component,
        envelope=args.envelope,
    )


def generate_ctf(args=None):
    """
    Generate the CTF image

    """

    # Create the command line parser
    parser = argparse.ArgumentParser(description="Generate the CTF image")

    parser.add_argument(
        "-o",
        dest="output",
        default="ctf.mrc",
        help="The output image file for the CTF image",
    )

    parser.add_argument(
        "--size",
        "--image_size",
        dest="image_size",
        type=lambda x: list(map(int, x.split(","))),
        default=None,
        required=True,
        help="The image size",
    )

    parser.add_argument(
        "--px",
        "--pixel_size",
        dest="pixel_size",
        type=float,
        default=1,
        help="The pixel size",
    )

    parser.add_argument(
        "--recentre",
        dest="recentre",
        type=bool,
        default=False,
        help="Recentre the CTF in Fourier space",
    )

    # Add some common arguments
    add_argument(parser, "energy")
    add_argument(parser, "defocus")
    add_argument(parser, "num_defocus")
    add_argument(parser, "step_defocus")
    add_argument(parser, "spherical_aberration")
    add_argument(parser, "astigmatism")
    add_argument(parser, "astigmatism_angle")
    add_argument(parser, "phase_shift")
    add_argument(parser, "defocus_spread")
    add_argument(parser, "source_spread")
    add_argument(parser, "chromatic_aberration")
    add_argument(parser, "energy_spread")
    add_argument(parser, "acceleration_voltage_spread")
    add_argument(parser, "current_spread")

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Do the reconstruction
    guanaco.plot.generate_ctf(
        filename=args.output,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        defocus=args.defocus,
        num_defocus=args.num_defocus,
        step_defocus=args.step_defocus,
        spherical_aberration=args.spherical_aberration,
        astigmatism=args.astigmatism,
        astigmatism_angle=args.astigmatism_angle,
        energy=args.energy,
        defocus_spread=args.defocus_spread,
        source_spread=args.source_spread,
        chromatic_aberration=args.chromatic_aberration,
        energy_spread=args.energy_spread,
        current_spread=args.current_spread,
        acceleration_voltage_spread=args.acceleration_voltage_spread,
        phase_shift=args.phase_shift,
        recentre=args.recentre,
    )


def correct(args=None):
    """
    Do the CTF correction

    """

    # Create the command line parser
    parser = argparse.ArgumentParser(description="Perform CTF correction")

    parser.add_argument(
        "-o",
        dest="output",
        default="corrected.mrc",
        type=str,
        help="The output file containing the corrected projections.",
    )

    parser.add_argument(
        "-i",
        dest="input",
        default=None,
        required=True,
        type=str,
        help="The input file containing a stack of projection images.",
    )

    parser.add_argument(
        "-d,--device",
        dest="device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Use either the CPU or GPU",
    )

    parser.add_argument(
        "--transform",
        dest="transform",
        default=None,
        choices=["none", "minus"],
        help="Set the transform to use on the corrected projections",
    )

    parser.add_argument(
        "-n,--ncore",
        dest="ncore",
        default=None,
        type=int,
        help="Set the number of cores to use",
    )

    parser.add_argument(
        "--centre",
        dest="centre",
        default=None,
        type=float,
        help="The rotation centre in pixels.",
    )

    # Add some common arguments
    add_argument(parser, "energy")
    add_argument(parser, "defocus")
    add_argument(parser, "num_defocus")
    add_argument(parser, "step_defocus")
    add_argument(parser, "spherical_aberration")
    add_argument(parser, "astigmatism")
    add_argument(parser, "astigmatism_angle")
    add_argument(parser, "phase_shift")

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Do the reconstruction
    guanaco.correct_file(
        input_filename=args.input,
        output_filename=args.output,
        centre=args.centre,
        energy=args.energy,
        defocus=args.defocus,
        num_defocus=args.num_defocus,
        step_defocus=args.step_defocus,
        spherical_aberration=args.spherical_aberration,
        astigmatism=args.astigmatism,
        astigmatism_angle=args.astigmatism_angle,
        phase_shift=args.phase_shift,
        device=args.device,
        ncore=args.ncore,
        transform=args.transform,
    )
