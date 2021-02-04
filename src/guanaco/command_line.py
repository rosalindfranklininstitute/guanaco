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


def main(args=None):
    """
    Do the reconstruction

    """

    # Create the command line parser
    parser = argparse.ArgumentParser(description="Reconstruct")
    parser.add_argument("-o", dest="output", default="rec.mrc", help="The output file")
    parser.add_argument(
        "-i", dest="input", default=None, required=True, help="The input file"
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
        choices=["none", "minus_log", "minus"],
        help="Set the transform to use",
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
        "--df",
        dest="defocus",
        default=None,
        type=float,
        help="The defocus at the rotation axis",
    )

    parser.add_argument(
        "--Cs",
        dest="spherical_aberration",
        default=0,
        type=float,
        help="The spherical aberration",
    )

    parser.add_argument(
        "--ndf",
        dest="num_defocus",
        default=None,
        type=int,
        help="The number of defocus steps to use",
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Do the reconstruction
    guanaco.reconstruct_file(
        input_filename=args.input,
        output_filename=args.output,
        defocus=args.defocus,
        num_defocus=args.num_defocus,
        spherical_aberration=args.spherical_aberration,
        device=args.device,
        ncore=args.ncore,
        transform=args.transform,
        chunk_size=args.chunk_size,
    )
