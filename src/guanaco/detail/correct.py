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
import tempfile
import time
import guanaco
from math import pi


__all__ = ["correct_file", "correct_projections"]


def get_centre(shape, centre, sinogram_order=True):
    if not sinogram_order:
        shape = list(shape)
        shape[0], shape[-2] = shape[-2], shape[0]
    if centre is None:
        centre = numpy.ones(shape[0], dtype="float32") * (shape[-1] / 2.0)
    elif numpy.array(centre).size == 1:
        centre = numpy.ones(shape[0], dtype="float32") * centre
    return centre.astype(dtype="float32", copy=False)


def get_defocus_steps(
    defocus,
    num_defocus=None,
    step_defocus=None,
    width=None,
    centre=None,
    pixel_size=None,
):
    """
    Compute the defocus steps

    Params:
        defocus (float): The defocus at the centre (A)
        num_defocus (int): The number of defocus values
        step_defocus (float): The defocus step (A)
        width (int): The number of pixels in x
        centre (float): The centre of rotation
        pixel_size (float): The pixel size (A)

    Returns:
        The defocus range (inclusive) - min, max, step, num

    """

    # Check the number of defoci
    if num_defocus is None or num_defocus <= 0:
        num_defocus = 1

    # Get the min and max defoci and the step
    if num_defocus == 1:
        min_defocus = defocus
        max_defocus = defocus
        step_defocus = 0
    elif num_defocus > 1:

        # If the step is set then compute min and max directly, otherwise
        # calculate based on the image shape and centre of rotation
        if step_defocus is not None and step_defocus > 0:
            min_defocus = defocus - 0.5 * (num_defocus - 1) * step_defocus
            max_defocus = defocus + 0.5 * (num_defocus - 1) * step_defocus
        else:
            z0 = defocus - centre * pixel_size
            z1 = defocus + centre * pixel_size
            z2 = defocus - (width - centre) * pixel_size
            z3 = defocus + (width - centre) * pixel_size
            min_defocus = min([z0.min(), z1.min(), z2.min(), z3.min()])
            max_defocus = max([z0.max(), z1.max(), z2.max(), z3.max()])
            step_defocus = (max_defocus - min_defocus) / (num_defocus - 1)

    # Return the min and max defocus and defocus step
    return min_defocus, max_defocus, step_defocus, num_defocus


def get_corrected_array(shape, filename_or_array=None):
    """
    Get the corrected array

    """

    # Create a memory mapped file of the correct dimension
    if filename_or_array is None:
        array = numpy.memmap(
            tempfile.TemporaryFile(), mode="w+", dtype="float32", shape=shape
        )
    elif isinstance(filename_or_array, str):
        array = numpy.memmap(filename_or_array, mode="w+", dtype="float32", shape=shape)
    else:
        array = filename_or_array.reshape(shape)

    # Return the array
    return array


def get_ctf(
    width,
    height,
    pixel_size,
    energy,
    defocus,
    spherical_aberration=0,
    astigmatism=0,
    astigmatism_angle=0,
    phase_shift=0,
):
    """
    Get the CTF

    Params:
        shape (tuple): The image size
        pixel_size (float): The pixel size
        energy (float): The electron energy (keV)
        defocus (float): The defocus (A)
        spherical_aberration (float): The spherical aberration (mm)
        astigmatism (float): The 2-fold astigmatism (A)
        astigmatism_angle (float): The angle for 2-fold astigmatism (deg)
        phase_shift (float): The phase shift (deg)

    Returns:
        array: The CTF image

    """
    ctf_calculator = guanaco.detail.CTF(
        l=guanaco.detail.get_electron_wavelength(energy * 1000),
        df=defocus,
        Cs=spherical_aberration * 1e7,
        Ca=astigmatism,
        Pa=astigmatism_angle * pi / 180,
        phi=phase_shift * pi / 180,
    )
    return ctf_calculator.get_ctf_simple(width, height, pixel_size)


def get_ctf_array(
    width,
    height,
    pixel_size,
    energy,
    min_defocus,
    num_defocus,
    step_defocus,
    spherical_aberration=0,
    astigmatism=0,
    astigmatism_angle=0,
    phase_shift=0,
):
    """
    Get the CTF array

    Params:
        shape (tuple): The image size
        pixel_size (float): The pixel size
        energy (float): The electron energy (keV)
        min_defocus (float): The minimum defocus (A)
        num_defocus (int): The number of defoci
        step_defocus (float): The defocus step (A)
        spherical_aberration (float): The spherical aberration (mm)
        astigmatism (float): The 2-fold astigmatism (A)
        astigmatism_angle (float): The angle for 2-fold astigmatism (deg)
        phase_shift (float): The phase shift (deg)

    Returns:
        array: The array of CTF images

    """

    # Precompute the CTF for each defoci
    ctf_array = numpy.zeros((num_defocus, height, width), dtype="complex64")

    # Loop through defoci
    for d in range(num_defocus):

        # Get the defocus
        df = min_defocus + d * step_defocus

        # Generate the ctf
        print("Computing CTF for defocus = %.2f" % df)
        ctf_array[d, :, :] = get_ctf(
            width,
            height,
            pixel_size,
            energy,
            df,
            spherical_aberration,
            astigmatism,
            astigmatism_angle,
            phase_shift,
        )

    # Return the CTF array
    return ctf_array


def correct_projections(
    projections,
    centre=None,
    pixel_size=1,
    energy=None,
    defocus=None,
    num_defocus=None,
    step_defocus=None,
    spherical_aberration=0,
    astigmatism=0,
    astigmatism_angle=0,
    phase_shift=0,
    corrected_output=None,
    device="cpu",
):
    """
    Correct the projections

    Params:
        projections (array): The array of projections
        centre (float): The rotation centre
        pixel_size (float): The pixel size
        energy (float): The electron energy (keV)
        defocus (float): The centre defocus (A)
        num_defocus (int): The number of defoci
        step_defocus (float): The defocus step (A)
        spherical_aberration (float): The spherical aberration (mm)
        astigmatism (float): The 2-fold astigmatism (A)
        astigmatism_angle (float): The angle for 2-fold astigmatism (deg)
        phase_shift (float): The phase shift (deg)
        corrected_output (object): The output data
        device (str): The cpu or gpu

    Returns:
        projections, min_defocus, max_defocus

    """

    # If we have no defocus set then do nothing. Otherwise correct them
    if defocus is not None:

        # Compute the defocus steps
        min_defocus, max_defocus, step_defocus, num_defocus = get_defocus_steps(
            defocus, num_defocus, step_defocus, projections.shape[2], centre, pixel_size
        )

        # The shape of the corrected projection array
        corrected_shape = (
            projections.shape[0],
            num_defocus,
            projections.shape[1],
            projections.shape[2],
        )

        # Init the corrected projections array
        corrected_projections = get_corrected_array(corrected_shape, corrected_output)

        # Precompute the CTF for each defoci
        ctf_array = get_ctf_array(
            projections.shape[2],
            projections.shape[1],
            pixel_size,
            energy,
            min_defocus,
            num_defocus,
            step_defocus,
            spherical_aberration,
            astigmatism,
            astigmatism_angle,
            phase_shift,
        )

        # Loop through all the projections and defoci and perform the CTF
        # correction
        for z in range(projections.shape[0]):
            image = projections[z, :, :]
            print(
                "Correcting image %d/%d with %d defoci"
                % (z + 1, projections.shape[0], len(ctf_array))
            )
            guanaco.detail.corr(
                image, ctf_array, corrected_projections[z, :, :, :], device
            )

        # Remove the dimension for corrections if only one correction
        if corrected_projections.shape[1] == 1:
            corrected_projections = corrected_projections.reshape(projections.shape)

        # Set the projections to be the corrected projections
        projections = corrected_projections

    else:
        defocus = 0
        min_defocus = 0
        max_defocus = 0

    # Return the projections with the min and max relative defoci
    return (projections, min_defocus - defocus, max_defocus - defocus)


def correct_file(
    input_filename,
    output_filename,
    centre=None,
    energy=None,
    defocus=None,
    num_defocus=None,
    step_defocus=None,
    spherical_aberration=None,
    astigmatism=None,
    astigmatism_angle=None,
    phase_shift=None,
    device="cpu",
    ncore=None,
    transform=None,
):
    """
    Perform the CTF correction

    """

    start_time = time.time()

    def read_projection_metadata(infile):

        # Read the voxel size
        voxel_size = infile.voxel_size
        print("Voxel size: ", infile.voxel_size)

        # Read the angles
        assert infile.data.shape[0] == infile.extended_header.shape[0]
        angles = numpy.zeros(infile.extended_header.shape[0], dtype=numpy.float32)
        for i in range(infile.extended_header.shape[0]):
            angles[i] = infile.extended_header[i]["Alpha tilt"] * pi / 180.0
            print("Image %d; angle %.4f" % (i + 1, angles[i]))

        # Return metadata
        return angles, voxel_size

    def open_corrected_file(output_filename, shape, voxel_size):

        # Open the file
        outfile = mrcfile.new_mmap(
            output_filename, overwrite=True, mrc_mode=2, shape=shape
        )

        # Set the voxel size
        outfile.voxel_size = voxel_size

        # Return the handle
        return outfile

    # Open the input file
    print("Reading %s" % input_filename)
    with mrcfile.mmap(input_filename) as infile:

        # Get the projection metadata
        angles, voxel_size = read_projection_metadata(infile)

        # Get the pixel size
        assert voxel_size["x"] == voxel_size["y"]
        pixel_size = voxel_size["x"]

        # Get the projection data
        projections = infile.data

        # Get the rotation centre
        centre = get_centre(projections.shape, centre, False)

        # Set the number of defoci
        if num_defocus is None or num_defocus == 0:
            num_defocus = 1

        # The shape of the corrected projection array
        output_shape = (
            projections.shape[0] * num_defocus,
            projections.shape[1],
            projections.shape[2],
        )

        # Open the output file
        print("Writing reconstruction to %s" % output_filename)
        with open_corrected_file(output_filename, output_shape, voxel_size) as outfile:

            # Get the corrected data
            corrected = outfile.data

            # Reconstruct
            correct_projections(
                projections,
                centre,
                pixel_size,
                energy,
                defocus,
                num_defocus,
                step_defocus,
                spherical_aberration,
                astigmatism,
                astigmatism_angle,
                phase_shift,
                corrected,
                device,
            )

    print("Time: %.2f seconds" % (time.time() - start_time))
