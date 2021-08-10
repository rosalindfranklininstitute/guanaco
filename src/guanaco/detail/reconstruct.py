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
import time
from math import pi
import guanaco.detail
import guanaco.detail.mp

__all__ = ["reconstruct_file", "reconstruct"]


def get_centre(shape, centre, sinogram_order=True):
    if not sinogram_order:
        shape = list(shape)
        shape[0], shape[-2] = shape[-2], shape[0]
    if centre is None:
        centre = numpy.ones(shape[0], dtype="float32") * (shape[-1] / 2.0)
    elif numpy.array(centre).size == 1:
        centre = numpy.ones(shape[0], dtype="float32") * centre
    return centre.astype(dtype="float32", copy=False)


def compute_angular_weights(angles):
    """
    Compute angular weights for the projections

    Params:
        angles (float): The projection angles

    Returns:
        array: The weights

    """

    # Sort the angle indices
    index = numpy.argsort(angles)

    # Initialise the weights
    weights = numpy.zeros(angles.size)

    # Compute the mean interval for normalisation
    if angles.size > 1:
        mean_interval = (angles[index[-1]] - angles[index[0]]) / (angles.size - 1)
    else:
        mean_interval = 1.0

    # Compute the angular weights
    for i in range(angles.size):
        sum_intervals = 0.0
        sum_weights = 0.0
        if i - 2 >= 0:
            sum_weights += 1.0 / 1.5
            sum_intervals += (1.0 / 1.5) * (angles[index[i - 1]] - angles[index[i - 2]])
        if i - 1 >= 0:
            sum_weights += 2.0
            sum_intervals += 2.0 * (angles[index[i]] - angles[index[i - 1]])
        if i + 1 < angles.size:
            sum_weights += 2.0
            sum_intervals += 2.0 * (angles[index[i + 1]] - angles[index[i]])
        if i + 2 < angles.size:
            sum_weights += 1.0 / 1.5
            sum_intervals += (1.0 / 1.5) * (angles[index[i + 2]] - angles[index[i + 1]])
        if sum_weights != 0:
            weights[index[i]] = sum_intervals / sum_weights
        else:
            weights[index[i]] = 1.0

    # Normalise the weights
    weights /= mean_interval

    # Return the weights
    return weights


def get_weights(angles, angular_weights=False):
    """
    Get the sinogram weights

    Params:
        angles (array): The array of angles
        angular_weights (bool): Use angular weighting

    Returns:
        array: List of sinogram weights

    """
    if angular_weights:
        weights = compute_angular_weights(angles)
    else:
        weights = numpy.ones(angles.shape)
    return weights


def reconstruct(
    tomogram,
    angles,
    reconstruction=None,
    centre=None,
    pixel_size=1,
    min_defocus=0,
    max_defocus=0,
    sinogram_order=False,
    transform=None,
    angular_weights=False,
    device="cpu",
    ncore=None,
    nchunk=None,
    gpu_list=None,
):
    """
    The sinogram is expected to be provided with dimensions as follows:

    - Stack of projections:
        ( THETA, Y, X ) - sinogram_order = False

    - Stack of sinograms:
        ( Y, THETA, X ) - sinogram_order = True

    - Stack of defocus corrected projections
        ( THETA, DEFOCUS, Y, X ) - sinogram_order = False

    - Stack of defocus corrected sinograms
        ( Y, DEFOCUS, THETA, X ) - sinogram_order = True

    """

    def initialise_sinogram(tomogram, sinogram_order):
        assert len(tomogram.shape) in [3, 4]
        tomogram = tomogram.astype(dtype="float32", copy=False)
        if not sinogram_order:
            tomogram = numpy.swapaxes(tomogram, 0, -2)  # doesn't copy data
        return tomogram

    def initialise_reconstruction(reconstruction, sinogram):
        if reconstruction is None:
            shape = (sinogram.shape[0], sinogram.shape[-1], sinogram.shape[-1])
            reconstruction = numpy.zeros(shape, dtype="float32")
        else:
            assert reconstruction.shape[0] == sinogram.shape[0]
            assert reconstruction.shape[1] == sinogram.shape[-1]
            assert reconstruction.shape[2] == sinogram.shape[-1]
        return reconstruction.astype(dtype="float32", copy=False)

    # Initialize sinogram
    sinogram = initialise_sinogram(tomogram, sinogram_order)

    # Initialise reconstruction
    reconstruction = initialise_reconstruction(reconstruction, sinogram)

    # Generate args for the algorithm.
    centre = get_centre(sinogram.shape, centre)

    # Get the weights
    weights = get_weights(angles, angular_weights)

    # Perform the reconstruction in multiple threads
    guanaco.detail.mp.reconstruction_dispatcher(
        sinogram,
        reconstruction,
        centre,
        angles,
        weights=weights,
        pixel_size=pixel_size,
        min_defocus=min_defocus,
        max_defocus=max_defocus,
        transform=transform,
        device=device,
        ncore=ncore,
        nchunk=nchunk,
        gpu_list=gpu_list,
    )

    # Return reconstruction
    return reconstruction


def reconstruct_file(
    input_filename,
    output_filename,
    corrected_filename="GUANACO_CORRECTED.dat",
    start_angle=None,
    step_angle=None,
    pixel_size=None,
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
    angular_weights=False,
    chunk_size=None,
    method="FBP_CTF",
    num_iter=None,
):
    """
    Do the reconstruction

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

    def open_reconstruction_file(output_filename, shape, voxel_size):

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

        # Get the projection data
        projections = infile.data

        # Get the projection metadata
        if start_angle is None or step_angle is None or pixel_size is None:
            angles, voxel_size = read_projection_metadata(infile)
            assert voxel_size["x"] == voxel_size["y"]
            pixel_size = voxel_size["x"]
        else:
            angles = (
                (start_angle + numpy.arange(projections.shape[0]) * step_angle)
                * pi
                / 180.0
            )
            print(angles)
            voxel_size = pixel_size

        # Get the pixel size
        print("Pixel size %d A" % pixel_size)

        # Set the reconstruction shape
        output_shape = (
            projections.shape[1],
            projections.shape[2],
            projections.shape[2],
        )

        # Get the rotation centre
        centre = get_centre(projections.shape, centre, False)

        # Get the corrected projections.
        if method in ["FBP_CTF"]:
            projections, min_defocus, max_defocus = guanaco.correct_projections(
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
                corrected_filename,
                device,
            )

        # Open the output file
        print("Writing reconstruction to %s" % output_filename)
        with open_reconstruction_file(
            output_filename, output_shape, voxel_size
        ) as outfile:

            # Get the reconstruction data
            reconstruction = outfile.data

            # Reconstruct
            if method in ["FBP_CTF"]:
                reconstruct(
                    projections,
                    angles,
                    reconstruction,
                    centre=centre,
                    pixel_size=pixel_size,
                    min_defocus=min_defocus,
                    max_defocus=max_defocus,
                    sinogram_order=False,
                    transform=transform,
                    angular_weights=angular_weights,
                    device=device,
                    ncore=ncore,
                )
            else:
                reconstruct_tomopy(
                    projections,
                    angles,
                    reconstruction,
                    centre=centre,
                    transform=transform,
                    device=device,
                    ncore=ncore,
                    method=method,
                    num_iter=num_iter,
                )

    print("Time: %.2f seconds" % (time.time() - start_time))


def reconstruct_tomopy(
    projections,
    angles,
    reconstruction,
    centre=None,
    transform=None,
    device="cpu",
    ncore=None,
    method="FBP",
    num_iter=200,
    output_rot90=False,
):
    """
    Do the reconstruction

    """
    try:
        import tomopy
    except ImportError:
        raise RuntimeError(
            "The reconstruction algorithm selected requires tomopy and astra"
        )

    # Set the device
    print("Using %s" % device)
    if device == "cpu":
        options = {"proj_type": "linear", "method": method}
    else:
        options = {"proj_type": "cuda", "method": "%s_CUDA" % method}
        if ncore == None:
            ncore = 1

    # Add number of iterations
    if method in ["SIRT", "SART", "CGLS"]:
        options["num_iter"] = num_iter

    # Transform the projections
    if transform == "minus":
        projections = -projections

    # Reconstruct
    reconstruction[:] = tomopy.recon(
        projections, angles, algorithm=tomopy.astra, options=options, ncore=ncore
    )
