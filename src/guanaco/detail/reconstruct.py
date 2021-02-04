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
import guanaco.detail.temp

__all__ = ["reconstruct_chunk", "reconstruct", "recon"]


def init_tomo(tomo, sinogram_order):
    tomo = tomo.astype(dtype="float32", copy=False)
    if not sinogram_order:
        tomo = numpy.swapaxes(tomo, 0, 1)  # doesn't copy data
    # ensure contiguous
    tomo = numpy.require(tomo, requirements="AC")
    return tomo


def get_center(shape, center):
    if center is None:
        center = numpy.ones(shape[0], dtype="float32") * (shape[2] / 2.0)
    elif numpy.array(center).size == 1:
        center = numpy.ones(shape[0], dtype="float32") * center
    return center.astype(dtype="float32", copy=False)


def recon(
    tomo,
    theta,
    center=None,
    sinogram_order=False,
    device="cpu",
    ncore=None,
    nchunk=None,
    gpu_list=None,
    num_gridx=None,
    num_gridy=None,
    **kwargs
):
    # Initialize tomography data.
    tomo = init_tomo(tomo, sinogram_order)

    if device == "gpu" and ncore == None:
        ncore = 1

    if num_gridx is None:
        num_gridx = tomo.shape[2]
    if num_gridy is None:
        num_gridy = tomo.shape[2]

    # Ensure we have the right type
    theta = theta.astype(dtype="float32", copy=False)

    # Generate args for the algorithm.
    center_arr = get_center(tomo.shape, center)

    # Initialize reconstruction.
    recon_shape = (tomo.shape[0], num_gridx, num_gridy)
    recon = numpy.full(recon_shape, 1e-6, dtype=numpy.float32)
    guanaco.detail.temp.recon(
        tomo,
        center_arr,
        recon,
        theta,
        device=device,
        ncore=ncore,
        nchunk=nchunk,
        gpu_list=gpu_list,
    )
    return recon


def reconstruct_chunk(projections, theta, device="cpu", ncore=None, transform=None):
    """
    Do the reconstruction

    """

    # Set the device
    print("Using %s" % device)

    # Transform the projections
    if transform == "minus_log":
        projections = tomopy.minus_log(projections)
    elif transform == "minus":
        projections = -projections

    # Reconstruct
    reconstruction = recon(projections, theta, device=device, ncore=ncore)

    # The reconstruction gives the rotation axis along Z whereas we want to
    # preserve the rotation axis along y. Rotate around the X axis by 90
    # degrees gives correct X and Y axis and assumes projections in Z direction
    return numpy.rot90(reconstruction, axes=(0, 1))


def reconstruct(
    input_filename,
    output_filename,
    device="cpu",
    ncore=None,
    transform=None,
    chunk_size=None,
):
    """
    Do the reconstruction

    """
    start_time = time.time()

    # Open the input file
    print("Reading %s" % input_filename)
    infile = mrcfile.mmap(input_filename)
    voxel_size = infile.voxel_size
    print("Voxel size: ", infile.voxel_size)

    # Get the infile data
    projections = infile.data

    # Read the angles
    theta = numpy.zeros(projections.shape[0], dtype=numpy.float32)
    for i in range(infile.extended_header.shape[0]):
        theta[i] = infile.extended_header[i]["Alpha tilt"] * pi / 180.0
        print("Image %d; angle %.4f" % (i, theta[i]))

    # Write the output data
    print("Writing reconstruction to %s" % output_filename)
    output_shape = (projections.shape[2], projections.shape[1], projections.shape[2])
    outfile = mrcfile.new_mmap(
        output_filename, overwrite=True, mrc_mode=2, shape=output_shape
    )
    outfile.voxel_size = voxel_size

    # Set the chunk size
    if chunk_size is None:
        chunk_size = projections.shape[1]

    # Reconstruct in chunks
    for y0 in range(0, projections.shape[1], chunk_size):
        y1 = min(y0 + chunk_size, projections.shape[1])
        print("Reconstructing slices %d -> %d" % (y0, y1))
        outfile.data[:, y0:y1, :] = reconstruct_chunk(
            projections[:, y0:y1, :],
            theta,
            device=device,
            ncore=ncore,
            transform=transform,
        )

    print("Time: %.2f seconds" % (time.time() - start_time))
