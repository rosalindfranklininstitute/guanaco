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
import guanaco.detail.mp

__all__ = ["reconstruct_file", "reconstruct"]


def reconstruct(
    tomogram,
    reconstruction,
    angles,
    centre=None,
    sinogram_order=False,
    device="cpu",
    ncore=None,
    nchunk=None,
    gpu_list=None,
):
    def initialise_sinogram(tomo, sinogram_order):
        tomo = tomo.astype(dtype="float32", copy=False)
        if not sinogram_order:
            tomo = numpy.swapaxes(tomo, 0, 1)  # doesn't copy data
        # ensure contiguous
        tomo = numpy.require(tomo, requirements="AC")
        return tomo

    def get_centre(shape, centre):
        if centre is None:
            centre = numpy.ones(shape[0], dtype="float32") * (shape[2] / 2.0)
        elif numpy.array(centre).size == 1:
            centre = numpy.ones(shape[0], dtype="float32") * centre
        return centre.astype(dtype="float32", copy=False)

    # Initialize sinogram
    sinogram = initialise_sinogram(tomogram, sinogram_order)

    # Generate args for the algorithm.
    centre = get_centre(sinogram.shape, centre)

    # Perform the reconstruction in multiple threads
    guanaco.detail.mp.reconstruction_dispatcher(
        sinogram,
        reconstruction,
        centre,
        angles,
        device=device,
        ncore=ncore,
        nchunk=nchunk,
        gpu_list=gpu_list,
    )


def reconstruct_file(
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

    def read_projection_metadata(infile):

        # Read the voxel size
        voxel_size = infile.voxel_size
        print("Voxel size: ", infile.voxel_size)

        # Read the angles
        assert infile.data.shape[0] == infile.extended_header.shape[0]
        angles = numpy.zeros(infile.extended_header.shape[0], dtype=numpy.float32)
        for i in range(infile.extended_header.shape[0]):
            angles[i] = infile.extended_header[i]["Alpha tilt"] * pi / 180.0
            print("Image %d; angle %.4f" % (i, angles[i]))

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

        # Get the projection metadata
        angles, voxel_size = read_projection_metadata(infile)

        # Get the projection data
        projections = infile.data

        # Transform the projections
        if transform == "minus_log":
            projections = -numpy.log(projections)
        elif transform == "minus":
            projections = -projections

        # Set the reconstruction shape
        output_shape = (
            projections.shape[1],
            projections.shape[2],
            projections.shape[2],
        )

        # Open the output file
        print("Writing reconstruction to %s" % output_filename)
        with open_reconstruction_file(
            output_filename, output_shape, voxel_size
        ) as outfile:

            # Get the reconstruction data
            reconstruction = outfile.data

            # Reconstruct
            reconstruct(projections, reconstruction, angles, device=device, ncore=ncore)

    print("Time: %.2f seconds" % (time.time() - start_time))
