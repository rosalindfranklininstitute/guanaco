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
import guanaco

__all__ = ["correct_stack"]


def correct_stack(data, ctf, corrected=None):
    """
    Compute the CTF correction

    Args:
        data (array): The input data
        ctf (array): The input ctf
        corrected (array): The output data

    Returns:
        array: The output corrected data

    """

    # Create a new array
    if corrected is None:
        corrected = numpy.zeros(data.shape, dtype="float32")
    assert corrected.shape == data.shape
    assert corrected.shape[1] == ctf.shape[0]
    assert corrected.shape[2] == ctf.shape[1]

    # Apply the correction
    for i in range(data.shape[0]):
        logger.info("Appling CTF correction to image %d" % (i + 1))
        guanaco.detail.correct(data[i, :, :], ctf, corrected[i, :, :])
        # corrected[i, :, :] = correct_image(data[i, :, :], ctf)

    # Return corrected
    return corrected


# def correct_file(
#     input_image_filename,
#     output_image_filename=None,
#     input_ctf_filename=None,
#     output_ctf_filename=None,
#     shift=False,
#     **kwargs
# ):
#     """
#     Compute the CC between two maps

#     Args:
#         input_image_filename (str): The input image filename
#         output_image_filename (str): The output image filename
#         input_ctf_filename (str): The input ctf filename
#         output_ctf_filename (str): The output ctf filename
#         shift (bool): Shift the CTF in Fourier space

#     """

#     # Open the input file
#     infile = selknam.ctf.util.read(input_image_filename)

#     # Get the data
#     data = infile.data

#     # Read or generate the CTF
#     if input_ctf_filename is not None:
#         ctffile = selknam.ctf.util.read(input_ctf_filename)
#         ctf = ctffile.data
#         if len(ctf.shape) == 3:
#             if ctf.shape[0] == 1:
#                 ctf = ctf.reshape(ctf.shape[1:3])
#                 assert ctf.shape[0] == data.shape[1]
#                 assert ctf.shape[1] == data.shape[2]
#             else:
#                 assert ctf.shape == data.shape
#         else:
#             assert ctf.shape[0] == data.shape[1]
#             assert ctf.shape[1] == data.shape[2]
#         if shift:
#             assert len(ctf.shape) == 2
#             ctf = numpy.fft.fftshift(ctf)
#         assert ctffile.voxel_size == infile.voxel_size
#     else:
#         assert infile.voxel_size["x"] == infile.voxel_size["y"]
#         assert infile.voxel_size["x"] == infile.voxel_size["z"]
#         pixel_size = infile.voxel_size["x"]
#         ctf = selknam.ctf.simulate(
#             output_image_filename=output_ctf_filename,
#             image_size=(data.shape[1], data.shape[2]),
#             pixel_size=pixel_size,
#             centre=False,
#             **kwargs
#         )

#     outfile = selknam.ctf.util.new(
#         output_image_filename, data.shape, data.dtype, reference=infile
#     )

#     # Compute the cc
#     correct_stack(data, ctf, corrected=outfile.data)

#     # Return the output file
#     return outfile
