from __future__ import absolute_import, division, print_function, unicode_literals

import guanaco.detail

import numpy
import multiprocessing as mp
import concurrent.futures as cf


def get_ncore_slices(axis_size, ncore=None, nchunk=None):
    # default ncore to max (also defaults ncore == 0)
    if not ncore:
        ncore = mp.cpu_count()
    if nchunk is None:
        # calculate number of slices to send to each GPU
        chunk_size = axis_size // ncore
        leftover = axis_size % ncore
        sizes = numpy.ones(ncore, dtype="int") * chunk_size
        # evenly distribute leftover across workers
        sizes[:leftover] += 1
        offsets = numpy.zeros(ncore + 1, dtype="int")
        offsets[1:] = numpy.cumsum(sizes)
        slices = [
            numpy.s_[offsets[i] : offsets[i + 1]] for i in range(offsets.shape[0] - 1)
        ]
    elif nchunk == 0:
        # nchunk == 0 is a special case, we will collapse the dimension
        slices = [numpy.s_[i] for i in range(axis_size)]
    else:
        # calculate offsets based on chunk size
        slices = [
            numpy.s_[offset : offset + nchunk] for offset in range(0, axis_size, nchunk)
        ]
    return ncore, slices


def reconstruction_dispatcher(
    sinogram,
    reconstruction,
    centre,
    angles,
    weights=None,
    pixel_size=1,
    min_defocus=0,
    max_defocus=0,
    transform=None,
    device="cpu",
    ncore=None,
    nchunk=None,
    gpu_list=None,
):

    # Unpack arguments
    nslices = sinogram.shape[0]

    # Set the number of cores in the case of GPU
    if device == "gpu" and ncore is None:
        ncore = 1

    # Compute the number of cores
    if device == "gpu" and gpu_list is not None:
        ngpu = len(gpu_list)
        ncore, slices = get_ncore_slices(nslices, ngpu, nchunk)
        assert ncore == len(slices)
        assert ncore == ngpu
    else:
        ncore, slices = get_ncore_slices(nslices, ncore, nchunk)
        assert ncore == len(slices)
        gpu_list = [None] * ncore

    # If only one core then run on this thread, otherwise spawn other threads
    if ncore == 1:
        for s in slices:
            reconstruction_worker(
                sinogram[s],
                reconstruction[s],
                centre[s],
                angles,
                weights,
                pixel_size,
                min_defocus,
                max_defocus,
                transform,
                device,
                None,
            )
    else:
        with cf.ThreadPoolExecutor(ncore) as e:
            for gpu, s in zip(gpu_list, slices):
                e.submit(
                    reconstruction_worker,
                    sinogram[s],
                    reconstruction[s],
                    centre[s],
                    angles,
                    weights,
                    pixel_size,
                    min_defocus,
                    max_defocus,
                    transform,
                    device,
                    gpu,
                )


def reconstruction_worker(
    sinogram,
    reconstruction,
    centre,
    angles,
    weights,
    pixel_size,
    min_defocus,
    max_defocus,
    transform,
    device,
    gpu_index,
):

    if gpu_index is None:
        gpu_index = 0

    nslices = sinogram.shape[0]

    for i in range(nslices):
        sino = sinogram[i]

        # Transform the corrected projections
        if transform == "minus":
            sino = -sino

        # Get as contiguous array
        sino = numpy.ascontiguousarray(sino)

        # Set weights
        if weights is not None:
            if len(sino.shape) == 3:
                for j in range(sino.shape[1]):
                    sino[:, j, :] *= weights[j]
            else:
                for j in range(sino.shape[1]):
                    sino[j, :] *= weights[j]

        print("Reconstructing slice %d/%d" % (i + 1, nslices))
        guanaco.detail.recon(
            sino,
            reconstruction[i],
            angles,
            centre[i],
            pixel_size,
            min_defocus,
            max_defocus,
            device,
            gpu_index,
        )
