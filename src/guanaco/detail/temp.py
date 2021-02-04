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
        sizes = numpy.ones(ncore, dtype=numpy.int) * chunk_size
        # evenly distribute leftover across workers
        sizes[:leftover] += 1
        offsets = numpy.zeros(ncore + 1, dtype=numpy.int)
        offsets[1:] = numpy.cumsum(sizes)
        slcs = [
            numpy.s_[offsets[i] : offsets[i + 1]] for i in range(offsets.shape[0] - 1)
        ]
    elif nchunk == 0:
        # nchunk == 0 is a special case, we will collapse the dimension
        slcs = [numpy.s_[i] for i in range(axis_size)]
    else:
        # calculate offsets based on chunk size
        slcs = [
            numpy.s_[offset : offset + nchunk] for offset in range(0, axis_size, nchunk)
        ]
    return ncore, slcs


def recon(
    tomo, center, recon, theta, device="cpu", ncore=None, nchunk=None, gpu_list=None
):

    # Unpack arguments
    nslices = tomo.shape[0]
    grid_shape = recon.shape[1:3]

    # Compute the number of cores
    if device == "gpu" and gpu_list is not None:
        ngpu = len(gpu_list)
        ncore, slcs = get_ncore_slices(nslices, ngpu, nchunk)
        assert ncore == len(slcs)
        assert ncore == ngpu
    else:
        ncore, slcs = get_ncore_slices(nslices, ncore, nchunk)
        assert ncore == len(slcs)
        gpu_list = [None] * ncore

    # If only one core then run on this thread, otherwise spawn other threads
    if ncore == 1:
        for slc in slcs:
            recon_worker(
                tomo[slc], center[slc], recon[slc], theta, grid_shape, device, None
            )
    else:
        with cf.ThreadPoolExecutor(ncore) as e:
            for gpu, slc in zip(gpu_list, slcs):
                e.submit(
                    recon_worker,
                    tomo[slc],
                    center[slc],
                    recon[slc],
                    theta,
                    grid_shape,
                    device,
                    gpu,
                )


def recon_worker(tomo, center, recon, theta, grid_shape, device, gpu_index):

    if gpu_index is None:
        gpu_index = 0

    nslices, nang, ndet = tomo.shape
    for i in range(nslices):

        if device == "cpu":
            device2 = "host"

            sino = tomo[i]
            shft = int(numpy.round(ndet / 2.0 - center[i]))
            if not shft == 0:
                sino = numpy.roll(tomo[i], shft)
                l = shft
                r = ndet + shft
                if l < 0:
                    l = 0
                if r > ndet:
                    r = ndet
                sino[:, :l] = 0
                sino[:, r:] = 0

            guanaco.detail.recon(
                "host",
                grid_shape[0],
                grid_shape[1],
                1.0,
                ndet,
                theta.astype(numpy.float64),
                center[i],
                sino,
                recon[i],
                gpu_index,
            )

        else:
            device2 = "device"

            sino = tomo[i]

            guanaco.detail.recon(
                "device",
                grid_shape[0],
                grid_shape[1],
                1.0,
                ndet,
                theta.astype(numpy.float64),
                center[i],
                sino,
                recon[i],
                gpu_index,
            )
