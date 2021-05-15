# import numpy
# import mrcfile
# import pytest


# def test_rec():

#     data_expected1 = mrcfile.open("test/rec_expected_gpu.mrc").data
#     data_expected2 = mrcfile.open("test/rec_expected_cpu.mrc").data
#     data1 = mrcfile.open("test/rec_gpu.mrc").data
#     data2 = mrcfile.open("test/rec_cpu.mrc").data
#     diff1 = numpy.max(numpy.abs(data_expected1 - data1))
#     diff2 = numpy.max(numpy.abs(data_expected2 - data2))
#     assert diff1 == pytest.approx(0)
#     assert diff2 == pytest.approx(0)
# data_expected1 = mrcfile.open("test/rec_expected_gpu.mrc").data
# data_expected2 = mrcfile.open("test/rec_expected_cpu.mrc").data
# data1 = mrcfile.open("test/rec_gpu.mrc").data
# data2 = mrcfile.open("test/rec_cpu.mrc").data
# diff1 = numpy.max(numpy.abs(data_expected1 - data1))
# diff2 = numpy.max(numpy.abs(data_expected1 - data2))
# from matplotlib import pylab

# pylab.imshow((data1 - data2)[100, :, :])
# pylab.show()
# diff3 = numpy.max(numpy.abs(data1 - data2))
# assert diff3 == pytest.approx(0)
# assert diff1 == pytest.approx(0)
# assert diff2 == pytest.approx(0)


# if __name__ == "__main__":
#     test_rec()
