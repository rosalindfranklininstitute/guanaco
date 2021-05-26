import guanaco
import numpy
import mrcfile
import pytest
import os.path
import yaml
from math import pi


@pytest.fixture
def test_data():
    class TestData(object):
        def path(self, filename):
            return os.path.join(os.path.dirname(__file__), "data", filename)

        def angles(self, handle):
            theta = numpy.zeros(handle.extended_header.shape[0], dtype=numpy.float32)
            for i in range(handle.extended_header.shape[0]):
                theta[i] = handle.extended_header[i]["Alpha tilt"] * pi / 180.0
            return theta

        def centred_rotation_axis_centred_object(self):
            images = None
            expected = None
            return images, expected

        def off_centre_rotation_axis_centred_object(self):
            images = None
            expected = None
            return images, expected

        def centred_rotation_axis_off_centre_object(self):
            images = None
            expected = None
            return images, expected

        def off_centre_rotation_axis_off_centre_object(self):
            images = None
            expected = None
            return images, expected

        def centred_rotation_axis_centred_object_with_ctf(self):

            metadata_filename = self.path(
                "centred_rotation_axis_centred_object_uncorrected_ctf.yaml"
            )
            images_filename = self.path(
                "centred_rotation_axis_centred_object_uncorrected_ctf.mrc"
            )
            expected_filename = self.path("expected.mrc")

            metadata = yaml.safe_load(open(metadata_filename))
            images = mrcfile.open(images_filename)
            expected = None  # mrcfile.open(expected_filename)
            metadata["angles"] = self.angles(images)
            return metadata, images.data, expected

        def off_centre_rotation_axis_centred_object_with_ctf(self):

            metadata_filename = self.path(
                "off_centre_rotation_axis_centred_object_uncorrected_ctf.yaml"
            )
            images_filename = self.path(
                "off_centre_rotation_axis_centred_object_uncorrected_ctf.mrc"
            )
            expected_filename = self.path("expected.mrc")

            metadata = yaml.safe_load(open(metadata_filename))
            images = mrcfile.open(images_filename)
            expected = None  # mrcfile.open(expected_filename)
            metadata["angles"] = self.angles(images)
            return metadata, images.data, expected

        def centred_rotation_axis_off_centre_object_with_ctf(self):

            metadata_filename = self.path(
                "centred_rotation_axis_off_centre_object_uncorrected_ctf.yaml"
            )
            images_filename = self.path(
                "centred_rotation_axis_off_centre_object_uncorrected_ctf.mrc"
            )
            expected_filename = self.path("expected.mrc")

            metadata = yaml.safe_load(open(metadata_filename))
            images = mrcfile.open(images_filename)
            expected = None  # mrcfile.open(expected_filename)
            metadata["angles"] = self.angles(images)
            return metadata, images.data, expected

        def off_centre_rotation_axis_off_centre_object_with_ctf(self):

            metadata_filename = self.path(
                "off_centre_rotation_axis_off_centre_object_uncorrected_ctf.yaml"
            )
            images_filename = self.path(
                "off_centre_rotation_axis_off_centre_object_uncorrected_ctf.mrc"
            )
            expected_filename = self.path("expected.mrc")

            metadata = yaml.safe_load(open(metadata_filename))
            images = mrcfile.open(images_filename)
            expected = None  # mrcfile.open(expected_filename)
            metadata["angles"] = self.angles(images)
            return metadata, images.data, expected

    return TestData()


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_centred_rotation_axis_centred_object(test_data, device):
    """
    Reconstruct an object at the centre of rotation where the rotation axis is
    in the centre of the image.

    -------(-o-)-------

    """
    try:
        images, expected = test_data.centred_rotation_axis_centred_object()
    except Exception:
        pass


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_off_centre_rotation_axis_centred_object(test_data, device):
    """
    Reconstruct an object at the centre of rotation where the rotation axis is
    not in the centre of the image.

    ---(-o-)-----------

    """
    try:
        images, expected = test_data.off_centre_rotation_axis_centred_object()
    except Exception:
        pass


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_centred_rotation_axis_off_centre_object(test_data, device):
    """
    Reconstruct an object not at the centre of rotation where the rotation axis is
    in the centre of the image.

    ---(---)-o---------

    """
    try:
        images, expected = test_data.centred_rotation_axis_off_centre_object()
    except Exception:
        pass


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_off_centre_rotation_axis_off_centre_object(test_data, device):
    """
    Reconstruct an object not at the centre of rotation where the rotation axis is
    not in the centre of the image.

    -----o-(---)-------

    """
    try:
        images, expected = test_data.off_centre_rotation_axis_off_centre_object()
    except Exception:
        pass


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_centred_rotation_axis_centred_object_with_ctf(test_data, device):
    """
    Reconstruct an object at the centre of rotation where the rotation axis is
    in the centre of the image. Apply the 3D CTF correction.

    -------(-o-)-------

    """
    try:
        (
            metadata,
            images,
            expected,
        ) = test_data.centred_rotation_axis_centred_object_with_ctf()
    except Exception:
        return

    xsize = metadata["xsize"]
    ysize = metadata["ysize"]
    nangles = metadata["nangles"]
    angles = metadata["angles"]
    centre = metadata["centre"]
    position = metadata["position"]
    size = metadata["size"]
    df = metadata["df"]
    Cs = metadata["Cs"]
    pixel_size = metadata["pixel_size"]

    assert images.shape == (nangles, ysize, xsize)
    assert len(angles) == nangles

    x0 = position[0] - size[0] // 2
    x1 = position[0] + size[0] // 2
    y0 = position[1] - size[1] // 2
    y1 = position[1] + size[1] // 2
    z0 = position[2] - size[2] // 2
    z1 = position[2] + size[2] // 2

    # Check the ROI has the particle
    # from matplotlib import pylab
    # pylab.imshow(images[nangles//2, y0:y1, x0:x1])
    # pylab.show()

    rec = guanaco.reconstruct(images, angles, centre=centre, device=device)
    rec = numpy.swapaxes(rec, 0, 1)

    # Get the ROI
    x0 = x0 + xsize // 2 - centre
    x1 = x1 + xsize // 2 - centre
    rec = rec[z0:z1, y0:y1, x0:x1]

    # Inspect the reconstruction
    # from matplotlib import pylab

    # pylab.imshow(rec[size[2] // 2, :, :])
    # pylab.show()


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_off_centre_rotation_axis_centred_object_with_ctf(test_data, device):
    """
    Reconstruct an object at the centre of rotation where the rotation axis is
    not in the centre of the image. Apply the 3D CTF correction.

    ---(-o-)-----------

    """
    try:
        (
            metadata,
            images,
            expected,
        ) = test_data.off_centre_rotation_axis_centred_object_with_ctf()
    except Exception:
        return

    xsize = metadata["xsize"]
    ysize = metadata["ysize"]
    nangles = metadata["nangles"]
    angles = metadata["angles"]
    centre = metadata["centre"]
    position = metadata["position"]
    size = metadata["size"]
    df = metadata["df"]
    Cs = metadata["Cs"]
    pixel_size = metadata["pixel_size"]

    assert images.shape == (nangles, ysize, xsize)
    assert len(angles) == nangles

    x0 = position[0] - size[0] // 2
    x1 = position[0] + size[0] // 2
    y0 = position[1] - size[1] // 2
    y1 = position[1] + size[1] // 2
    z0 = position[2] - size[2] // 2
    z1 = position[2] + size[2] // 2

    # Check the ROI has the particle
    # from matplotlib import pylab
    # pylab.imshow(images[nangles//2, y0:y1, x0:x1])
    # pylab.show()

    rec = guanaco.reconstruct(images, angles, centre=centre, device=device)
    rec = numpy.swapaxes(rec, 0, 1)

    # Get the ROI
    x0 = x0 + xsize // 2 - centre
    x1 = x1 + xsize // 2 - centre
    rec = rec[z0:z1, y0:y1, x0:x1]

    # Inspect the reconstruction
    # from matplotlib import pylab

    # pylab.imshow(rec[size[2] // 2, :, :])
    # pylab.show()


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_centred_rotation_axis_off_centre_object_with_ctf(test_data, device):
    """
    Reconstruct an object not at the centre of rotation where the rotation axis is
    in the centre of the image. Apply the 3D CTF correction.

    ---(---)-o---------

    """
    try:
        (
            metadata,
            images,
            expected,
        ) = test_data.centred_rotation_axis_off_centre_object_with_ctf()
    except Exception:
        return

    xsize = metadata["xsize"]
    ysize = metadata["ysize"]
    nangles = metadata["nangles"]
    angles = metadata["angles"]
    centre = metadata["centre"]
    position = metadata["position"]
    size = metadata["size"]
    df = metadata["df"]
    Cs = metadata["Cs"]
    pixel_size = metadata["pixel_size"]

    assert images.shape == (nangles, ysize, xsize)
    assert len(angles) == nangles

    x0 = position[0] - size[0] // 2
    x1 = position[0] + size[0] // 2
    y0 = position[1] - size[1] // 2
    y1 = position[1] + size[1] // 2
    z0 = position[2] - size[2] // 2
    z1 = position[2] + size[2] // 2

    # Check the ROI has the particle
    # from matplotlib import pylab
    # pylab.imshow(images[nangles//2, y0:y1, x0:x1])
    # pylab.show()

    rec = guanaco.reconstruct(images, angles, centre=centre, device=device)
    rec = numpy.swapaxes(rec, 0, 1)

    # Get the ROI
    x0 = x0 + xsize // 2 - centre
    x1 = x1 + xsize // 2 - centre
    rec = rec[z0:z1, y0:y1, x0:x1]

    # Inspect the reconstruction
    # from matplotlib import pylab

    # pylab.imshow(rec[size[2] // 2, :, :])
    # pylab.show()


@pytest.mark.parametrize("device", ["gpu"])  # ["gpu", "cpu"])
def test_off_centre_rotation_axis_off_centre_object_with_ctf(test_data, device):
    """
    Reconstruct an object not at the centre of rotation where the rotation axis is
    not in the centre of the image. Apply the 3D CTF correction.

    -----o-(---)-------

    """
    try:
        (
            metadata,
            images,
            expected,
        ) = test_data.off_centre_rotation_axis_off_centre_object_with_ctf()
    except Exception:
        return

    xsize = metadata["xsize"]
    ysize = metadata["ysize"]
    nangles = metadata["nangles"]
    angles = metadata["angles"]
    centre = metadata["centre"]
    position = metadata["position"]
    size = metadata["size"]
    df = metadata["df"]
    Cs = metadata["Cs"]
    pixel_size = metadata["pixel_size"]

    assert images.shape == (nangles, ysize, xsize)
    assert len(angles) == nangles

    x0 = position[0] - size[0] // 2
    x1 = position[0] + size[0] // 2
    y0 = position[1] - size[1] // 2
    y1 = position[1] + size[1] // 2
    z0 = position[2] - size[2] // 2
    z1 = position[2] + size[2] // 2

    # Check the ROI has the particle
    # from matplotlib import pylab
    # pylab.imshow(images[nangles//2, y0:y1, x0:x1])
    # pylab.show()

    rec = guanaco.reconstruct(images, angles, centre=centre, device=device)
    rec = numpy.swapaxes(rec, 0, 1)

    # Get the ROI
    x0 = x0 + xsize // 2 - centre
    x1 = x1 + xsize // 2 - centre
    rec = rec[z0:z1, y0:y1, x0:x1]

    # Inspect the reconstruction
    # from matplotlib import pylab

    # pylab.imshow(rec[rec.shape[0] // 2, :, :])
    # pylab.show()
