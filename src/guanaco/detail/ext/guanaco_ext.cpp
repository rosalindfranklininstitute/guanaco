/*
 * This file is part of guanaco-ctf.
 * Copyright 2021 Diamond Light Source
 * Copyright 2021 Rosalind Franklin Institute
 *
 * Author: James Parkhurst
 *
 * guanaco-ctf is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * guanaco-ctf is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with guanaco-ctf. If not, see <http:// www.gnu.org/licenses/>.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <guanaco/guanaco.h>

namespace py = pybind11;

namespace guanaco {

  template <typename T>
  void reconstruct(const py::array_t<T> &sinogram,
                   py::array_t<T> &reconstruction,
                   const py::array_t<T> angles,
                   float centre = 0,
                   float pixel_size = 1,
                   std::string device = "cpu",
                   int gpu_index = -1) {

    // Check the input
    GUANACO_ASSERT(sinogram.ndim() == 2);
    GUANACO_ASSERT(reconstruction.ndim() == 2);
    GUANACO_ASSERT(angles.ndim() == 1);
    GUANACO_ASSERT(sinogram.shape()[0] == angles.size());
    GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[0]);
    GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[1]);

    // Initialise the configuration
    auto args = [&] {
      auto c = Config();
      c.device = device == "cpu" ? e_host : e_device;
      c.gpu_index = gpu_index;
      c.num_pixels = sinogram.shape()[1];
      c.num_angles = sinogram.shape()[0];
      c.grid_width = reconstruction.shape()[1];
      c.grid_height = reconstruction.shape()[0];
      c.pixel_size = pixel_size;
      c.centre = centre;
      c.angles.assign(angles.data(), angles.data() + angles.size());
      return c;
    }();

    // Create the reconstructor object
    auto rec = make_reconstructor(args);

    // Perform the reconstruction
    rec(sinogram.data(), reconstruction.mutable_data());
  }

}

PYBIND11_MODULE(guanaco_ext, m) {

  m.def("recon",
        &guanaco::reconstruct<float>,
        py::arg("sinogram"),
        py::arg("reconstruction"),
        py::arg("angles"),
        py::arg("centre"),
        py::arg("pixel_size") = 1.0,
        py::arg("device") = "cpu",
        py::arg("gpu_index") = -1);
}
