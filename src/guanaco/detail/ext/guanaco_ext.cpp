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
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <guanaco/guanaco.h>
#include <guanaco/detail/ext/ctf.h>

namespace py = pybind11;

namespace guanaco { namespace python {

  /**
   * Allow conversion from string to enum
   */
  template <typename T>
  py::enum_<T> enum_wrapper(const py::handle& scope, const char* name) {
    py::enum_<T> obj(scope, name);
    obj.def("__init__", [](T& v, const std::string& str) {
      py::object obj = py::cast(v);
      py::dict entries = obj.get_type().attr("__entries");
      for (auto item : entries) {
        if (str == item.first.cast<std::string>()) {
          v = item.second.cast<py::tuple>()[0].cast<T>();
          return;
        }
      }
      std::string tname = typeid(T).name();
      py::detail::clean_type_id(tname);
      throw py::value_error("\"" + std::string(str)
                            + "\" is not a valid value for enum type " + tname);
    });

    // Allow implicit conversions from string and int to enum
    py::implicitly_convertible<py::str, T>();
    py::implicitly_convertible<int, T>();
    return obj;
  }

  /**
   * A short wrapper function to call the reconstruction code
   */
  template <typename T>
  void reconstruct(const py::array_t<T>& sinogram,
                   py::array_t<T>& reconstruction,
                   const py::array_t<T>& angles,
                   float centre = 0,
                   float pixel_size = 1,
                   float min_defocus = 0,
                   float max_defocus = 0,
                   eDevice device = e_host,
                   int gpu_index = -1) {
    // Check the input
    GUANACO_ASSERT(sinogram.ndim() == 2 || sinogram.ndim() == 3);
    GUANACO_ASSERT(reconstruction.ndim() == 2);
    GUANACO_ASSERT(angles.ndim() == 1);

    // Check the sinogram dimensions
    if (sinogram.ndim() == 2) {
      GUANACO_ASSERT(sinogram.shape()[0] == angles.size());
      GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[0]);
      GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[1]);
    } else {
      GUANACO_ASSERT(sinogram.shape()[1] == angles.size());
      GUANACO_ASSERT(sinogram.shape()[2] == reconstruction.shape()[0]);
      GUANACO_ASSERT(sinogram.shape()[2] == reconstruction.shape()[1]);
    }

    // Initialise the configuration
    auto args = [&] {
      auto c = Config();
      c.device = device;
      c.gpu_index = gpu_index;
      c.num_pixels = sinogram.shape()[sinogram.ndim() - 1];
      c.num_angles = angles.size();
      c.num_defocus = sinogram.ndim() == 2 ? 1 : sinogram.shape()[0];
      c.grid_width = reconstruction.shape()[1];
      c.grid_height = reconstruction.shape()[0];
      c.pixel_size = pixel_size;
      c.min_defocus = min_defocus;
      c.max_defocus = max_defocus;
      c.centre = centre;
      c.angles.assign(angles.data(), angles.data() + angles.size());
      return c;
    }();

    // Create the reconstructor object
    auto rec = make_reconstructor(args);

    // Perform the reconstruction
    rec(sinogram.data(), reconstruction.mutable_data());
  }

  template <typename T>
  void correct(const py::array_t<T>& image,
               const py::array_t<std::complex<T>>& ctf,
               py::array_t<T>& rec,
               eDevice device) {
    // Check the input
    GUANACO_ASSERT(image.ndim() == 2);
    GUANACO_ASSERT(ctf.ndim() == 2 || ctf.ndim() == 3);
    GUANACO_ASSERT(rec.ndim() == image.ndim() + (ctf.ndim() - 2));

    // Get the number of images and ctfs
    auto num_ctf = ctf.ndim() == 2 ? 1 : ctf.shape()[0];
    auto ysize = image.shape()[0];
    auto xsize = image.shape()[1];

    // Check the shape
    GUANACO_ASSERT(num_ctf > 0);
    GUANACO_ASSERT(ysize > 0);
    GUANACO_ASSERT(xsize > 0);
    GUANACO_ASSERT(ctf.shape()[ctf.ndim() - 2] == ysize);
    GUANACO_ASSERT(ctf.shape()[ctf.ndim() - 1] == xsize);
    GUANACO_ASSERT(rec.shape()[rec.ndim() - 2] == ysize);
    GUANACO_ASSERT(rec.shape()[rec.ndim() - 1] == xsize);

    // Call the function to CTF corret the data
    correct(image.data(),
            ctf.data(),
            rec.mutable_data(),
            xsize,
            ysize,
            num_ctf,
            device);
  }

}}  // namespace guanaco::python

PYBIND11_MODULE(guanaco_ext, m) {
  // Export the device enum
  guanaco::python::enum_wrapper<guanaco::eDevice>(m, "eDevice")
    .value("cpu", guanaco::e_host)
    .value("gpu", guanaco::e_device)
    .export_values();

  // Export the CTF correction function
  m.def("corr",
        &guanaco::python::correct<float>,
        py::arg("image"),
        py::arg("ctf"),
        py::arg("rec"),
        py::arg("device") = guanaco::e_host);

  // Export the reconstruction function
  m.def("recon",
        &guanaco::python::reconstruct<float>,
        py::arg("sinogram"),
        py::arg("reconstruction"),
        py::arg("angles"),
        py::arg("centre"),
        py::arg("pixel_size") = 1.0,
        py::arg("min_defocus") = 0.0,
        py::arg("max_defocus") = 0.0,
        py::arg("device") = guanaco::e_host,
        py::arg("gpu_index") = -1);

  export_ctf(m);
}
