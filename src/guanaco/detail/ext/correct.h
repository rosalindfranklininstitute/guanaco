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
#ifndef GUANACO_PYTHON_CORRECT_H
#define GUANACO_PYTHON_CORRECT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <guanaco/guanaco.h>

namespace py = pybind11;

namespace guanaco { namespace python {

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

inline void export_correct(py::module m) {
  // Export the CTF correction function
  m.def("corr",
        &guanaco::python::correct<float>,
        py::arg("image"),
        py::arg("ctf"),
        py::arg("rec"),
        py::arg("device") = guanaco::e_host);
}

#endif
