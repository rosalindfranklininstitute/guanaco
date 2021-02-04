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
#include <guanaco/guanaco.h>

namespace py = pybind11;

PYBIND11_MODULE(guanaco_ext, m) {
  /* m.def("recon", &guanaco::rec_temp<float>); */

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
