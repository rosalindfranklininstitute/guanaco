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
#ifndef GUANACO_PYTHON_ENUMS_H
#define GUANACO_PYTHON_ENUMS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <guanaco/guanaco.h>

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

}}  // namespace guanaco::python

inline void export_enums(py::module m) {
  // Export the device enum
  guanaco::python::enum_wrapper<guanaco::eDevice>(m, "eDevice")
    .value("cpu", guanaco::e_host)
    .value("gpu", guanaco::e_device)
    .export_values();
}

#endif
