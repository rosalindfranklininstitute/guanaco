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
#ifndef GUANACO_PYTHON_CTF_H
#define GUANACO_PYTHON_CTF_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <guanaco/guanaco.h>

namespace py = pybind11;

namespace guanaco { namespace python {

  namespace detail {

    template <typename T>
    py::array::ShapeContainer get_shape(py::array_t<T> a) {
      return py::array::ShapeContainer(a.shape(), a.shape() + a.ndim());
    }

  }  // namespace detail

  template <typename T>
  py::tuple get_q_and_theta(std::size_t w, std::size_t h, T ps) {
    py::array_t<T> q({h, w});
    py::array_t<T> a({h, w});
    auto q_ptr = q.mutable_data();
    auto a_ptr = a.mutable_data();
    GUANACO_ASSERT(ps > 0);
    auto inv_ps = 1.0 / ps;
    for (auto j = 0; j < h; ++j) {
      for (auto i = 0; i < w; ++i) {
        auto t = get_q(i, j, w, h, inv_ps);
        *q_ptr++ = std::get<0>(t);
        *a_ptr++ = std::get<1>(t);
      }
    }
    return make_tuple(q, a);
  }

  template <typename T>
  py::array_t<std::complex<T>> get_ctf_q(py::array_t<T> q,
                                         T l,
                                         T df,
                                         T Cs,
                                         T Ca,
                                         T Pa,
                                         T dd,
                                         T theta_c,
                                         T phi = 0) {
    py::array_t<std::complex<T>> result(detail::get_shape(q));
    get_ctf_n(q.data(),
              result.mutable_data(),
              q.size(),
              l,
              df,
              Cs,
              Ca,
              Pa,
              dd,
              theta_c,
              phi);
    return result;
  }

  template <typename T>
  py::array_t<std::complex<T>>
  get_ctf_simple_q(py::array_t<T> q, T l, T df, T Cs, T Ca, T Pa, T phi = 0) {
    py::array_t<std::complex<T>> result(detail::get_shape(q));
    get_ctf_n_simple(
      q.data(), result.mutable_data(), q.size(), l, df, Cs, Ca, Pa, phi);
    return result;
  }

  template <typename T>
  py::array_t<T> get_ctf_simple_real_q(py::array_t<T> q,
                                       T l,
                                       T df,
                                       T Cs,
                                       T Ca,
                                       T Pa,
                                       T phi = 0) {
    py::array_t<T> result(detail::get_shape(q));
    get_ctf_n_simple_real(
      q.data(), result.mutable_data(), q.size(), l, df, Cs, Ca, Pa, phi);
    return result;
  }

  template <typename T>
  py::array_t<T> get_ctf_simple_imag_q(py::array_t<T> q,
                                       T l,
                                       T df,
                                       T Cs,
                                       T Ca,
                                       T Pa,
                                       T phi = 0) {
    py::array_t<T> result(detail::get_shape(q));
    get_ctf_n_simple_imag(
      q.data(), result.mutable_data(), q.size(), l, df, Cs, Ca, Pa, phi);
    return result;
  }

  template <typename T>
  py::array_t<std::complex<T>> get_ctf_wh(std::size_t w,
                                          std::size_t h,
                                          T ps,
                                          T l,
                                          T df,
                                          T Cs,
                                          T Ca,
                                          T Pa,
                                          T dd,
                                          T theta_c,
                                          T phi = 0) {
    py::array_t<std::complex<T>> result({h, w});
    get_ctf_n(
      result.mutable_data(), w, h, ps, l, df, Cs, Ca, Pa, dd, theta_c, phi);
    return result;
  }

  template <typename T>
  py::array_t<std::complex<T>> get_ctf_simple_wh(std::size_t w,
                                                 std::size_t h,
                                                 T ps,
                                                 T l,
                                                 T df,
                                                 T Cs,
                                                 T Ca,
                                                 T Pa,
                                                 T phi = 0) {
    py::array_t<std::complex<T>> result({h, w});
    get_ctf_n_simple(result.mutable_data(), w, h, ps, l, df, Cs, Ca, Pa, phi);
    return result;
  }

  template <typename T>
  py::array_t<T> get_ctf_simple_real_wh(std::size_t w,
                                        std::size_t h,
                                        T ps,
                                        T l,
                                        T df,
                                        T Cs,
                                        T Ca,
                                        T Pa,
                                        T phi = 0) {
    py::array_t<T> result({h, w});
    get_ctf_n_simple_real(
      result.mutable_data(), w, h, ps, l, df, Cs, Ca, Pa, phi);
    return result;
  }

  template <typename T>
  py::array_t<T> get_ctf_simple_imag_wh(std::size_t w,
                                        std::size_t h,
                                        T ps,
                                        T l,
                                        T df,
                                        T Cs,
                                        T Ca,
                                        T Pa,
                                        T phi = 0) {
    py::array_t<T> result({h, w});
    get_ctf_n_simple_imag(
      result.mutable_data(), w, h, ps, l, df, Cs, Ca, Pa, phi);
    return result;
  }

}}  // namespace guanaco::python

inline void export_ctf(py::module m) {
  // Export the function to compute electron wavelength
  m.def("get_electron_wavelength",
        &guanaco::get_electron_wavelength<double>,
        py::arg("V"));

  // Export the function to compute defocus spread
  m.def("get_defocus_spread",
        &guanaco::get_defocus_spread<double>,
        py::arg("Cc"),
        py::arg("dEE"),
        py::arg("dII"),
        py::arg("dVV"));

  // Export the function to compute the spatial coherence envelope
  m.def("get_Es",
        &guanaco::get_Es<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("dd") = 0,
        py::arg("theta_c") = 0);

  // Export the function to compute the temporal coherence envelope
  m.def("get_Et",
        &guanaco::get_Et<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("dd") = 0,
        py::arg("theta_c") = 0);

  // Convert spatial frequencies to dimensionless quantities
  m.def("q_to_Q",
        &guanaco::q_to_Q<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("Cs"));

  // Convert spatial frequencies from dimensionless quantities
  m.def("Q_to_q",
        &guanaco::Q_to_q<double>,
        py::arg("Q"),
        py::arg("l"),
        py::arg("Cs"));

  // Convert defocus to dimensionless quantities
  m.def("df_to_D",
        &guanaco::df_to_D<double>,
        py::arg("df"),
        py::arg("l"),
        py::arg("Cs"));

  // Convert defocus from dimensionless quantities
  m.def("D_to_df",
        &guanaco::D_to_df<double>,
        py::arg("D"),
        py::arg("l"),
        py::arg("Cs"));

  // Get Chi
  m.def("get_chi",
        &guanaco::get_chi<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0);

  // Get CTF
  m.def("get_ctf",
        &guanaco::python::get_ctf_q<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs"),
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("dd") = 0,
        py::arg("theta_c") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple",
        &guanaco::python::get_ctf_simple_q<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple_real",
        &guanaco::python::get_ctf_simple_real_q<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple_imag",
        &guanaco::python::get_ctf_simple_imag_q<double>,
        py::arg("q"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);

  // Get the spatial frequency and angle
  m.def("get_q_and_theta",
        &guanaco::python::get_q_and_theta<double>,
        py::arg("w"),
        py::arg("h"),
        py::arg("ps"));

  // Get CTF
  m.def("get_ctf",
        &guanaco::python::get_ctf_wh<double>,
        py::arg("w"),
        py::arg("h"),
        py::arg("ps"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("dd") = 0,
        py::arg("theta_c") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple",
        &guanaco::python::get_ctf_simple_wh<double>,
        py::arg("w"),
        py::arg("h"),
        py::arg("ps"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple_real",
        &guanaco::python::get_ctf_simple_real_wh<double>,
        py::arg("w"),
        py::arg("h"),
        py::arg("ps"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);

  // Get CTF
  m.def("get_ctf_simple_imag",
        &guanaco::python::get_ctf_simple_imag_wh<double>,
        py::arg("w"),
        py::arg("h"),
        py::arg("ps"),
        py::arg("l"),
        py::arg("df"),
        py::arg("Cs") = 0,
        py::arg("Ca") = 0,
        py::arg("Pa") = 0,
        py::arg("phi") = 0);
}

#endif
