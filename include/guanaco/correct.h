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
#ifndef GUANACO_CORRECT_H
#define GUANACO_CORRECT_H

#include <complex>
#include <guanaco/constants.h>
#include <guanaco/system.h>

namespace guanaco {

template <typename T>
HOST_DEVICE constexpr int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename ComplexType>
HOST_DEVICE constexpr ComplexType phase_flip(const ComplexType &x,
                                             const ComplexType &h) {
  return -x * ComplexType(sign(h.imag()));
}

namespace detail {

  template <eDevice device, typename T>
  struct Corrector;

  template <typename T>
  struct Corrector<e_host, T> {
    void correct(const T *image,
                 const std::complex<T> *ctf,
                 T *rec,
                 std::size_t xsize,
                 std::size_t ysize,
                 std::size_t num_ctf = 1);
  };

  template <typename T>
  struct Corrector<e_device, T> {
    void correct(const T *image,
                 const std::complex<T> *ctf,
                 T *rec,
                 std::size_t xsize,
                 std::size_t ysize,
                 std::size_t num_ctf = 1);
  };

  template <eDevice device, typename T>
  void correct_internal(const T *image,
                        const std::complex<T> *ctf,
                        T *rec,
                        std::size_t xsize,
                        std::size_t ysize,
                        std::size_t num_ctf = 1) {
    Corrector<device, T>().correct(image, ctf, rec, xsize, ysize, num_ctf);
  }
}  // namespace detail

template <typename T>
void correct(const T *image,
             const std::complex<T> *ctf,
             T *rec,
             std::size_t xsize,
             std::size_t ysize,
             std::size_t num_ctf = 1,
             eDevice device = e_host) {
  switch (device) {
  case e_device:
    detail::correct_internal<e_device, T>(
      image, ctf, rec, xsize, ysize, num_ctf);
    break;
  case e_host:
  default:
    detail::correct_internal<e_host, T>(image, ctf, rec, xsize, ysize, num_ctf);
    break;
  };
}

}  // namespace guanaco

#endif
