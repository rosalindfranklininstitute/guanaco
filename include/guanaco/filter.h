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
#ifndef GUANACO_FILTER_H
#define GUANACO_FILTER_H

#include <complex>
#include <vector>
#include <guanaco/constants.h>
#include <guanaco/error.h>
#include <guanaco/fft.h>

namespace guanaco {

template <eDevice device>
class Filter {
public:
  using size_type = std::size_t;

  using vector_type = std::vector<float>;
  using complex_vector_type = std::vector<std::complex<float>>;

  Filter(size_type num_pixels, size_type num_angles, size_type num_defocus);

  void operator()(float *data) const;

  const vector_type &filter() const;

protected:
  std::vector<float> create_filter(size_type size) const;

  size_type num_pixels_;
  size_type num_angles_;
  size_type num_defocus_;
  vector_type filter_;
  FFT<device> fft_;
};

}  // namespace guanaco

#endif
