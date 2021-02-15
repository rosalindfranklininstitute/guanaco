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
#ifndef GUANACO_FFT_H
#define GUANACO_FFT_H

#include <memory>
#include <guanaco/constants.h>
#include <guanaco/error.h>

namespace guanaco {

template <eDevice device>
class FFT {
public:
  using size_type = std::size_t;

  FFT(size_type size);
  FFT(size_type xsize, size_type ysize);
  FFT(const FFT &);
  FFT(FFT &&);
  FFT &operator=(FFT &&);
  FFT &operator=(const FFT &);
  ~FFT();

  void forward(void *in, void *out) const;
  void inverse(void *in, void *out) const;
  void forward(void *data) const;
  void inverse(void *data) const;

  static FFT<device> make_1d(size_type size,
                             size_type nbatch = 1,
                             bool real = false);

  static FFT<device> make_2d(size_type xsize,
                             size_type ysize,
                             size_type nbatch = 1,
                             bool real = false);

protected:
  FFT();

  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace guanaco
#endif
