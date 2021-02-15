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
#ifndef GUANACO_RECONSTRUCTOR_H
#define GUANACO_RECONSTRUCTOR_H

namespace guanaco {

template <eDevice device>
class Reconstructor_t {
public:
  using size_type = std::size_t;
  Reconstructor_t(const Config &config);

  void operator()(const float *sinogram, float *reconstruction) const;

protected:
  void project(const float *sinogram, float *reconstruction) const;

  Config config_;
  Filter<device> filter_;
};

class Reconstructor {
public:
  using size_type = std::size_t;

  Reconstructor(const Config &config);

  void operator()(const float *sinogram, float *reconstruction) const;

protected:
  Config config_;
};

inline Reconstructor make_reconstructor(const Config &config) {
  return Reconstructor(config);
}

}  // namespace guanaco

#endif
