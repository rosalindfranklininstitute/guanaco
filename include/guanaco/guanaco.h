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
#include <complex>
#include <cmath>
#include <vector>
#include <guanaco/constants.h>
#include <guanaco/error.h>
#include <guanaco/fft.h>

namespace guanaco {


struct Config {
  using size_type = std::size_t;

  eDevice device;
  int gpu_index;
  size_type num_pixels;
  size_type num_angles;
  size_type num_defocus;
  size_type grid_height;
  size_type grid_width;
  float pixel_size;
  float centre;
  std::vector<float> angles;
  float min_defocus;
  float max_defocus;

  Config()
      : device(e_host),
        gpu_index(0),
        num_pixels(0),
        num_angles(0),
        num_defocus(0),
        grid_height(0),
        grid_width(0),
        pixel_size(1),
        centre(0),
        min_defocus(0),
        max_defocus(0) {}

  size_type sino_size() const {
    return num_pixels * num_angles * num_defocus;
  }

  size_type grid_size() const {
    return grid_height * grid_width;
  }

  float pixel_area() const {
    return pixel_size * pixel_size;
  }

  bool is_valid() const {
    return (device == e_host || device == e_device) 
      && num_pixels > 0 
      && num_angles > 0
      && num_defocus > 0
      && grid_height > 0
      && grid_width > 0 
      && pixel_size > 0
      && min_defocus <= max_defocus
      && angles.size() == num_angles;
  }
};


template <eDevice device>
class Filter;

template <>
class Filter<e_host> {
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
  FFT<e_host> fft_;
};

template <eDevice device>
class Reconstructor_t {};

template <>
class Reconstructor_t<e_host> {
public:
  using size_type = std::size_t;
  Reconstructor_t(const Config &config);

  void operator()(const float *sinogram, float *reconstruction) const;

protected:
  void project(const float *sinogram, float *reconstruction) const;

  Config config_;
  Filter<e_host> filter_;
};

template <>
class Reconstructor_t<e_device> {
public:
  using size_type = std::size_t;
  Reconstructor_t(const Config &config);

  void operator()(const float *sinogram, float *reconstruction) const;

protected:
  void project(const float *sinogram, float *reconstruction) const;

  Config config_;
};

class Reconstructor {
public:
  using size_type = std::size_t;

  Reconstructor(const Config &config) : config_(config) {}

  void operator()(const float *sinogram, float *reconstruction) const {
    switch (config_.device) {
    case e_device: {
      auto alg = Reconstructor_t<e_device>(config_);
      alg(sinogram, reconstruction);
    } break;
    case e_host:
    default: {
      auto alg = Reconstructor_t<e_host>(config_);
      alg(sinogram, reconstruction);
    } break;
    };
  }

protected:
  Config config_;
};

inline Reconstructor make_reconstructor(const Config &config) {
  return Reconstructor(config);
}

template <typename T> 
int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
std::complex<T> phase_flip(std::complex<T> x, std::complex<T> h) {
  return x * T(sign(h.imag()));
}

template <typename T, eDevice device>
void correct_internal(
    const T *image, 
    const std::complex<T> *ctf, 
    T *rec, 
    std::size_t xsize,
    std::size_t ysize,
    std::size_t num_ctf = 1) {
  GUANACO_ASSERT(xsize > 0);
  GUANACO_ASSERT(ysize > 0);
  GUANACO_ASSERT(num_ctf > 0);
  
  using complex_vector_type = std::vector<std::complex<float>>;
     
  // Initialise the FFT
  auto fft = FFT<e_host>(xsize, ysize);

  // Allocate a complex buffer for the row
  auto row = complex_vector_type(xsize * ysize);

  // Loop through all the projections and all the CTFs
  for (auto j = 0; j < num_ctf; ++j) {

    // Get the CTF and output arrays
    auto c = ctf + j * ysize*xsize;
    auto r = rec + j * ysize*xsize;

    // Copy the row into the zero padded array
    std::copy(image, image + ysize*xsize, row.begin());

    // Perform the forward FT
    fft.forward(reinterpret_cast<float *>(row.data()));

    // Do the CTF correction
    for (auto k = 0; k < ysize*xsize; ++k) {
      row[k] = -phase_flip(row[k], c[k]);
    }
    
    // Perform the inverse FT
    fft.inverse(reinterpret_cast<float *>(row.data()));

    // Copy the data to the output array
    for (auto k = 0; k < ysize*xsize; ++k) {
      r[k] = row[k].real();
    }
  }
}

template <typename T>
void correct(
    const T *image, 
    const std::complex<T> *ctf, 
    T *rec, 
    std::size_t xsize,
    std::size_t ysize,
    std::size_t num_ctf = 1,
    eDevice device = e_host) {
  switch (e_host) {
  case e_device:
  case e_host:
  default:
    correct_internal<T, e_host>(image, ctf, rec, xsize, ysize, num_ctf);
  };
}

}  // namespace guanaco
