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
#include <pybind11/numpy.h>
#include <fftw3.h>

#include <guanaco/error.h>

namespace py = pybind11;

namespace guanaco {

enum eDevice { e_host = 0, e_device = 1 };

struct Config {
  using size_type = std::size_t;

  eDevice device;
  int gpu_index;
  size_type num_pixels;
  size_type num_angles;
  size_type grid_height;
  size_type grid_width;
  float pixel_size;
  float centre;
  std::vector<float> angles;

  Config()
      : device(e_host),
        gpu_index(0),
        num_pixels(0),
        num_angles(0),
        grid_height(0),
        grid_width(0),
        pixel_size(1),
        centre(0) {}

  size_type sino_size() const {
    return num_pixels * num_angles;
  }

  size_type grid_size() const {
    return grid_height * grid_width;
  }

  float pixel_area() const {
    return pixel_size * pixel_size;
  }

  bool is_valid() const {
    return (device == e_host || device == e_device) && grid_height > 0
           && grid_width > 0 && num_pixels > 0 && pixel_size > 0
           && num_angles > 0
           && angles.size() == num_angles;
  }
};

template <eDevice device>
class FFT;

template <>
class FFT<e_host> {
public:
  using size_type = std::size_t;

  FFT(size_type size, size_type num_threads)
      : plan_forward_(nullptr), plan_inverse_(nullptr) {
    create_plan(size, num_threads);
  }

  ~FFT() {
    destroy_plan();
  }

  void forward(float *data) const {
    fftwf_complex *v = reinterpret_cast<fftwf_complex *>(data);
    fftwf_execute_dft(plan_forward_, v, v);
  }

  void inverse(float *data) const {
    fftwf_complex *v = reinterpret_cast<fftwf_complex *>(data);
    fftwf_execute_dft(plan_inverse_, v, v);
  }

protected:
  void create_plan(size_type size, size_type num_threads) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(num_threads);
    auto data = (fftwf_complex *)fftw_malloc(sizeof(fftwf_complex) * size);
    plan_forward_ = fftwf_plan_dft_1d(size, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse_ = fftwf_plan_dft_1d(size, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_free(data);
  }

  void destroy_plan() {
    if (plan_forward_ != nullptr) {
      fftwf_destroy_plan(plan_forward_);
      plan_forward_ = nullptr;
    }
    if (plan_inverse_ != nullptr) {
      fftwf_destroy_plan(plan_inverse_);
      plan_inverse_ = nullptr;
    }
    fftwf_cleanup_threads();
  }

  fftwf_plan plan_forward_;
  fftwf_plan plan_inverse_;
};

template <eDevice device>
class Filter;

template <>
class Filter<e_host> {
public:
  using size_type = std::size_t;

  using vector_type = std::vector<float>;
  using complex_vector_type = std::vector<std::complex<float>>;

  Filter(size_type num_pixels, size_type num_angles);
  void operator()(float *data) const;

  const vector_type &filter() const;

protected:
  std::vector<float> create_filter(size_type size) const;

  size_type num_pixels_;
  size_type num_angles_;
  vector_type filter_;
  FFT<e_host> fft_;
};

template <eDevice device>
class Reconstructor_t {};

template <>
class Reconstructor_t<e_host> {
public:
  Reconstructor_t(const Config &config);

  void operator()(float *sinogram, float *reconstruction) const;

protected:
  void project(float *sinogram, float *reconstruction) const;

  Config config_;
  Filter<e_host> filter_;
};

template <>
class Reconstructor_t<e_device> {
public:
  Reconstructor_t(const Config &config);

  void operator()(float *sinogram, float *reconstruction) const;

protected:
  void project(float *sinogram, float *reconstruction) const;

  Config config_;
};

class Reconstructor {
public:
  using size_type = std::size_t;

  Reconstructor(const Config &config) : config_(config) {}

  void operator()(float *sinogram, float *reconstruction) const {
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
void reconstruct(py::array_t<T> sinogram,
                 py::array_t<T> reconstruction,
                 py::array_t<T> angles,
                 float centre = 0,
                 float pixel_size = 1,
                 std::string device = "cpu",
                 int gpu_index = -1) {

  // Check the input
  GUANACO_ASSERT(sinogram.ndim() == 2);
  GUANACO_ASSERT(reconstruction.ndim() == 2);
  GUANACO_ASSERT(angles.ndim() == 1);
  GUANACO_ASSERT(sinogram.shape()[0] == angles.size());
  GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[0]);
  GUANACO_ASSERT(sinogram.shape()[1] == reconstruction.shape()[1]);
/* template <typename T> */
/* void rec_temp(std::string devstr, */
/*               std::size_t grid_height, */
/*               std::size_t grid_width, */
/*               float pixel_size, */
/*               std::size_t num_pixels, */
/*               py::array_t<float> &angles, */
/*               float centre, */
/*               py::array_t<float> &sino, */
/*               py::array_t<float> &rec, */
/*               int gpu_index) { */

  // Initialise the configuration
  auto args = [&] {
    auto c = Config();
    c.device = device == "cpu" ? e_host : e_device;
    c.gpu_index = gpu_index;
    c.num_pixels = sinogram.shape()[1];
    c.num_angles = sinogram.shape()[0];
    c.grid_width = reconstruction.shape()[1];
    c.grid_height = reconstruction.shape()[0];
    c.pixel_size = pixel_size;
    c.centre = centre;
    c.angles.assign(angles.data(), angles.data() + angles.size());
    return c;
  }();

  // Create the reconstructor object
  auto rec = make_reconstructor(args);

  // Perform the reconstruction
  rec(sinogram.mutable_data(), reconstruction.mutable_data());
}

}  // namespace guanaco
