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
#include <cuda.h>
#include <cufft.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <guanaco/guanaco.h>

#define GUANACO_ASSERT_CUDA(assertion)                                   \
  if (!assertion) {                                                      \
    auto error = cudaGetLastError();                                     \
    throw guanaco::Error(__FILE__, __LINE__, cudaGetErrorString(error)); \
  }

namespace guanaco {

template <>
class FFT<e_device> {
public:
  using size_type = std::size_t;

  FFT(size_type size, size_type nbatch) : plan_forward_(0), plan_inverse_(0) {
    create_plan(size, nbatch);
  }

  ~FFT() {
    destroy_plan();
  }

  void forward(void *in, void *out) const {
    auto i = reinterpret_cast<cufftReal *>(in);
    auto o = reinterpret_cast<cufftComplex *>(out);
    cufftExecR2C(plan_forward_, i, o);
  }

  void inverse(void *in, void *out) const {
    auto i = reinterpret_cast<cufftComplex *>(in);
    auto o = reinterpret_cast<cufftReal *>(out);
    cufftExecC2R(plan_inverse_, i, o);
  }

protected:
  void create_plan(size_type size, size_type nbatch) {
    cufftPlan1d(&plan_forward_, size, CUFFT_R2C, nbatch);
    cufftPlan1d(&plan_inverse_, size, CUFFT_C2R, nbatch);
  }

  void destroy_plan() {
    cudaDeviceSynchronize();
    cufftDestroy(plan_forward_);
    cufftDestroy(plan_inverse_);
  }

  cufftHandle plan_forward_;
  cufftHandle plan_inverse_;
};

template <>
class Filter<e_device> {
public:
  using size_type = std::size_t;

  using vector_type = std::vector<float>;

  Filter(size_type num_pixels, size_type num_angles);

  void operator()(float *data) const;

  const vector_type &filter() const;

protected:
  vector_type create_filter(size_type size) const;

  size_type num_pixels_;
  size_type num_angles_;
  vector_type filter_;
  FFT<e_device> fft_;
};

Filter<e_device>::Filter(size_type num_pixels, size_type num_angles)
    : num_pixels_(num_pixels),
      num_angles_(num_angles),
      filter_(create_filter(num_pixels_ + 1)),
      fft_(num_pixels_ * 2, num_angles_) {
  GUANACO_ASSERT(num_pixels_ > 0);
  GUANACO_ASSERT(num_angles_ > 0);
}

const Filter<e_device>::vector_type &Filter<e_device>::filter() const {
  return filter_;
}

Filter<e_device>::vector_type Filter<e_device>::create_filter(size_type size) const {
  GUANACO_ASSERT(size > 0);

  // Initialise the filter array
  auto filter = vector_type(size, 0);

  // Create a Ramp filter
  for (auto i = 0; i < size; ++i) {
    float w = float(i) / float(size - 1);
    filter[i] = w;
  }

  // Return the filter
  return filter;
}

void Filter<e_device>::operator()(float *data) const {
  GUANACO_ASSERT(filter_.size() == num_pixels_ + 1);

  // Make some typedefs
  using device_vector_f = thrust::device_vector<float>;
  using device_vector_c = thrust::device_vector<thrust::complex<float>>;

  // Copy the filter to the device for each projection
  auto filter_d = device_vector_c(num_angles_ * filter_.size(), 0);
  for (auto i = 0; i < num_angles_; ++i) {
    thrust::copy(filter_.begin(), filter_.end(), filter_d.begin() + i * filter_.size());
  }

  // Copy the rows of the sinogram to a zero padded array. When taking the FT
  // of the data, we are going from real to complex so the output array only
  // stores the non-redundant complex coefficients so the complex array
  // requires (N/2 + 1) elements.
  auto rows_c = device_vector_c(num_angles_ * filter_.size(), 0);
  auto rows_f = device_vector_f(num_angles_ * num_pixels_ * 2, 0);
  for (auto i = 0; i < num_angles_; ++i) {
    thrust::copy(data + i * num_pixels_,
                 data + i * num_pixels_ + num_pixels_,
                 rows_f.begin() + i * num_pixels_ * 2);
  }

  // Take the FT of the rows of the data
  fft_.forward(rows_f.data().get(), rows_c.data().get());

  // Apply the filter
  thrust::transform(filter_d.begin(),
                    filter_d.end(),
                    rows_c.begin(),
                    rows_c.begin(),
                    thrust::multiplies<thrust::complex<float>>());

  // Take the inverse FT of the rows of the data
  fft_.inverse(rows_c.data().get(), rows_f.data().get());

  // Scale the filtered data
  auto factor = num_pixels_ * 2;
  thrust::transform(
    rows_f.begin(), rows_f.end(), rows_f.begin(), [factor] __device__(auto x) {
      return x / factor;
    });

  // Copy the filtered data back into the array
  for (int i = 0; i < num_angles_; ++i) {
    thrust::copy(rows_f.begin() + i * num_pixels_ * 2,
                 rows_f.begin() + i * num_pixels_ * 2 + num_pixels_,
                 data + i * num_pixels_);
  }
}

namespace detail {

  // Declare global variables
  namespace global {

    const size_t MAX_ANGLES = 2048;

    typedef texture<float, 2, cudaReadModeElementType> texture_type;

    static texture_type sinogram;

    __constant__ float angle_param_b[MAX_ANGLES];
    __constant__ float angle_param_a[MAX_ANGLES];
    __constant__ float angle_offset[MAX_ANGLES];
    __constant__ float angle_scale[MAX_ANGLES];

  }  // namespace global

  namespace g = global;

  /**
   * A functor that implements the back projection per voxel
   */
  struct BPFunction {
    static const size_t max_angles = g::MAX_ANGLES;

    size_t num_angles;
    size_t grid_width;
    size_t grid_height;
    float scale;

    BPFunction(size_t num_angles_, size_t grid_width_, size_t grid_height_, float scale_)
        : num_angles(num_angles_), grid_width(grid_width_), grid_height(grid_height_), scale(scale_) {
      GUANACO_ASSERT(num_angles_ <= max_angles);
    }

    __device__ float operator()(size_t index, float voxel) const {
      // Get the X and Y indices
      const size_t j = index / grid_width;
      const size_t i = index - j * grid_width;

      // Compute the x and y coordinates
      const float x = (i - 0.5 * grid_width + 0.5);
      const float y = (j - 0.5 * grid_height + 0.5);

      // Loop through all the angles and compute the value of the voxel
      float value = 0.0f;
      for (size_t angle = 0; angle < num_angles; ++angle) {

        // Get parameters
        const float a = g::angle_param_a[angle];
        const float b = g::angle_param_b[angle];
        const float c = g::angle_offset[angle];
        const float scale = g::angle_scale[angle];

        // Compute the pixel coordinate
        const float pixel = a * x + b * y + c;

        // Sum the sinogram value for the pixel and angle
        value += tex2D(g::sinogram, pixel, angle + 0.5) * scale;
      }

      // Add the contribution to the voxel
      return voxel + value * scale;
    }
  };

  struct BP {
    using size_type = std::size_t;

    static const size_type max_angles = BPFunction::max_angles;

    float *pitched_sinogram_;
    size_type pitch_;
    size_type num_angles_;

    BP(const float *angles,
       size_type num_angles,
       float centre,
       const float *sinogram,
       size_type num_pixels)
        : pitched_sinogram_(nullptr), num_angles_(num_angles) {
      // Check number of angles
      GUANACO_ASSERT(num_angles <= max_angles);

      // Allocate a pitched array needed to bind texture
      auto error = cudaMallocPitch(
        &pitched_sinogram_, &pitch_, sizeof(float) * num_pixels, num_angles);
      GUANACO_ASSERT_CUDA(error == cudaSuccess);
      GUANACO_ASSERT((pitch_ % sizeof(float)) == 0);
      pitch_ /= sizeof(float);

      // Copy the angle data to device symbols
      copy_angles(angles, num_angles_, centre);

      // Copy the sinogram to the texture memory
      copy_sinogram(sinogram, num_pixels, num_angles);
    }

    ~BP() {
      cudaFree(pitched_sinogram_);
    }

    void copy_angles(const float *angles, size_type num_angles, float centre) const {
      // Copy the data to the symbol. For some reason I can't pass the symbol
      // pointer as normal (no idea) so I have to pass a pointer to the
      // symbol array pointer and then dereference!
      auto copy = [](auto symbol, auto data, auto n) {
        GUANACO_ASSERT(n <= max_angles);
        auto error = cudaMemcpyToSymbol(
          *symbol, data, n * sizeof(float), 0, cudaMemcpyHostToDevice);
        GUANACO_ASSERT_CUDA(error == cudaSuccess);
      };

      // Allocate host vectors
      auto angle_param_b = thrust::host_vector<float>(num_angles);
      auto angle_param_a = thrust::host_vector<float>(num_angles);
      auto angle_offset = thrust::host_vector<float>(num_angles);
      auto angle_scale = thrust::host_vector<float>(num_angles);

      // Compute the quanities to store in the symbols
      for (auto i = 0; i < num_angles; ++i) {
        // Get the ray vector and length of the ray vector
        auto angle = angles[i];
        auto dir_x = std::cos(angle);
        auto dir_y = std::sin(angle);
        auto det_x0 = -centre * dir_x;
        auto det_y0 = -centre * dir_y;
        auto ray_length = 1.0;//std::sqrt(dir_x * dir_x + (-dir_y) * (-dir_y));
        auto d = 1.0;//dir_x * dir_x - dir_y * (-dir_y);

        // Fill the arrays
        angle_param_a[i] = dir_x / d;
        angle_param_b[i] = (-dir_y) / d;
        angle_offset[i] = (det_y0 * (-dir_y) - det_x0 * dir_x) / d;
        angle_scale[i] = ray_length / std::abs(d);
      }

      // Copy the arrays to the symbols
      copy(&g::angle_param_b, angle_param_b.data(), num_angles);
      copy(&g::angle_param_a, angle_param_a.data(), num_angles);
      copy(&g::angle_offset, angle_offset.data(), num_angles);
      copy(&g::angle_scale, angle_scale.data(), num_angles);
    }

    void copy_sinogram(const float *sinogram, size_type num_pixels, size_type num_angles) {
      auto channel_desc = cudaCreateChannelDesc<float>();

      g::sinogram.addressMode[0] = cudaAddressModeBorder;
      g::sinogram.addressMode[1] = cudaAddressModeBorder;
      g::sinogram.filterMode = cudaFilterModeLinear;
      g::sinogram.normalized = false;

      // Copy the sinogram to a pitched array needed for texture binding
      for (auto i = 0; i < num_angles; ++i) {
        auto in = thrust::device_pointer_cast(sinogram + i * num_pixels);
        auto out = thrust::device_pointer_cast(pitched_sinogram_ + i * pitch_);
        thrust::copy(in, in + num_pixels, out);
      }

      auto error = cudaBindTexture2D(0,
                                     g::sinogram,
                                     pitched_sinogram_,
                                     channel_desc,
                                     num_pixels,
                                     num_angles,
                                     pitch_ * sizeof(float));

      GUANACO_ASSERT_CUDA(error == cudaSuccess);
    }

    void launch(float *reconstruction,
                size_type grid_width,
                size_type grid_height,
                float scale) const {
      auto grid_size = grid_width * grid_height;
      auto index = thrust::counting_iterator<size_t>(0);
      auto recon = thrust::device_pointer_cast(reconstruction);
      BPFunction func(num_angles_, grid_width, grid_height, scale);
      thrust::transform(index, index + grid_size, recon, recon, func);
    }
  };

}  // namespace detail

Reconstructor_t<e_device>::Reconstructor_t(const Config &config) : config_(config) {
  GUANACO_ASSERT(config_.device == e_device);
  GUANACO_ASSERT(config_.is_valid());
}

void Reconstructor_t<e_device>::operator()(const float *sinogram,
                                           float *reconstruction) const {
  Filter<e_device> filter_(config_.num_pixels, config_.num_angles);

  // A function to set the gpu index
  auto set_gpu_index = [](int index) {
    if (index >= 0) {
      cudaSetDevice(index);
      auto error = cudaGetLastError();
      GUANACO_ASSERT_CUDA((error == cudaSuccess)
                          || (error == cudaErrorSetOnActiveProcess));
    }
  };

  // Make some typedefs
  using vector_type = thrust::device_vector<float>;

  // Get the sinogram and reconstruction sizes along with the number of
  // angles and the pixel area
  auto sino_size = config_.sino_size();
  auto grid_size = config_.grid_size();
  auto num_angles = config_.num_angles;
  auto pixel_area = config_.pixel_area();

  // Allocate device vectors for sinogram and reconstruction
  auto sinogram_d = vector_type(sinogram, sinogram + sino_size);
  auto reconstruction_d = vector_type(grid_size, 0);

  // Set the gpu
  set_gpu_index(config_.gpu_index);

  // Filter the sinogram
  filter_(sinogram_d.data().get());

  // Perform the backprojection
  project(sinogram_d.data().get(), reconstruction_d.data().get());

  // Copy the data back to the host ptr
  thrust::copy(reconstruction_d.begin(), reconstruction_d.end(), reconstruction);
}

void Reconstructor_t<e_device>::project(const float *sinogram_d,
                                        float *reconstruction_d) const {
  auto scale = M_PI / (2 * config_.num_angles * config_.pixel_area());
  scale *= config_.pixel_size;  // ONLY VALID FOR SQUARE

  // Loop through the number of angles and only do the max number of
  // projections at a time
  for (auto start_angle = 0; start_angle < config_.num_angles;
       start_angle += detail::BP::max_angles) {
    auto end_angle =
      std::min(start_angle + detail::BP::max_angles, config_.num_angles);
    auto num_angles = end_angle - start_angle;
    auto angles = config_.angles.data() + start_angle;
    auto sino = sinogram_d + start_angle * config_.num_pixels;
    auto bp =
      detail::BP(angles, num_angles, config_.centre, sino, config_.num_pixels);
    bp.launch(reconstruction_d, config_.grid_width, config_.grid_height, scale);
  }
}

}  // namespace guanaco
