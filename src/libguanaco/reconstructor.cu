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
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <guanaco/guanaco.h>

namespace guanaco {

namespace detail {

  // Declare global variables
  namespace global {

    const size_t MAX_ANGLES = 4096;

    typedef texture<float, 3, cudaReadModeElementType> texture_type;

    static texture_type sinogram;

    __constant__ float angle_sin[MAX_ANGLES];
    __constant__ float angle_cos[MAX_ANGLES];
    __constant__ float angle_offset[MAX_ANGLES];
    __constant__ float angle_scale[MAX_ANGLES];

  }  // namespace global

  namespace g = global;

  /**
   * A functor that implements the back projection per voxel
   */
  struct BPFunction {
    size_t num_angles;
    size_t grid_width;
    size_t grid_height;
    float output_scale;
    float dscale;
    float doffset;

    BPFunction(size_t num_angles_,
               size_t grid_width_,
               size_t grid_height_,
               float output_scale_,
               float dscale_,
               float doffset_)
        : num_angles(num_angles_),
          grid_width(grid_width_),
          grid_height(grid_height_),
          output_scale(output_scale_),
          dscale(dscale_),
          doffset(doffset_) {
      GUANACO_ASSERT(num_angles_ <= g::MAX_ANGLES);
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
        const float cos_angle = g::angle_cos[angle];
        const float sin_angle = g::angle_sin[angle];
        const float offset = g::angle_offset[angle];
        const float scale = g::angle_scale[angle];

        // Compute the pixel and defocus coordinate
        const float pixel = cos_angle * x - sin_angle * y + offset;
        const float height = -sin_angle * x - cos_angle * y;
        const float defocus = height * dscale + doffset;

        // Sum the sinogram value for the pixel and angle
        value += tex3D(g::sinogram, pixel, angle + 0.5, defocus) * scale;
      }

      // Add the contribution to the voxel
      return voxel + value * output_scale;
    }
  };

  struct BP {
    using size_type = std::size_t;

    cudaArray *sinogram_array_;
    size_type num_pixels_;
    size_type num_angles_;
    size_type num_defocus_;
    float pixel_size_;
    float min_defocus_;
    float max_defocus_;

    BP(size_type num_pixels,
       size_type num_angles,
       size_type num_defocus,
       float centre,
       float pixel_size,
       float min_defocus,
       float max_defocus,
       const float *sinogram,
       const float *angles)
        : sinogram_array_(nullptr),
          num_pixels_(num_pixels),
          num_angles_(num_angles),
          num_defocus_(num_defocus),
          pixel_size_(pixel_size),
          min_defocus_(min_defocus),
          max_defocus_(max_defocus) {
      // Check input
      GUANACO_ASSERT(num_pixels_ > 0);
      GUANACO_ASSERT(num_angles_ > 0);
      GUANACO_ASSERT(num_defocus_ > 0);
      GUANACO_ASSERT(pixel_size_ > 0);
      GUANACO_ASSERT(max_defocus_ >= min_defocus_);

      // Copy the angle data to device symbols
      copy_angles(angles, num_angles_, centre);

      // Copy the sinogram to the texture memory
      copy_sinogram(sinogram, num_pixels, num_angles, num_defocus);
    }

    ~BP() {
      cudaFreeArray(sinogram_array_);
    }

    void copy_angles(const float *angles,
                     size_type num_angles,
                     float centre) const {
      // Copy the data to the symbol. For some reason I can't pass the symbol
      // pointer as normal (no idea) so I have to pass a pointer to the
      // symbol array pointer and then dereference!
      auto copy = [](auto symbol, auto data, auto n) {
        GUANACO_ASSERT(n <= g::MAX_ANGLES);
        auto error = cudaMemcpyToSymbol(
          *symbol, data, n * sizeof(float), 0, cudaMemcpyHostToDevice);
        GUANACO_ASSERT_CUDA(error == cudaSuccess);
      };

      // Allocate host vectors
      auto angle_sin = thrust::host_vector<float>(num_angles);
      auto angle_cos = thrust::host_vector<float>(num_angles);
      auto angle_offset = thrust::host_vector<float>(num_angles);
      auto angle_scale = thrust::host_vector<float>(num_angles);

      // Compute the quanities to store in the symbols
      for (auto i = 0; i < num_angles; ++i) {
        // Get the ray vector and length of the ray vector
        auto angle = angles[i];
        auto cos_angle = std::cos(angle);
        auto sin_angle = std::sin(angle);
        auto det_x0 = -centre * cos_angle;
        auto det_y0 = -centre * sin_angle;
        auto ray_length =
          std::sqrt(cos_angle * cos_angle + sin_angle * sin_angle);
        auto d = cos_angle * cos_angle + sin_angle * sin_angle;

        // Fill the arrays
        angle_cos[i] = cos_angle / d;
        angle_sin[i] = sin_angle / d;
        angle_offset[i] = (det_y0 * (-sin_angle) - det_x0 * cos_angle) / d;
        angle_scale[i] = ray_length / std::abs(d);
      }

      // Copy the arrays to the symbols
      copy(&g::angle_sin, angle_sin.data(), num_angles);
      copy(&g::angle_cos, angle_cos.data(), num_angles);
      copy(&g::angle_offset, angle_offset.data(), num_angles);
      copy(&g::angle_scale, angle_scale.data(), num_angles);
    }

    void copy_sinogram(const float *sinogram,
                       size_type num_pixels,
                       size_type num_angles,
                       size_type num_defocus) {
      // Allocate a cuda array needed to bind 3D texture
      auto channel_desc = cudaCreateChannelDesc<float>();
      auto extent = make_cudaExtent(num_pixels, num_angles, num_defocus);
      auto error = cudaMalloc3DArray(&sinogram_array_, &channel_desc, extent);
      GUANACO_ASSERT_CUDA(error == cudaSuccess);

      // Copy the data
      cudaMemcpy3DParms copy_params{0};
      copy_params.srcPtr = make_cudaPitchedPtr((void *)sinogram,
                                               extent.width * sizeof(float),
                                               extent.width,
                                               extent.height);
      copy_params.dstArray = sinogram_array_;
      copy_params.extent = extent;
      copy_params.kind = cudaMemcpyDeviceToDevice;
      cudaMemcpy3D(&copy_params);

      // Set texture parameters.
      // For examples and pixels outside the expected range, this sets the
      // value to zero (border). For defocus outside of expected range, use the
      // closest (clamp).
      g::sinogram.addressMode[0] = cudaAddressModeBorder;
      g::sinogram.addressMode[1] = cudaAddressModeBorder;
      g::sinogram.addressMode[2] = cudaAddressModeClamp;
      g::sinogram.filterMode = cudaFilterModeLinear;
      g::sinogram.normalized = false;

      // Bind the texture to the array
      error =
        cudaBindTextureToArray(g::sinogram, sinogram_array_, channel_desc);
      GUANACO_ASSERT_CUDA(error == cudaSuccess);
    }

    void launch(float *reconstruction,
                size_type grid_width,
                size_type grid_height,
                float scale) const {
      // Check the input
      GUANACO_ASSERT(num_defocus_ == 1 || max_defocus_ > min_defocus_);

      // Compute the defocus scale and offset
      auto dscale = num_defocus_ > 1 ? num_defocus_ * pixel_size_
                                         / (max_defocus_ - min_defocus_)
                                     : 0;
      auto doffset = -dscale * (min_defocus_ / pixel_size_);

      // Get some other quantities
      auto grid_size = grid_width * grid_height;
      auto index = thrust::counting_iterator<size_t>(0);
      auto recon = thrust::device_pointer_cast(reconstruction);

      // Initialise the functor
      BPFunction func(
        num_angles_, grid_width, grid_height, scale, dscale, doffset);

      // Do the reconstruction
      thrust::transform(index, index + grid_size, recon, recon, func);
    }
  };

}  // namespace detail

template <>
Reconstructor_t<e_device>::Reconstructor_t(const Config &config)
    : config_(config),
      filter_(config_.num_pixels, config_.num_angles, config_.num_defocus) {
  GUANACO_ASSERT(config_.device == e_device);
  GUANACO_ASSERT(config_.is_valid());
}

template <>
void Reconstructor_t<e_device>::project(const float *sinogram,
                                        float *reconstruction) const {
  // Check the dimensions against the maximum texture size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  GUANACO_ASSERT(prop.maxTexture3D[1] >= detail::g::MAX_ANGLES);
  GUANACO_ASSERT(prop.maxTexture3D[0] >= config_.num_pixels);
  GUANACO_ASSERT(prop.maxTexture3D[1] >= config_.num_angles);
  GUANACO_ASSERT(prop.maxTexture3D[2] >= config_.num_defocus);

  // Compute the scale
  auto scale = M_PI / (2 * config_.num_angles);

  // Initialise the back projector class
  auto bp = detail::BP(config_.num_pixels,
                       config_.num_angles,
                       config_.num_defocus,
                       config_.centre,
                       config_.pixel_size,
                       config_.min_defocus,
                       config_.max_defocus,
                       sinogram,
                       config_.angles.data());

  // Launch the back projector
  bp.launch(reconstruction, config_.grid_width, config_.grid_height, scale);
}

template <>
void Reconstructor_t<e_device>::operator()(const float *sinogram,
                                           float *reconstruction) const {
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
  thrust::copy(
    reconstruction_d.begin(), reconstruction_d.end(), reconstruction);
}

}  // namespace guanaco
