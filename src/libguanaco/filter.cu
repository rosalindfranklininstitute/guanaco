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
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <guanaco/filter.h>

namespace guanaco {

template <>
Filter<e_device>::Filter(size_type num_pixels,
                         size_type num_angles,
                         size_type num_defocus)
    : num_pixels_(num_pixels),
      num_angles_(num_angles),
      num_defocus_(num_defocus),
      filter_(create_filter(num_pixels_ + 1)),
      fft_(FFT<e_device>::make_1d(num_pixels_ * 2, num_angles_, true)) {
  GUANACO_ASSERT(num_pixels_ > 0);
  GUANACO_ASSERT(num_angles_ > 0);
  GUANACO_ASSERT(num_defocus_ > 0);
}

template <>
void Filter<e_device>::operator()(float *data) const {
  GUANACO_ASSERT(filter_.size() == num_pixels_ + 1);

  // Make some typedefs
  using device_vector_f = thrust::device_vector<float>;
  using device_vector_c = thrust::device_vector<thrust::complex<float>>;

  // Copy the filter to the device
  auto filter_d = device_vector_c(filter_.size(), 0);
  thrust::copy(filter_.begin(), filter_.end(), filter_d.begin());

  // When taking the FT of the data, we are going from real to complex so the
  // output array only stores the non-redundant complex coefficients so the
  // complex array requires (N/2 + 1) elements.
  for (auto j = 0; j < num_defocus_; ++j) {
    auto rows_c = device_vector_c(num_angles_ * filter_.size(), 0);
    auto rows_f = device_vector_f(num_angles_ * num_pixels_ * 2, 0);

    // Get a pointer to the sinogram
    auto data_ptr = data + j * num_angles_ * num_pixels_;

    // Copy the rows of the sinogram to a zero padded array.
    for (auto i = 0; i < num_angles_; ++i) {
      thrust::copy(data_ptr + i * num_pixels_,
                   data_ptr + i * num_pixels_ + num_pixels_,
                   rows_f.begin() + i * num_pixels_ * 2);
    }

    // Take the FT of the rows of the data
    fft_.forward(rows_f.data().get(), rows_c.data().get());

    // Apply the filter to each projection
    for (auto i = 0; i < num_angles_; ++i) {
      thrust::transform(filter_d.begin(),
                        filter_d.end(),
                        rows_c.begin() + i * filter_.size(),
                        rows_c.begin() + i * filter_.size(),
                        thrust::multiplies<thrust::complex<float>>());
    }

    // Take the inverse FT of the rows of the data
    fft_.inverse(rows_c.data().get(), rows_f.data().get());

    // Scale the filtered data
    auto factor = num_pixels_ * 2;
    thrust::transform(rows_f.begin(),
                      rows_f.end(),
                      rows_f.begin(),
                      [factor] __device__(auto x) { return x / factor; });

    // Copy the filtered data back into the array
    for (int i = 0; i < num_angles_; ++i) {
      thrust::copy(rows_f.begin() + i * num_pixels_ * 2,
                   rows_f.begin() + i * num_pixels_ * 2 + num_pixels_,
                   data_ptr + i * num_pixels_);
    }
  }
}

}  // namespace guanaco
