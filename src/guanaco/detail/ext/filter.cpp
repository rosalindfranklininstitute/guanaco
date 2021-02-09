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
#include <guanaco/filter.h>

namespace guanaco {

template <eDevice device>
const std::vector<float> &Filter<device>::filter() const {
  return filter_;
}

template <eDevice device>
std::vector<float> Filter<device>::create_filter(size_type size) const {
  GUANACO_ASSERT(size > 0);

  // Initialise the filter array
  auto filter = vector_type(size, 0);

  // Create a Ramp filter
  for (auto i = 0; i < size; i++) {
    float w = float(i) / float(size - 1);
    filter[i] = w;
  }

  // Return the filter
  return filter;
}

template <>
Filter<e_host>::Filter(size_type num_pixels,
                       size_type num_angles,
                       size_type num_defocus)
    : num_pixels_(num_pixels),
      num_angles_(num_angles),
      num_defocus_(num_defocus),
      filter_(create_filter(num_pixels_ + 1)),
      fft_(num_pixels_ * 2) {
  GUANACO_ASSERT(num_pixels_ > 0);
  GUANACO_ASSERT(num_angles_ > 0);
  GUANACO_ASSERT(num_defocus_ > 0);
}

template <>
void Filter<e_host>::operator()(float *data) const {
  GUANACO_ASSERT(data != NULL);
  GUANACO_ASSERT(filter_.size() == num_pixels_ + 1);

  // Allocate a complex buffer for the row
  auto row = complex_vector_type(num_pixels_ * 2);

  // Loop through all the projection images
  for (auto i = 0; i < num_defocus_ * num_angles_; ++i) {
    // Get a row of data
    float *data_row = data + i * num_pixels_;

    // Copy the row into the zero padded array
    std::copy(data_row, data_row + num_pixels_, row.begin());
    std::fill(row.begin() + num_pixels_, row.end(), 0);

    // Take the FT of the row
    fft_.forward(reinterpret_cast<float *>(row.data()));

    // Apply the filter
    for (auto j = 0; j < filter_.size(); ++j) {
      row[j] *= filter_[j];
    }
    for (auto j = filter_.size(); j < row.size(); ++j) {
      row[j] *= filter_[row.size() - j];
    }

    // Take the inverse FT of the row
    fft_.inverse(reinterpret_cast<float *>(row.data()));

    // Copy the filtered data back
    for (auto j = 0; j < num_pixels_; ++j) {
      data_row[j] = row[j].real() / row.size();
    }
  }
}

// Explicit instantiation needed due to non-specialized template functions
template class Filter<e_host>;
template class Filter<e_device>;
}  // namespace guanaco
