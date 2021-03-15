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
#include <guanaco/error.h>
#include <guanaco/fft.h>
#include <guanaco/correct.h>

namespace guanaco {
namespace detail {

  template <typename T>
  void Corrector<e_device, T>::correct(const T *image,
                                       const std::complex<T> *ctf,
                                       T *rec,
                                       std::size_t xsize,
                                       std::size_t ysize,
                                       std::size_t num_ctf) {
    GUANACO_ASSERT(xsize > 0);
    GUANACO_ASSERT(ysize > 0);
    GUANACO_ASSERT(num_ctf > 0);

    using vector_type = thrust::device_vector<T>;
    using complex_vector_type = thrust::device_vector<thrust::complex<T>>;

    // Initialise the FFT
    auto fft = FFT<e_device>(xsize, ysize);

    // Allocate buffers for the complex data, return data and ctf
    auto size = ysize * xsize;
    auto image_d = complex_vector_type(size);
    auto fft_d = complex_vector_type(size);
    auto ctf_d = complex_vector_type(size);
    auto rec_d = vector_type(size);

    // Copy image to device
    thrust::copy(image, image + size, image_d.begin());
      
    // Perform the forward FT
    fft.forward(image_d.data().get());

    // Loop through all the projections and all the CTFs
    for (auto j = 0; j < num_ctf; ++j) {
      // Copy the data into the device buffers
      thrust::copy(ctf + j * size, ctf + (j + 1) * size, ctf_d.begin());
      thrust::copy(image_d.begin(), image_d.end(), fft_d.begin());

      // Do the CTF correction
      thrust::transform(
        fft_d.begin(),
        fft_d.end(),
        ctf_d.begin(),
        fft_d.begin(),
        [] __device__(auto x, auto y) { return phase_flip(x, y); });

      // Perform the inverse FT
      fft.inverse(fft_d.data().get());

      // Get the real component
      thrust::transform(
        fft_d.begin(), fft_d.end(), rec_d.begin(), [] __device__(auto x) {
          return x.real();
        });

      // Copy the data to the output array
      thrust::copy(rec_d.begin(), rec_d.end(), rec + j * size);
    }
  }
}  // namespace detail

template class detail::Corrector<e_device, float>;

}  // namespace guanaco
