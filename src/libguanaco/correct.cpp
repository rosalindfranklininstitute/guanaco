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
#include <future>
#include <iostream>
#include <vector>
#include <guanaco/error.h>
#include <guanaco/fft.h>
#include <guanaco/correct.h>

namespace guanaco {
namespace detail {
  template <typename T>
  void Corrector<e_host, T>::correct(const T *image,
                                     const std::complex<T> *ctf,
                                     T *rec,
                                     std::size_t xsize,
                                     std::size_t ysize,
                                     std::size_t num_ctf) {
    GUANACO_ASSERT(xsize > 0);
    GUANACO_ASSERT(ysize > 0);
    GUANACO_ASSERT(num_ctf > 0);

    using complex_vector_type = std::vector<std::complex<T>>;

    // Initialise the FFT
    auto fft = FFT<e_host>(xsize, ysize);

    // Allocate a complex buffer for the complex data
    auto buffer = complex_vector_type(num_ctf * xsize * ysize);
    auto buf = buffer.data();
     
    // Copy image to a buffer and perform the forward FT
    auto image_ft = complex_vector_type(xsize * ysize);
    auto img = image_ft.data();
    std::copy(image, image + xsize * ysize, img);
    fft.forward(img);

    // Loop through all the projections and do CTF correction
    std::vector<std::future<void>> result;
    for (auto j = 0; j < num_ctf; ++j) {
      result.push_back(
        std::async(
          std::launch::async, 
          [&](auto j) {

            // Get the CTF and output arrays
            auto c = ctf + j * ysize * xsize;
            auto r = rec + j * ysize * xsize;
            auto b = buf + j * ysize * xsize;

            // Copy the image into the complex zero padded array
            std::copy(img, img + ysize * xsize, b);

            // Do the CTF correction
            for (auto k = 0; k < ysize * xsize; ++k) {
              b[k] = phase_flip(b[k], c[k]);
            }

            // Perform the inverse FT
            fft.inverse(b);

            // Copy the data to the output array
            for (auto k = 0; k < ysize * xsize; ++k) {
              r[k] = b[k].real();
            }

          }, j));
    }

    // Wait for the results
    for(auto &r : result) { r.wait(); }
  }
}  // namespace detail

// Explicitly instantiate template functions
template class detail::Corrector<e_host, float>;

}  // namespace guanaco
