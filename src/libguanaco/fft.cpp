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

#include <fftw3.h>
#include <guanaco/fft.h>

namespace guanaco {

template <>
struct FFT<e_host>::Impl {
  fftwf_plan plan_forward;
  fftwf_plan plan_inverse;

  Impl() : plan_forward(nullptr), plan_inverse(nullptr) {}

  ~Impl() {
    destroy_plan();
  }

  void create_plan_1d(size_type size) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);
    auto data = (fftwf_complex *)fftw_malloc(sizeof(fftwf_complex) * size);
    plan_forward =
      fftwf_plan_dft_1d(size, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse =
      fftwf_plan_dft_1d(size, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_free(data);
  }

  void create_plan_2d(size_type xsize, size_type ysize) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);
    auto data =
      (fftwf_complex *)fftw_malloc(sizeof(fftwf_complex) * xsize * ysize);
    plan_forward =
      fftwf_plan_dft_2d(ysize, xsize, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse =
      fftwf_plan_dft_2d(ysize, xsize, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_free(data);
  }

  void forward(void *in, void *out) const {
    fftwf_execute_dft(plan_forward,
                      reinterpret_cast<fftwf_complex *>(in),
                      reinterpret_cast<fftwf_complex *>(out));
  }

  void inverse(void *in, void *out) const {
    fftwf_execute_dft(plan_inverse,
                      reinterpret_cast<fftwf_complex *>(in),
                      reinterpret_cast<fftwf_complex *>(out));
  }

  void destroy_plan() {
    if (plan_forward != nullptr) {
      fftwf_destroy_plan(plan_forward);
      plan_forward = nullptr;
    }
    if (plan_inverse != nullptr) {
      fftwf_destroy_plan(plan_inverse);
      plan_inverse = nullptr;
    }
    fftwf_cleanup_threads();
  }
};

template <>
FFT<e_host>::FFT() : pimpl(std::make_unique<Impl>()) {}

template <>
FFT<e_host>::FFT(const FFT &other)
    : pimpl(std::make_unique<Impl>(*other.pimpl)) {}

template <>
FFT<e_host>::FFT(FFT &&other) = default;

template <>
FFT<e_host> &FFT<e_host>::operator=(const FFT &other) {
  *pimpl = *other.pimpl;
  return *this;
}

template <>
FFT<e_host> &FFT<e_host>::operator=(FFT &&) = default;

template <>
FFT<e_host>::~FFT() = default;

template <>
FFT<e_host>::FFT(size_type size) : pimpl(std::make_unique<Impl>()) {
  pimpl->create_plan_1d(size);
}

template <>
FFT<e_host>::FFT(size_type xsize, size_type ysize)
    : pimpl(std::make_unique<Impl>()) {
  pimpl->create_plan_2d(xsize, ysize);
}

template <>
void FFT<e_host>::forward(void *in, void *out) const {
  pimpl->forward(in, out);
}

template <>
void FFT<e_host>::inverse(void *in, void *out) const {
  pimpl->inverse(in, out);
}

template <>
void FFT<e_host>::forward(void *data) const {
  forward(data, data);
}

template <>
void FFT<e_host>::inverse(void *data) const {
  inverse(data, data);
}

template <>
FFT<e_host> FFT<e_host>::make_1d(size_type size, size_type nbatch, bool real) {
  GUANACO_ASSERT(nbatch == 1);
  GUANACO_ASSERT(real == false);
  FFT<e_host> self;
  self.pimpl->create_plan_1d(size);
  return self;
}

template <>
FFT<e_host> FFT<e_host>::make_2d(size_type xsize,
                                 size_type ysize,
                                 size_type nbatch,
                                 bool real) {
  GUANACO_ASSERT(nbatch == 1);
  GUANACO_ASSERT(real == false);
  FFT<e_host> self;
  self.pimpl->create_plan_2d(xsize, ysize);
  return self;
}

}  // namespace guanaco
