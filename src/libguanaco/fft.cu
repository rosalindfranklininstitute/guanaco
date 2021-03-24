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
#include <guanaco/fft.h>

namespace guanaco {

template <>
struct FFT<e_device>::Impl {
  cufftHandle plan_forward;
  cufftHandle plan_inverse;
  bool real;

  Impl() : plan_forward(0), plan_inverse(0), real(false) {}

  ~Impl() {
    destroy_plan();
  }

  void create_plan_1d(size_type size, size_type nbatch, bool real_) {
    real = real_;
    if (real) {
      cufftPlan1d(&plan_forward, size, CUFFT_R2C, nbatch);
      cufftPlan1d(&plan_inverse, size, CUFFT_C2R, nbatch);
    } else {
      cufftPlan1d(&plan_forward, size, CUFFT_C2C, nbatch);
      plan_inverse = plan_forward;
    }
  }

  void create_plan_2d(size_type xsize,
                      size_type ysize,
                      size_type nbatch,
                      bool real_) {
    real = real_;
    if (real) {
      cufftPlan2d(&plan_forward, ysize, xsize, CUFFT_R2C);
      cufftPlan2d(&plan_inverse, ysize, xsize, CUFFT_C2R);
    } else {
      cufftPlan2d(&plan_forward, ysize, xsize, CUFFT_C2C);
      plan_inverse = plan_forward;
    }
  }

  void forward(void *in, void *out) const {
    if (real) {
      auto i = reinterpret_cast<cufftReal *>(in);
      auto o = reinterpret_cast<cufftComplex *>(out);
      cufftExecR2C(plan_forward, i, o);
    } else {
      auto i = reinterpret_cast<cufftComplex *>(in);
      auto o = reinterpret_cast<cufftComplex *>(out);
      cufftExecC2C(plan_forward, i, o, CUFFT_FORWARD);
    }
  }

  void inverse(void *in, void *out) const {
    if (real) {
      auto i = reinterpret_cast<cufftComplex *>(in);
      auto o = reinterpret_cast<cufftReal *>(out);
      cufftExecC2R(plan_inverse, i, o);
    } else {
      auto i = reinterpret_cast<cufftComplex *>(in);
      auto o = reinterpret_cast<cufftComplex *>(out);
      cufftExecC2C(plan_inverse, i, o, CUFFT_INVERSE);
    }
  }

  void destroy_plan() {
    cudaDeviceSynchronize();
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    plan_forward = 0;
    plan_inverse = 0;
  }
};

template <>
FFT<e_device>::FFT() : pimpl(std::make_unique<Impl>()) {}

template <>
FFT<e_device>::FFT(const FFT &other)
    : pimpl(std::make_unique<Impl>(*other.pimpl)) {}

template <>
FFT<e_device>::FFT(FFT &&other) = default;

template <>
FFT<e_device> &FFT<e_device>::operator=(const FFT &other) {
  *pimpl = *other.pimpl;
  return *this;
}

template <>
FFT<e_device> &FFT<e_device>::operator=(FFT &&) = default;

template <>
FFT<e_device>::~FFT() = default;

template <>
FFT<e_device>::FFT(size_type size) : pimpl(std::make_unique<Impl>()) {
  pimpl->create_plan_1d(size, 1, false);
}

template <>
FFT<e_device>::FFT(size_type xsize, size_type ysize)
    : pimpl(std::make_unique<Impl>()) {
  pimpl->create_plan_2d(xsize, ysize, 1, false);
}

template <>
void FFT<e_device>::forward(void *in, void *out) const {
  pimpl->forward(in, out);
}

template <>
void FFT<e_device>::inverse(void *in, void *out) const {
  pimpl->inverse(in, out);
}

template <>
void FFT<e_device>::forward(void *data) const {
  forward(data, data);
}

template <>
void FFT<e_device>::inverse(void *data) const {
  inverse(data, data);
}

template <>
FFT<e_device> FFT<e_device>::make_1d(size_type size,
                                     size_type nbatch,
                                     bool real) {
  FFT<e_device> self;
  self.pimpl->create_plan_1d(size, nbatch, real);
  return self;
}

template <>
FFT<e_device> FFT<e_device>::make_2d(size_type xsize,
                                     size_type ysize,
                                     size_type nbatch,
                                     bool real) {
  FFT<e_device> self;
  self.pimpl->create_plan_2d(xsize, ysize, nbatch, real);
  return self;
}

}  // namespace guanaco
