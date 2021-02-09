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

    Impl()
      : plan_forward(0), plan_inverse(0) {}

    ~Impl() {
      destroy_plan();
    }

    void create_plan_1d_batch(size_type size, size_type nbatch) {
      cufftPlan1d(&plan_forward, size, CUFFT_R2C, nbatch);
      cufftPlan1d(&plan_inverse, size, CUFFT_C2R, nbatch);
    }
      
    void create_plan_2d(size_type xsize, size_type ysize) {
      cufftPlan2d(&plan_forward, ysize, xsize, CUFFT_R2C);
      cufftPlan2d(&plan_inverse, ysize, xsize, CUFFT_C2R);
    }
    
    void destroy_plan() {
      cudaDeviceSynchronize();
      if (plan_forward != 0) {
        cufftDestroy(plan_forward);
        plan_forward = 0;
      }
      if (plan_forward != 0) {
        cufftDestroy(plan_inverse);
        plan_inverse = 0;
      }
    }

  };
    
  template <>
  FFT<e_device>::FFT()
    : pimpl(std::make_unique<Impl>()) {}
  
  template <>
  FFT<e_device>::FFT(const FFT& other)
    : pimpl(std::make_unique<Impl>(*other.pimpl)) {}

  template <>
  FFT<e_device>::FFT(FFT&& other) = default;

  template <>
  FFT<e_device>& FFT<e_device>::operator=(const FFT &other) {
    *pimpl = *other.pimpl;
    return *this;
  }

  template <>
  FFT<e_device>& FFT<e_device>::operator=(FFT&&) = default;

  template <>
  FFT<e_device>::~FFT() = default;

  template <>
  FFT<e_device>::FFT(size_type size)
    : pimpl(std::make_unique<Impl>()) {
      pimpl->create_plan_1d_batch(size, 1);
  }
    
  template <>
  FFT<e_device>::FFT(size_type xsize, size_type ysize)
  : pimpl(std::make_unique<Impl>()) {
    pimpl->create_plan_2d(xsize, ysize);
  }

  template <>
  void FFT<e_device>::forward(void *in, void *out) const {
    auto i = reinterpret_cast<cufftReal *>(in);
    auto o = reinterpret_cast<cufftComplex *>(out);
    cufftExecR2C(pimpl->plan_forward, i, o);
  }

  template <>
  void FFT<e_device>::inverse(void *in, void *out) const {
    auto i = reinterpret_cast<cufftComplex *>(in);
    auto o = reinterpret_cast<cufftReal *>(out);
    cufftExecC2R(pimpl->plan_inverse, i, o);
  }

  template <>
  FFT<e_device> FFT<e_device>::make_1d(size_type size) {
    FFT<e_device> self;
    self.pimpl->create_plan_1d_batch(size, 1);
    return self;
  }
  
  template <>
  FFT<e_device> FFT<e_device>::make_1d_batch(size_type size, size_type nbatch) {
    FFT<e_device> self;
    self.pimpl->create_plan_1d_batch(size, nbatch);
    return self;
  }
    
  template <>
  FFT<e_device> FFT<e_device>::make_2d(size_type xsize, size_type ysize) {
    FFT<e_device> self;
    self.pimpl->create_plan_2d(xsize, ysize);
    return self;
  }

}

