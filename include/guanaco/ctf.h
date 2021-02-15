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
#ifndef GUANACO_CTF_H
#define GUANACO_CTF_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <tuple>
#include <guanaco/constants.h>
#include <guanaco/error.h>
#include <guanaco/system.h>

namespace guanaco {

/**
 * Compute the electron wavelength
 * @param v: energy in electron volts
 * @returns: Wavelength (A)
 */
template <typename T>
constexpr T get_electron_wavelength(T V) {
  using constants::c;
  using constants::e;
  using constants::h;
  using constants::m_e;
  using constants::m_to_A;
  return (h * c / std::sqrt((e * V) * (e * V) + 2 * m_e * e * V * c * c))
         * m_to_A;
}

/**
 * From equation 3.41 in Kirkland: Advanced Computing in Electron Microscopy
 *
 * The dE, dI, dV are the 1/e half widths or E, I and V respectively
 *
 * @param Cc: The chromatic aberration (A)
 * @param dEE: dE/E, the fluctuation in the electron energy
 * @param dII: dI/I, the fluctuation in the lens current
 * @param dVV: dV/V, the fluctuation in the acceleration voltage
 * @returns: The defocus spread
 */
template <typename T>
constexpr T get_defocus_spread(T Cc, T dEE, T dII, T dVV) {
  return Cc * std::sqrt((dEE) * (dEE) + (2 * dII) * (2 * dII) + (dVV) * (dVV));
}

/**
 * Get the spatial frequency squared
 * @param x: The x index
 * @param y: The y index
 * @param w: The width of the image
 * @param h: The height of the image
 * @param inv_ps: The inverse pixel size
 * @returns: Tuple of q^2 and angle
 */

template <typename T>
constexpr std::tuple<T, T> get_q2(std::size_t x,
                                  std::size_t y,
                                  std::size_t w,
                                  std::size_t h,
                                  T inv_ps) {
  auto w2 = w / 2;
  auto h2 = h / 2;
  auto dx = (T(((x + w2) % w)) - w2) / w;
  auto dy = (T(((y + h2) % h)) - h2) / h;
  return std::make_tuple((dx * dx + dy * dy) * inv_ps * inv_ps,
                         std::atan2(dy, dx));
}

/**
 * Get the spatial frequency
 * @param x: The x index
 * @param y: The y index
 * @param w: The width of the image
 * @param h: The height of the image
 * @param inv_ps: The inverse pixel size
 * @returns: Tuple of q and angle
 */
template <typename T>
constexpr std::tuple<T, T> get_q(std::size_t x,
                                 std::size_t y,
                                 std::size_t w,
                                 std::size_t h,
                                 T inv_ps) {
  auto t = get_q2(x, y, w, h, inv_ps);
  return std::make_tuple(std::sqrt(std::get<0>(t)), std::get<1>(t));
}

/**
 * Convert spatial frequencies to dimensionless quantities
 *
 * From Equation 10.10 in DeGraef
 *
 * @param q: The spatial frequency (1/A)
 * @param l: The electron wavelength (A)
 * @param Cs: The spherical aberration (A)
 * @returns: The dimensionless spatial frequencies
 */
template <typename T>
constexpr T q_to_Q(T q, T l, T Cs) {
  return q * std::pow(Cs * l * l * l, 1.0 / 4.0);
}

/**
 * Get spatial frequencies from dimensionless quantities
 *
 * From Equation 10.10 in DeGraef
 *
 * @param Q: The dimensionless spatial frequency
 * @param l: The electron wavelength (A)
 * @param Cs: The spherical aberration (A)
 * @returns: The spatial frequencies (1/A)
 */
template <typename T>
constexpr T Q_to_q(T Q, T l, T Cs) {
  return Q / std::pow(Cs * l * l * l, 1.0 / 4.0);
}

/**
 * Convert defocus to dimensionless quantities
 *
 * From Equation 10.11 in DeGraef
 *
 * @param df: The defocus (A)
 * @param l: The electron wavelength (A)
 * @param Cs: The spherical aberration (A)
 * @returns: The dimensionless spatial frequencies
 */
template <typename T>
constexpr T df_to_D(T df, T l, T Cs) {
  return df / std::sqrt(Cs * l);
}

/**
 * Get defocus from dimensionless quantities
 *
 * From Equation 10.11 in DeGraef
 *
 * @param D: The dimensionless defocus
 * @param l: The electron wavelength (A)
 * @param Cs: The spherical aberration (A)
 * @returns: The defocus (A)
 */
template <typename T>
constexpr T D_to_df(T D, T l, T Cs) {
  return D * sqrt(Cs * l);
}

/**
 * Compute the spatial incoherence as equation 10.53 in De Graef
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @returns: The spatial incoherence envelope evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_Es(T q, T l, T df, T Cs, T dd, T theta_c) {
  using constants::pi;
  auto u = 1 + 2 * pi * pi * theta_c * theta_c * dd * dd * q * q;
  auto v = Cs * l * l * l * q * q * q + df * l * q;
  return -(pi * pi * theta_c * theta_c * v * v / (l * l * u));
}

/**
 * Compute the temporal incoherence envelope as equation 10.53 in De Graef
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @returns: The temporal incoherence envelope evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_Et(T q, T l, T dd, T theta_c) {
  using constants::pi;
  auto u = 1 + 2 * pi * pi * theta_c * theta_c * dd * dd * q * q;
  return -(pi * pi * l * l * dd * dd * q * q * q * q / (2 * u));
}

/**
 * Compute the envelope scale factor
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @returns: The scale factor at q
 */
template <typename T>
HOST_DEVICE constexpr T get_A(T q, T dd, T theta_c) {
  using constants::pi;
  auto u = 1 + 2 * pi * pi * theta_c * theta_c * dd * dd * q * q;
  return 1 / sqrt(u);
}

/**
 * Compute chi as in Equation 10.9 in De Graef
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_chi(T q, T l, T df, T Cs, T Ca, T Pa) {
  using constants::pi;
  df += Ca * std::cos(2 * Pa);
  return pi * (Cs * l * l * l * q * q * q * q / 2 - l * df * q * q);
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @param phi: The phase shift
 * @returns: The CTF evaluated at q
 */
template <typename T, typename U = std::complex<T>>
HOST_DEVICE constexpr U
get_ctf(T q, T l, T df, T Cs, T Ca, T Pa, T dd, T theta_c, T phi = 0) {
  auto chi = get_chi(q, l, df, Cs, Ca, Pa);
  auto Et = get_Et(q, l, dd, theta_c);
  auto Es = get_Es(q, l, df, Cs, dd, theta_c);
  auto A = get_A(q, dd, theta_c);
  return A * std::exp(U(Es + Et, -(chi - phi)));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 * @returns: The CTF evaluated at q
 */
template <typename T, typename U = std::complex<T>>
HOST_DEVICE constexpr U
get_ctf_simple(T q, T l, T df, T Cs, T Ca, T Pa, T phi = 0) {
  auto chi = get_chi(q, l, df, Cs, Ca, Pa);
  return std::exp(U(0, -(chi - phi)));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T
get_ctf_simple_real(T q, T l, T df, T Cs, T Ca, T Pa, T phi = 0) {
  return std::cos(-(get_chi(q, l, df, Cs, Ca, Pa) - phi));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T
get_ctf_simple_imag(T q, T l, T df, T Cs, T Ca, T Pa, T phi = 0) {
  return std::sin(-(get_chi(q, l, df, Cs, Ca, Pa) - phi));
}

/**
 * Compute the spatial incoherence as equation 10.53 in De Graef
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param Es: The spatial incoherence envelope evaluated at q
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 */
template <typename T>
void get_Es_n(const T *q,
              T *Es,
              std::size_t n,
              T l,
              T df,
              T Cs,
              T dd,
              T theta_c) {
  std::transform(q, q + n, Es, [l, df, Cs, dd, theta_c](auto q) {
    return get_Es(q, l, df, Cs, dd, theta_c);
  });
}

/**
 * Compute the temporal incoherence envelope as equation 10.53 in De Graef
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param Et: The temporal incoherence envelope evaluated at q
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 */
template <typename T>
void get_Et_n(const T *q, T *Et, std::size_t n, T l, T dd, T theta_c) {
  std::transform(q, q + n, Et, [l, dd, theta_c](auto q) {
    return get_Et(q, l, dd, theta_c);
  });
}

/**
 * Compute the envelope scale factor
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param A: The output scale factor
 * @param n: The number of array elements
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 */
template <typename T>
void get_A_n(const T *q, T *A, std::size_t n, T dd, T theta_c) {
  std::transform(
    q, q + n, A, [dd, theta_c](auto q) { return get_A(q, dd, theta_c); });
}

/**
 * Compute chi
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param chi: The output CTF array
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 */
template <typename T>
void get_chi_n(const T *q, T *chi, std::size_t n, T l, T df, T Cs, T Ca, T Pa) {
  std::transform(q, q + n, chi, [l, df, Cs, Ca, Pa](auto q) {
    return get_chi(q, l, df, Cs, Ca, Pa);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @param phi: The phase shift
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n(const T *q,
               U *ctf,
               std::size_t n,
               T l,
               T df,
               T Cs,
               T Ca,
               T Pa,
               T dd,
               T theta_c,
               T phi = 0) {
  std::transform(q, q + n, ctf, [l, df, Cs, Ca, Pa, dd, theta_c, phi](auto q) {
    return get_ctf(q, l, df, Cs, Ca, Pa, dd, theta_c, phi);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n_simple(const T *q,
                      U *ctf,
                      std::size_t n,
                      T l,
                      T df,
                      T Cs,
                      T Ca,
                      T Pa,
                      T phi = 0) {
  std::transform(q, q + n, ctf, [l, df, Cs, Ca, Pa, phi](auto q) {
    return get_ctf_simple(q, l, df, Cs, Ca, Pa, phi);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T>
void get_ctf_n_simple_real(const T *q,
                           T *ctf,
                           std::size_t n,
                           T l,
                           T df,
                           T Cs,
                           T Ca,
                           T Pa,
                           T phi = 0) {
  std::transform(q, q + n, ctf, [l, df, Cs, Ca, Pa, phi](auto q) {
    return get_ctf_simple_real(q, l, df, Cs, Ca, Pa, phi);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T>
void get_ctf_n_simple_imag(const T *q,
                           T *ctf,
                           std::size_t n,
                           T l,
                           T df,
                           T Cs,
                           T Ca,
                           T Pa,
                           T phi = 0) {
  std::transform(q, q + n, ctf, [l, df, Cs, Ca, Pa, phi](auto q) {
    return get_ctf_simple_imag(q, l, df, Cs, Ca, Pa, phi);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param ctf: The output CTF array
 * @param w: The width of the image
 * @param h: The height of the image
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param dd: The defocus spread (A)
 * @param theta_c: The source spread (rad)
 * @param phi: The phase shift
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n(U *ctf,
               std::size_t w,
               std::size_t h,
               T ps,
               T l,
               T df,
               T Cs,
               T Ca,
               T Pa,
               T dd,
               T theta_c,
               T phi = 0) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto q = get_q(i, j, w, h, inv_ps);
      auto a = std::get<0>(q) - Pa;
      *ctf++ = get_ctf(std::get<0>(q), l, df, Cs, Ca, a, dd, theta_c, phi);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param ctf: The output CTF array
 * @param w: The width of the image
 * @param h: The height of the image
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n_simple(U *ctf,
                      std::size_t w,
                      std::size_t h,
                      T ps,
                      T l,
                      T df,
                      T Cs,
                      T Ca,
                      T Pa,
                      T phi = 0) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto q = get_q(i, j, w, h, inv_ps);
      auto a = std::get<0>(q) - Pa;
      *ctf++ = get_ctf_simple(std::get<0>(q), l, df, Cs, Ca, a, phi);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param ctf: The output CTF array
 * @param w: The width of the image
 * @param h: The height of the image
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T>
void get_ctf_n_simple_real(T *ctf,
                           std::size_t w,
                           std::size_t h,
                           T ps,
                           T l,
                           T df,
                           T Cs,
                           T Ca,
                           T Pa,
                           T phi = 0) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto q = get_q(i, j, w, h, inv_ps);
      auto a = std::get<0>(q) - Pa;
      *ctf++ = get_ctf_simple_real(std::get<0>(q), l, df, Cs, Ca, a, phi);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param ctf: The output CTF array
 * @param w: The width of the image
 * @param h: The height of the image
 * @param l: The electron wavelength (A)
 * @param df: The defocus (A)
 * @param Cs: The spherical aberration (A)
 * @param Ca: The 2-fold astigmatism (A)
 * @param Pa: The astigmatism angle (rad)
 * @param phi: The phase shift
 */
template <typename T>
void get_ctf_n_simple_imag(T *ctf,
                           std::size_t w,
                           std::size_t h,
                           T ps,
                           T l,
                           T df,
                           T Cs,
                           T Ca,
                           T Pa,
                           T phi = 0) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto q = get_q(i, j, w, h, inv_ps);
      auto a = std::get<0>(q) - Pa;
      *ctf++ = get_ctf_simple_imag(std::get<0>(q), l, df, Cs, Ca, a, phi);
    }
  }
}

}  // namespace guanaco

#endif
