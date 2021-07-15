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
  auto eV = e * V;
  return (h * c / std::sqrt(eV * eV + 2 * m_e * eV * c * c)) * m_to_A;
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
 * A struct to hold polar coordinates
 */
template <typename T>
struct Polar {
  T r;
  T theta;

  Polar() : r(0), theta(0) {}
  Polar(T r_, T theta_) : r(r_), theta(theta_) {}
};

/**
 * Get the spatial frequency squared
 * @param x: The x index
 * @param y: The y index
 * @param w: The width of the image
 * @param h: The height of the image
 * @param inv_ps: The inverse pixel size
 * @returns The polar coordinates
 */

template <typename T>
Polar<T> get_r_and_theta(std::size_t x,
                                  std::size_t y,
                                  std::size_t w,
                                  std::size_t h,
                                  T inv_ps) {
  auto w2 = w / 2;
  auto h2 = h / 2;
  auto dx = (T(((x + w2) % w)) - w2) / w;
  auto dy = (T(((y + h2) % h)) - h2) / h;
  auto r = std::sqrt((dx * dx + dy * dy) * inv_ps * inv_ps);
  auto theta = std::atan2(dy, dx);
  return Polar<T>(r, theta);
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
 * A struct to hold the CTF parameters
 */
struct CTF {
  double l;                 // Wavelength (A)
  double df;                // Defocus (A)
  double Cs;                // Spherical aberration (A)
  double Ca;                // 2-fold astigmatism (A)
  double Pa;                // Angle of 2-fold astigmatism (rad)
  double dd;                // Defocus spread (A)
  double theta_c;           // Source spread (rad)
  double phi;               // Phase shift (rad)

  CTF() : l(0), df(0), Cs(0), Ca(0), Pa(0), dd(0), theta_c(0), phi(0) {}
};

/**
 * Compute the spatial incoherence as equation 10.53 in De Graef
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The spatial incoherence envelope evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_log_Es(const CTF &c, T q, T theta) {
  using constants::pi;
  auto df = c.df + c.Ca * std::cos(2 * (theta - c.Pa));
  auto u = 1 + 2 * pi * pi * c.theta_c * c.theta_c * c.dd * c.dd * q * q;
  auto v = c.Cs * c.l * c.l * c.l * q * q * q - df * c.l * q;
  return -(pi * pi * c.theta_c * c.theta_c * v * v / (c.l * c.l * u));
}

/**
 * Compute the temporal incoherence envelope as equation 10.53 in De Graef
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @returns: The temporal incoherence envelope evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_log_Et(const CTF &c, T q) {
  using constants::pi;
  auto u = 1 + 2 * pi * pi * c.theta_c * c.theta_c * c.dd * c.dd * q * q;
  return -(pi * pi * c.l * c.l * c.dd * c.dd * q * q * q * q / (2 * u));
}

/**
 * Compute the envelope scale factor
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @returns: The scale factor at q
 */
template <typename T>
HOST_DEVICE constexpr T get_A(const CTF &c, T q) {
  using constants::pi;
  auto u = 1 + 2 * pi * pi * c.theta_c * c.theta_c * c.dd * c.dd * q * q;
  return 1 / sqrt(u);
}

/**
 * Compute chi as in Equation 10.9 in De Graef
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_chi(const CTF &c, T q, T theta) {
  using constants::pi;
  auto df = c.df + c.Ca * std::cos(2 * (theta - c.Pa));
  return pi * (c.Cs * c.l * c.l * c.l * q * q * q * q / 2 - c.l * df * q * q);
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The CTF evaluated at q
 */
template <typename T, typename U = std::complex<T>>
HOST_DEVICE constexpr U get_ctf(const CTF &c, T q, T theta) {
  auto chi = get_chi(c, q, theta);
  auto log_Et = get_log_Et(c, q);
  auto log_Es = get_log_Es(c, q, theta);
  auto A = get_A(c, q);
  return A * std::exp(U((log_Es + log_Et), -(chi - c.phi)));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The CTF evaluated at q
 */
template <typename T, typename U = std::complex<T>>
HOST_DEVICE constexpr U get_ctf_simple(const CTF &c, T q, T theta) {
  auto chi = get_chi(c, q, theta);
  return std::exp(U(0, -(chi - c.phi)));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_ctf_simple_real(const CTF &c, T q, T theta) {
  return std::cos(-(get_chi(c, q, theta) - c.phi));
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The spatial frequency to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @returns: The CTF evaluated at q
 */
template <typename T>
HOST_DEVICE constexpr T get_ctf_simple_imag(const CTF &c, T q, T theta) {
  return std::sin(-(get_chi(c, q, theta) - c.phi));
}

/**
 * Compute the spatial incoherence as equation 10.53 in De Graef
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angle
 * @param Es: The spatial incoherence envelope evaluated at q
 * @param n: The number of array elements
 */
template <typename T>
void get_Es_n(const CTF &c, const T *q, const T *theta, T *Es, std::size_t n) {
  std::transform(q, q + n, theta, Es, [c](auto q, auto theta) { 
      return std::exp(get_log_Es(c, q, theta)); 
  });
}

/**
 * Compute the temporal incoherence envelope as equation 10.53 in De Graef
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param Et: The temporal incoherence envelope evaluated at q
 * @param n: The number of array elements
 */
template <typename T>
void get_Et_n(const CTF &c, const T *q, T *Et, std::size_t n) {
  std::transform(q, q + n, Et, [c](auto q) { return std::exp(get_log_Et(c, q)); });
}

/**
 * Compute the envelope scale factor
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param A: The output scale factor
 * @param n: The number of array elements
 */
template <typename T>
void get_A_n(const CTF &c, const T *q, T *A, std::size_t n) {
  std::transform(q, q + n, A, [c](auto q) { return get_A(c, q); });
}

/**
 * Compute chi
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angles
 * @param chi: The output CTF array
 * @param n: The number of array elements
 */
template <typename T>
void get_chi_n(const CTF &c,
               const T *q,
               const T *theta,
               T *chi,
               std::size_t n) {
  std::transform(q, q + n, theta, chi, [c](auto q, auto theta) {
    return get_chi(c, q, theta);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angles
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n(const CTF &c,
               const T *q,
               const T *theta,
               U *ctf,
               std::size_t n) {
  std::transform(q, q + n, theta, ctf, [c](auto q, auto theta) {
    return get_ctf(c, q, theta);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angles
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n_simple(const CTF &c,
                      const T *q,
                      const T *theta,
                      U *ctf,
                      std::size_t n) {
  std::transform(q, q + n, theta, ctf, [c](auto q, auto theta) {
    return get_ctf_simple(c, q, theta);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angles
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 */
template <typename T>
void get_ctf_n_simple_real(const CTF &c,
                           const T *q,
                           const T *theta,
                           T *ctf,
                           std::size_t n) {
  std::transform(q, q + n, theta, ctf, [c](auto q, auto theta) {
    return get_ctf_simple_real(c, q, theta);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param q: The array of spatial frequencies to evaluate the CTF (1/A)
 * @param theta: The polar angles
 * @param ctf: The output CTF array
 * @param n: The number of array elements
 */
template <typename T>
void get_ctf_n_simple_imag(const CTF &c,
                           const T *q,
                           const T *theta,
                           T *ctf,
                           std::size_t n) {
  std::transform(q, q + n, theta, ctf, [c](auto q, auto theta) {
    return get_ctf_simple_imag(c, q, theta);
  });
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param ctf: The CTF
 * @param w: The width of the image
 * @param h: The height of the image
 * @param ps: The pixel size (A)
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n(const CTF &c, U *ctf, std::size_t w, std::size_t h, T ps) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto rt = get_r_and_theta(i, j, w, h, inv_ps);
      *ctf++ = get_ctf(c, rt.r, rt.theta);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param ctf: The CTF
 * @param w: The width of the image
 * @param h: The height of the image
 * @param ps: The pixel size (A)
 */
template <typename T, typename U = std::complex<T>>
void get_ctf_n_simple(const CTF &c,
                      U *ctf,
                      std::size_t w,
                      std::size_t h,
                      T ps) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto rt = get_r_and_theta(i, j, w, h, inv_ps);
      *ctf++ = get_ctf_simple(c, rt.r, rt.theta);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param ctf: The CTF
 * @param w: The width of the image
 * @param h: The height of the image
 * @param ps: The pixel size (A)
 */
template <typename T>
void get_ctf_n_simple_real(const CTF &c,
                           T *ctf,
                           std::size_t w,
                           std::size_t h,
                           T ps) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto rt = get_r_and_theta(i, j, w, h, inv_ps);
      *ctf++ = get_ctf_simple_real(c, rt.r, rt.theta);
    }
  }
}

/**
 * Compute the CTF
 *
 * The defocus is positive for underfocus
 *
 * @param c: The CTF parameters
 * @param ctf: The CTF
 * @param w: The width of the image
 * @param h: The height of the image
 * @param ps: The pixel size (A)
 */
template <typename T>
void get_ctf_n_simple_imag(const CTF &c,
                           T *ctf,
                           std::size_t w,
                           std::size_t h,
                           T ps) {
  GUANACO_ASSERT(ps > 0);
  T inv_ps = 1.0 / ps;
  for (auto j = 0; j < h; ++j) {
    for (auto i = 0; i < w; ++i) {
      auto rt = get_r_and_theta(i, j, w, h, inv_ps);
      *ctf++ = get_ctf_simple_imag(c, rt.r, rt.theta);
    }
  }
}

}  // namespace guanaco

#endif
