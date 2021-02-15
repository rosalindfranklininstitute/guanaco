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
#ifndef GUANACO_CONSTANTS_H
#define GUANACO_CONSTANTS_H

#include <cmath>

namespace guanaco {

namespace constants {
  const double pi = M_PI;               // Pi
  const double c = 2.99792458e8;        // Speed of light (m/s)
  const double e = 1.602176634e-19;     // Elementary charge (C)
  const double m_e = 9.1093837015e-31;  // Electron mass (kg)
  const double h = 6.62607004e-34;      // Planck's constant (m^2 kg/s)
  const double m_to_A = 1e10;           // metres to Angstroms
  const double A_to_m = 1e-10;          // Angstroms to metres
}  // namespace constants

/**
 * Which device to execute on
 */
enum eDevice {
  e_host = 0,   // The CPU
  e_device = 1  // The GPU
};

}  // namespace guanaco

#endif
