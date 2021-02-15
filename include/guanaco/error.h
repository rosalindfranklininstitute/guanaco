/*
 * This file is part of guanaco-ctf.
 * Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
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
#ifndef GUANACO_ERROR_H
#define GUANACO_ERROR_H

#include <stdexcept>
#include <string>

namespace guanaco {

/**
 * An error class that also prints the file and line number
 */
class Error : public std::runtime_error {
public:
  Error(const std::string& what_arg) : std::runtime_error(what_arg) {}

  Error(const char* what_arg) : std::runtime_error(what_arg) {}

  Error(const std::string& file, std::size_t line, const std::string& message)
      : std::runtime_error(file + ":" + std::to_string(line) + " " + message) {}
};

}  // namespace guanaco

/**
 * Throw an error if the assertion fails
 */
#define GUANACO_ASSERT(assertion)                            \
  if (!(assertion)) {                                        \
    throw guanaco::Error(                                    \
      __FILE__, __LINE__, "ASSERT (" #assertion ") failed"); \
  }

/**
 * Throw an error if the assertion fails
 */
#ifdef __CUDACC__
#define GUANACO_ASSERT_CUDA(assertion)                                   \
  if (!assertion) {                                                      \
    auto error = cudaGetLastError();                                     \
    throw guanaco::Error(__FILE__, __LINE__, cudaGetErrorString(error)); \
  }
#endif

#endif
