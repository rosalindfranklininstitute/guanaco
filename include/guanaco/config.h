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
#ifndef GUANACO_CONFIG_H
#define GUANACO_CONFIG_H

#include <vector>
#include <guanaco/constants.h>

namespace guanaco {

struct Config {
  using size_type = std::size_t;

  eDevice device;
  int gpu_index;
  size_type num_pixels;
  size_type num_angles;
  size_type num_defocus;
  size_type grid_height;
  size_type grid_width;
  float pixel_size;
  float centre;
  std::vector<float> angles;
  float min_defocus;
  float max_defocus;

  Config()
      : device(e_host),
        gpu_index(0),
        num_pixels(0),
        num_angles(0),
        num_defocus(0),
        grid_height(0),
        grid_width(0),
        pixel_size(1),
        centre(0),
        min_defocus(0),
        max_defocus(0) {}

  size_type sino_size() const {
    return num_pixels * num_angles * num_defocus;
  }

  size_type grid_size() const {
    return grid_height * grid_width;
  }

  bool is_valid() const {
    return (device == e_host || device == e_device) 
      && num_pixels > 0                 // 
      && num_angles > 0                 //
      && num_defocus > 0                //
      && grid_height > 0                //
      && grid_width > 0                 //
      && pixel_size > 0                 //
      && min_defocus <= max_defocus     //
      && angles.size() == num_angles;   //
  }
};

}

#endif
