#include <iostream>
#include <guanaco/guanaco.h>

namespace guanaco {

template <>
Reconstructor_t<e_host>::Reconstructor_t(const Config &config)
    : config_(config),
      filter_(config_.num_pixels, config_.num_angles, config_.num_defocus) {
  GUANACO_ASSERT(config_.device == e_host);
  GUANACO_ASSERT(config_.is_valid());
}

template <>
void Reconstructor_t<e_host>::project(const float *sinogram,
                                      float *reconstruction) const {
  auto grid_width = config_.grid_width;
  auto grid_height = config_.grid_height;
  auto num_pixels = config_.num_pixels;
  auto num_angles = config_.num_angles;
  auto num_defocus = config_.num_defocus;
  auto min_defocus = config_.min_defocus;
  auto max_defocus = config_.max_defocus;
  auto pixel_size = config_.pixel_size;
  auto centre = config_.centre;
  auto output_scale = M_PI / (2 * config_.num_angles);

  // Compute the defocus scale and offset
  auto dscale = num_defocus > 1
                  ? num_defocus * pixel_size / (max_defocus - min_defocus)
                  : 0;
  auto doffset = -dscale * (min_defocus / pixel_size);

  // Loop through projections
  for (auto angle = 0; angle < config_.num_angles; ++angle) {
    auto theta = config_.angles[angle];
    auto cos_angle = std::cos(theta);
    auto sin_angle = std::sin(theta);
    auto det_x0 = -centre * cos_angle;
    auto det_y0 = -centre * sin_angle;
    auto ray_length = std::sqrt(cos_angle * cos_angle + sin_angle * sin_angle);
    auto d = cos_angle * cos_angle + sin_angle * sin_angle;

    float offset = (det_y0 * (-sin_angle) - det_x0 * cos_angle) / d;
    float scale = output_scale * ray_length / std::abs(d);
    cos_angle = cos_angle / d;
    sin_angle = sin_angle / d;

    // Get the row (ignoring defocus)
    auto row_angle = sinogram + angle * num_pixels;

    // Loop through all grid pixels
    for (int j = 0; j < grid_height; ++j) {
      for (int i = 0; i < grid_width; ++i) {
        auto row = row_angle;

        // Compute the x and y coordinates
        float x = (i - 0.5 * grid_width + 0.5);
        float y = (j - 0.5 * grid_height + 0.5);
        int index = i + j * grid_width;

        // Compute the pixel and height
        float pixel = x * cos_angle - y * sin_angle + offset;

        // Add the defocus offset
        if (num_defocus > 1) {
          float height = -x * sin_angle - y * cos_angle;
          float defocus = height * dscale + doffset;
          int defocus_index = (int)std::floor(defocus + 0.5);
          if (defocus_index < 0) defocus_index = 0;
          if (defocus_index > num_defocus - 1) defocus_index = num_defocus - 1;
          row += defocus_index * num_pixels * num_angles;
        }

        // Interpolate the pixel
        pixel -= 0.5;
        int ind = (int)std::floor(pixel);
        float t = pixel - ind;
        if (ind >= 0 && ind < num_pixels - 1) {
          int i0 = ind;
          int i1 = i0 + 1;
          float v0 = row[i0];
          float v1 = row[i1];
          float value = (1 - t) * v0 + t * v1;
          reconstruction[index] += scale * value;
        }
      }
    }
  }
}

template <>
void Reconstructor_t<e_host>::operator()(const float *sinogram,
                                         float *reconstruction) const {
  GUANACO_ASSERT(sinogram != nullptr);
  GUANACO_ASSERT(reconstruction != nullptr);

  // Get the sinogram and reconstruction sizes along with the number of
  // angles and the pixel area
  auto sino_size = config_.sino_size();
  auto grid_size = config_.grid_size();
  auto num_angles = config_.num_angles;

  // Copy the sinogram before filtering
  std::vector<float> filtered_sinogram(sinogram, sinogram + sino_size);

  // Filter the sinogram
  filter_(filtered_sinogram.data());

  // Initialise the reconstruction to zero
  std::fill(reconstruction, reconstruction + grid_size, 0);

  // Perform the backprojection
  project(filtered_sinogram.data(), reconstruction);
}

Reconstructor::Reconstructor(const Config &config) : config_(config) {}

void Reconstructor::operator()(const float *sinogram,
                               float *reconstruction) const {
  switch (config_.device) {
  case e_device: {
    auto alg = Reconstructor_t<e_device>(config_);
    alg(sinogram, reconstruction);
  } break;
  case e_host:
  default: {
    auto alg = Reconstructor_t<e_host>(config_);
    alg(sinogram, reconstruction);
  } break;
  };
}

}  // namespace guanaco
