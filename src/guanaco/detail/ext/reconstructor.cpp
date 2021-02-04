
#include <guanaco/guanaco.h>

namespace guanaco {

Filter<e_host>::Filter(size_type num_pixels, size_type num_angles)
    : num_pixels_(num_pixels),
      num_angles_(num_angles),
      filter_(create_filter(num_pixels_ + 1)),
      fft_(num_pixels_ * 2, 1) {
  GUANACO_ASSERT(num_pixels_ > 0);
  GUANACO_ASSERT(num_angles_ > 0);
}

const std::vector<float> &Filter<e_host>::filter() const {
  return filter_;
}

std::vector<float> Filter<e_host>::create_filter(size_type size) const {
  GUANACO_ASSERT(size > 0);

  // Initialise the filter array
  auto filter = vector_type(size, 0);

  // Create a Ramp filter
  for (auto i = 0; i < size; i++) {
    float w = float(i) / float(size - 1);
    filter[i] = w;
  }

  // Return the filter
  return filter;
}

void Filter<e_host>::operator()(float *data) const {
  GUANACO_ASSERT(data != NULL);
  GUANACO_ASSERT(filter_.size() == num_pixels_ + 1);

  // Allocate a complex buffer for the row
  auto row = complex_vector_type(num_pixels_ * 2);

  // Loop through all the projection images
  for (auto i = 0; i < num_angles_; ++i) {
    // Get a row of data
    float *data_row = data + i * num_pixels_;

    // Copy the row into the zero padded array
    std::copy(data_row, data_row + num_pixels_, row.begin());
    std::fill(row.begin() + num_pixels_, row.end(), 0);

    // Take the FT of the row
    fft_.forward(reinterpret_cast<float *>(row.data()));

    // Apply the filter
    for (auto j = 0; j < filter_.size(); ++j) {
      row[j] *= filter_[j];
    }
    for (auto j = filter_.size(); j < row.size(); ++j) {
      row[j] *= filter_[row.size() - j];
    }

    // Take the inverse FT of the row
    fft_.inverse(reinterpret_cast<float *>(row.data()));

    // Copy the filtered data back
    for (auto j = 0; j < num_pixels_; ++j) {
      data_row[j] = row[j].real() / row.size();
    }
  }
}

Reconstructor_t<e_host>::Reconstructor_t(const Config &config)
    : config_(config), filter_(config_.num_pixels, config_.num_angles) {
  GUANACO_ASSERT(config_.device == e_host);
  GUANACO_ASSERT(config_.is_valid());
}

void Reconstructor_t<e_host>::operator()(float *sinogram, float *reconstruction) const {
  GUANACO_ASSERT(sinogram != nullptr);
  GUANACO_ASSERT(reconstruction != nullptr);

  // Get the sinogram and reconstruction sizes along with the number of
  // angles and the pixel area
  auto sino_size = config_.sino_size();
  auto grid_size = config_.grid_size();
  auto num_angles = config_.num_angles;
  auto pixel_area = config_.pixel_area();

  // Copy the sinogram before filtering
  std::vector<float> filtered_sinogram(sinogram, sinogram + sino_size);

  // Filter the sinogram
  filter_(filtered_sinogram.data());

  // Initialise the reconstruction to zero
  std::fill(reconstruction, reconstruction + grid_size, 0);

  // Perform the backprojection
  project(filtered_sinogram.data(), reconstruction);

  // Normalize the reconstruction
  for (auto i = 0; i < grid_size; ++i) {
    reconstruction[i] *= M_PI / (2 * num_angles * pixel_area);
  }
}

void Reconstructor_t<e_host>::project(float *sinogram, float *reconstruction) const {
  // precomputations
  const auto pixelLengthX = config_.pixel_size;
  const auto pixelLengthY = config_.pixel_size;
  const auto inv_pixelLengthX = 1.0f / pixelLengthX;
  const auto inv_pixelLengthY = 1.0f / pixelLengthY;
  const int grid_width = config_.grid_width;
  const int grid_height = config_.grid_height;
  const auto num_pixels = config_.num_pixels;
  const auto num_angles = config_.num_angles;
  const auto Ex = -float(config_.grid_width) / 2.0 + pixelLengthX * 0.5f;
  const auto Ey = float(config_.grid_height) / 2.0 - pixelLengthY * 0.5f;
  /* // Allocate host vectors */
  /* auto angle_scaled_sin = std::vector<float>(config_.num_angles); */
  /* auto angle_scaled_cos = std::vector<float>(config_.num_angles); */
  /* auto angle_offset = std::vector<float>(config_.num_angles); */
  /* auto angle_scale = std::vector<float>(config_.num_angles); */
  /* auto scale = M_PI / (2 * config_.num_angles * config_.pixel_area()); */
  /* scale *= config_.pixel_length_x();  // ONLY VALID FOR SQUARE */

  /* // Compute the quanities to store in the symbols */
  /* for (auto i = 0; i < config_.num_angles; ++i) { */
  /*   // Get the ray vector and length of the ray vector */
  /*   auto proj = config_.angles[i]; */
  /*   auto ray_x = proj.fRayX; */
  /*   auto ray_y = proj.fRayY; */
  /*   auto det_x0 = proj.fDetSX; */
  /*   auto det_y0 = proj.fDetSY; */
  /*   auto dir_x = proj.fDetUX; */
  /*   auto dir_y = proj.fDetUY; */
  /*   auto ray_length = std::sqrt(ray_y * ray_y + ray_x * ray_x); */
  /*   auto d = dir_x * ray_y - dir_y * ray_x; */

  /*   // Fill the arrays */
  /*   angle_scaled_cos[i] = ray_y / d; */
  /*   angle_scaled_sin[i] = -ray_x / d; */
  /*   angle_offset[i] = (det_y0 * ray_x - det_x0 * ray_y) / d; */
  /*   angle_scale[i] = ray_length / std::abs(d); */
  /* } */

  /* for (auto j = 0; j < grid_height; ++j) { */
  /*   std::cout << j << std::endl; */
  /*   for (auto i = 0; i < grid_width; ++i) { */

  /*     // Compute the x and y coordinates */
  /*     const float x = (i - 0.5 * grid_width + 0.5); */
  /*     const float y = (j - 0.5 * grid_height + 0.5); */
  /*     const std::size_t index = i + j * grid_width; */

  /*     // Loop through all the angles and compute the value of the voxel */
  /*     float fVal = 0.0f; */
  /*     for (size_t angle = 0; angle < num_angles; ++angle) { */
  /*       const float scaled_cos_theta = angle_scaled_cos[angle]; */
  /*       const float scaled_sin_theta = angle_scaled_sin[angle]; */
  /*       const float TOffset = angle_offset[angle]; */
  /*       const float scale = angle_scale[angle]; */

  /*       const float fT = x * scaled_cos_theta - y * scaled_sin_theta + TOffset; */
  /*       auto jj = (int)std::floor(angle+0.5); */
  /*       auto ii = (int)std::floor(fT); */
  /*       auto xx = angle+0.5-jj; */
  /*       auto yy = fT - ii; */
  /*       auto index00 = ii + jj*num_angles; */
  /*       auto index10 = ii+1 + jj*num_angles; */
  /*       auto index01 = ii + (jj+1)*num_angles; */
  /*       auto index11 = ii+1 + (jj+1)*num_angles; */
  /*       auto f00 = sinogram[index00]; */
  /*       auto f10 = sinogram[index10]; */
  /*       auto f01 = sinogram[index01]; */
  /*       auto f11 = sinogram[index11]; */
  /*       auto f = f00*(1-x)*(1-y) + f10*x*(1-y)+f01*(1-x)*y+f11*x*y; */
  /*       fVal += f * scale; */
  /*     } */

  /*     // Add the contribution to the voxel */
  /*     reconstruction[index] = fVal * scale; */

  /*   } */
  /* } */

  // Do the back projection of a pixel. Depending on the projection angle,
  // this function may be called with either rows or cols along x and y.
  //   - usize - The size of the grid along "u"
  //   - vsize - The size of the grid along "v"
  //   - ustride - The number of elements to the next "u" element
  //   - vstride - The number of elements to the next "v" element
  //   - v0 - The ray at u = 0
  //   - deltav - The change in "v" for a change in "u"
  //   - pixel - The pre-weighted pixel to project
  //   - reconstruction - The reconstruction grid
  auto project_internal = [](auto usize,
                             auto vsize,
                             auto ustride,
                             auto vstride,
                             auto v0,
                             auto deltav,
                             auto pixel,
                             auto reconstruction) {
    // Look along all u grid points
    bool inside = false;
    for (auto u = 0; u < usize; ++u) {
      // Compute the v index and offset
      auto vv = v0 + u * deltav;
      auto v = int(std::floor(vv));
      auto weight = vv - float(v);

      // If v is outside the boundary skip
      if (v < -1 || v >= vsize) {
        if (inside) {
          break;
        } else {
          continue;
        }
      }

      // The first point inside
      inside = true;

      // Add the contribution of the pixel to the adjacent grid points
      if (v >= 0) {
        auto index = v * vstride + u * ustride;
        reconstruction[index] += pixel * (1.0 - weight);
      }

      if (v + 1 < vsize) {
        auto index = (v + 1) * vstride + u * ustride;
        reconstruction[index] += pixel * weight;
      }
    }
  };

  // Loop through all the projections and back project each pixel in turn
  for (auto i = 0; i < num_angles; ++i) {
    // Get the rotation angles
    auto angle = config_.angles[i];
    auto row = sinogram + i * num_pixels;

    // Get the ray vector and length of the ray vector
    auto ray_x = -std::sin(angle);
    auto ray_y = std::cos(angle);
    auto det_x0 = -std::cos(angle) * 0.5 * num_pixels;
    auto det_y0 = -std::sin(angle) * 0.5 * num_pixels;
    auto dir_x = std::cos(angle);
    auto dir_y = std::sin(angle);
    auto ray_length = std::sqrt(ray_y * ray_y + ray_x * ray_x);

    // Check if the ray vector is pointing more along x or y
    const bool vertical = std::abs(ray_x) < std::abs(ray_y);

    for (auto j = 0; j < num_pixels; ++j) {
      // Compute the x and y position of the pixel in the volume and get the
      // pixel value
      auto x = det_x0 + (j + 0.5f) * dir_x;
      auto y = det_y0 + (j + 0.5f) * dir_y;
      auto pixel = row[j];

      int usize;
      int vsize;
      int ustride;
      int vstride;
      float v0;
      float deltav;
      float factor;
      if (vertical) {
        // "u" is along rows and "v" is along cols
        usize = grid_height;
        vsize = grid_width;
        ustride = grid_width;
        vstride = 1;

        // Compute the pixel factor
        factor = pixelLengthX * ray_length / std::abs(ray_y);

        // Compute the initial and delta "v"
        v0 = (x + (Ey - y) * (ray_x / ray_y) - Ex) * inv_pixelLengthX;
        deltav = -pixelLengthY * (ray_x / ray_y) * inv_pixelLengthX;

      } else {
        // "u" is along cols and "v" is along rows
        usize = grid_width;
        vsize = grid_height;
        ustride = 1;
        vstride = grid_width;

        // Compute the pixel factor
        factor = pixelLengthY * ray_length / std::abs(ray_x);

        // Compute the initial and delta "v"
        v0 = -(y + (Ex - x) * (ray_y / ray_x) - Ey) * inv_pixelLengthY;
        deltav = -pixelLengthX * (ray_y / ray_x) * inv_pixelLengthY;
      }

      // Project the pixel
      project_internal(
        usize, vsize, ustride, vstride, v0, deltav, pixel * factor, reconstruction);
    }
  }
}

}  // namespace guanaco
