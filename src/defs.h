#pragma once

#include <vector>
#include <cstdint>

struct PointSet {
  std::vector<float> coordinates;
  size_t d, n;
  float* GetPoint(size_t i) { return &coordinates[0] + i*d; }
};

using AdjGraph = std::vector<std::vector<int>>;

using NNVec = std::vector<std::pair<float, uint32_t>>;
