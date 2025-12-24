#pragma once
#include <array>

namespace rt {

struct State {
  double time{};                    // timestamp (seconds since epoch)
  std::array<double, 7> q{};        // joint positions
  std::array<double, 7> dq{};       // joint velocities
  std::array<double, 7> tau{};      // commanded torques
  std::array<double, 16> ee_pose{}; // 4x4 column-major
};

} // namespace rt
