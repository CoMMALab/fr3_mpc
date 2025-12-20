#pragma once
#include <array>
#include <atomic>
#include <string>
#include <thread>

#include "rt/torque_buffer.hpp"

namespace rt {

class FR3Controller {
public:
  static constexpr size_t kBufferSize = 1024;

  explicit FR3Controller(std::string robot_ip);
  ~FR3Controller();

  FR3Controller(const FR3Controller&) = delete;
  FR3Controller& operator=(const FR3Controller&) = delete;

  // Python calls this frequently:
  bool send_torque(const std::array<double, 7>& tau) noexcept;

  // Stop control loop and join thread
  void stop() noexcept;

  bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }
  std::string last_error() const; // non-RT diagnostic

private:
  void control_loop_();

  std::string robot_ip_;

  alignas(64) TorqueBuffer<kBufferSize> torque_buffer_{};
  alignas(64) std::atomic<bool> running_{false};

  // Non-RT error reporting
  mutable std::atomic<bool> has_error_{false};
  std::string error_message_;

  std::thread control_thread_;
};

} // namespace rt
