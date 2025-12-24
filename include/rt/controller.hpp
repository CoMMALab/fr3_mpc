#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "rt/state.hpp"
#include "rt/torque_buffer.hpp"

namespace rt {

class FR3Robot {
public:
  static constexpr size_t kBufferSize = 1024;

  explicit FR3Robot(std::string robot_ip);
  ~FR3Robot();

  FR3Robot(const FR3Robot&) = delete;
  FR3Robot& operator=(const FR3Robot&) = delete;

  // Python calls this frequently:
  bool push(const std::array<double, 7>& tau) noexcept;

  // Read the most recent robot state
  State read() const noexcept;

  // Stop control loop and join thread
  void stop() noexcept;

  bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }
  std::string last_error() const; // non-RT diagnostic

private:
  void control_loop_();

  std::string robot_ip_;

  alignas(64) TorqueBuffer<kBufferSize> torque_buffer_{};
  alignas(64) std::atomic<bool> running_{false};

  // Most recent robot state (read from RT thread, written to from control loop)
  mutable std::mutex state_mutex_;
  State state_{};

  // Non-RT error reporting
  mutable std::atomic<bool> has_error_{false};
  std::string error_message_;

  std::thread control_thread_;
};

} // namespace rt
