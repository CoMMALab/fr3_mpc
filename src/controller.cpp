#include "rt/controller.hpp"

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>

#include <utility>
#include <iostream>


namespace rt {

FR3Robot::FR3Robot(std::string robot_ip)
  : robot_ip_(std::move(robot_ip)) {
  running_.store(true, std::memory_order_relaxed);
  control_thread_ = std::thread(&FR3Robot::control_loop_, this);
}

FR3Robot::~FR3Robot() {
  stop();
}

bool FR3Robot::push(const std::array<double, 7>& tau) noexcept {
  return torque_buffer_.try_write(tau);
}

State FR3Robot::read() const noexcept {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

void FR3Robot::stop() noexcept {
  const bool was_running = running_.exchange(false, std::memory_order_relaxed);
  if (!was_running) return;

  if (control_thread_.joinable()) {
    control_thread_.join();
  }
}

std::string FR3Robot::last_error() const {
  if (!has_error_.load(std::memory_order_relaxed)) return {};
  return error_message_;
}

void FR3Robot::control_loop_() {
  try {
    franka::Robot robot(robot_ip_);

    // Load dynamic model once
    franka::Model model = robot.loadModel();

    std::atomic<uint64_t> iter{0};

    robot.read([&](const franka::RobotState& robot_state) {
      uint64_t k = iter.fetch_add(1, std::memory_order_relaxed);

      // Read latest commanded torque from buffer (if any)
      std::array<double, 7> tau_cmd{};
      bool has_tau = torque_buffer_.try_read(tau_cmd);

      // Update our state buffer
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.time = robot_state.time.toSec();
        std::copy(robot_state.q.begin(), robot_state.q.end(), state_.q.begin());
        std::copy(robot_state.dq.begin(), robot_state.dq.end(), state_.dq.begin());
        if (has_tau) {
          state_.tau = tau_cmd;
        }
        std::copy(robot_state.O_T_EE.begin(), robot_state.O_T_EE.end(), state_.ee_pose.begin());
      }

      // Compute coriolis safely (no control)
      const std::array<double, 7> tau_coriolis = model.coriolis(robot_state);

      // Throttled print ~1 Hz (read runs at ~1 kHz)
      if (k % 1000 == 0) {
        std::cout << "[READ ONLY] q[0..2] = [ "
                  << robot_state.q[0] << " "
                  << robot_state.q[1] << " "
                  << robot_state.q[2] << " ]"
                  << std::endl;
      }

      // Exit condition
      if (!running_.load(std::memory_order_relaxed)) {
        return false;  // stops robot.read()
      }

      return true; // continue reading
    });

    std::cout << "[READ ONLY] Exited read loop cleanly\n";

  } catch (const franka::Exception& e) {
    error_message_ = e.what();
    has_error_.store(true, std::memory_order_relaxed);
  }
}

} // namespace rt
