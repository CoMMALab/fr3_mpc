#include "rt/controller.hpp"

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>

#include <utility>
#include <iostream>


namespace rt {

FR3Controller::FR3Controller(std::string robot_ip)
  : robot_ip_(std::move(robot_ip)) {
  running_.store(true, std::memory_order_relaxed);
  control_thread_ = std::thread(&FR3Controller::control_loop_, this);
}

FR3Controller::~FR3Controller() {
  stop();
}

bool FR3Controller::send_torque(const std::array<double, 7>& tau) noexcept {
  return torque_buffer_.try_write(tau);
}

void FR3Controller::stop() noexcept {
  const bool was_running = running_.exchange(false, std::memory_order_relaxed);
  if (!was_running) return;

  if (control_thread_.joinable()) {
    control_thread_.join();
  }
}

std::string FR3Controller::last_error() const {
  if (!has_error_.load(std::memory_order_relaxed)) return {};
  return error_message_;
}
void FR3Controller::control_loop_() {
  try {
    franka::Robot robot(robot_ip_);

    // Load dynamic model once
    franka::Model model = robot.loadModel();

    std::atomic<uint64_t> iter{0};

    robot.read([&](const franka::RobotState& state) {
      uint64_t k = iter.fetch_add(1, std::memory_order_relaxed);

      // Compute coriolis safely (no control)
      const std::array<double, 7> tau_coriolis = model.coriolis(state);

      // Throttled print ~1 Hz (read runs at ~1 kHz)
      if (k % 1000 == 0) {
        std::cout << "[READ ONLY] coriolis[0..2] = [ "
                  << tau_coriolis[0] << " "
                  << tau_coriolis[1] << " "
                  << tau_coriolis[2] << " ]"
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
