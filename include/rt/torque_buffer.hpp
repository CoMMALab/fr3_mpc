#pragma once
#include <array>
#include <atomic>
#include <cstdint>

namespace rt {

template <size_t N>
struct TorqueBuffer {
  static_assert((N & (N - 1)) == 0, "N must be power of two");

  alignas(64) std::array<std::array<double, 7>, N> data{};
  alignas(64) std::atomic<uint64_t> head{0};
  alignas(64) std::atomic<uint64_t> tail{0};

  bool try_write(const std::array<double, 7>& v) noexcept {
    const uint64_t h = head.load(std::memory_order_relaxed);
    const uint64_t t = tail.load(std::memory_order_acquire);

    if (h - t < N) {
      data[h & (N - 1)] = v;
      head.store(h + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  bool try_read(std::array<double, 7>& v) noexcept {
    const uint64_t t = tail.load(std::memory_order_relaxed);
    const uint64_t h = head.load(std::memory_order_acquire);

    if (h > t) {
      v = data[t & (N - 1)];
      tail.store(t + 1, std::memory_order_release);
      return true;
    }
    return false;
  }
};

} // namespace rt
