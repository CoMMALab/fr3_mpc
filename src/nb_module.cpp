#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "rt/controller.hpp"

namespace nb = nanobind;

NB_MODULE(control, m) {
  nb::class_<rt::FR3Controller>(m, "FR3Controller")
    .def(nb::init<std::string>(),
         nb::arg("robot_ip"))
    .def("send_torque",
         [](rt::FR3Controller& self,
            const std::array<double, 7>& tau) {
           nb::gil_scoped_release release;
           return self.send_torque(tau);
         })
    .def("stop",
         [](rt::FR3Controller& self) {
           nb::gil_scoped_release release;
           self.stop();
         })
    .def("is_running", &rt::FR3Controller::is_running)
    .def("last_error", &rt::FR3Controller::last_error);
}
