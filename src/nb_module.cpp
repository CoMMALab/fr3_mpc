#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "rt/controller.hpp"
#include "rt/state.hpp"

namespace nb = nanobind;

NB_MODULE(control, m) {
  nb::class_<rt::FR3Robot>(m, "FR3Robot")
    .def(nb::init<std::string>(),
         nb::arg("robot_ip"))
    .def("push",
         [](rt::FR3Robot& self,
            const std::array<double, 7>& tau) {
           nb::gil_scoped_release release;
           return self.push(tau);
         })
    .def("read",
         [](const rt::FR3Robot& self) {
           nb::gil_scoped_release release;
           return self.read();
         })
    .def("stop",
         [](rt::FR3Robot& self) {
           nb::gil_scoped_release release;
           self.stop();
         })
    .def("is_running", &rt::FR3Robot::is_running)
    .def("last_error", &rt::FR3Robot::last_error);

  nb::class_<rt::State>(m, "State")
    .def_ro("time", &rt::State::time)
    .def_ro("q", &rt::State::q)
    .def_ro("dq", &rt::State::dq)
    .def_ro("tau", &rt::State::tau)
    .def_ro("ee_pose", &rt::State::ee_pose);
}
