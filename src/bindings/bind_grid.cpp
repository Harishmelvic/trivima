#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "trivima/cell_grid.h"
#include "trivima/serialization.h"

namespace nb = nanobind;

// Forward declaration
void bind_cell(nb::module_& m);

void bind_grid(nb::module_& m) {
    nb::class_<trivima::CellGridCPU>(m, "CellGrid")
        .def(nb::init<float>(), nb::arg("base_cell_size") = trivima::BASE_CELL_SIZE)
        .def_rw("base_cell_size", &trivima::CellGridCPU::base_cell_size)
        .def("insert", &trivima::CellGridCPU::insert)
        .def("find", &trivima::CellGridCPU::find)
        .def("contains", &trivima::CellGridCPU::contains)
        .def("remove", &trivima::CellGridCPU::remove)
        .def("size", &trivima::CellGridCPU::size)
        .def("empty", &trivima::CellGridCPU::empty)
        .def("clear", &trivima::CellGridCPU::clear)
        .def("reserve", &trivima::CellGridCPU::reserve)
        .def("geo", nb::overload_cast<uint32_t>(&trivima::CellGridCPU::geo),
             nb::rv_policy::reference_internal)
        .def("vis", nb::overload_cast<uint32_t>(&trivima::CellGridCPU::vis),
             nb::rv_policy::reference_internal)
        .def("key", &trivima::CellGridCPU::key,
             nb::rv_policy::reference_internal)
        .def("__len__", &trivima::CellGridCPU::size)
        .def("__repr__", [](const trivima::CellGridCPU& g) {
            return "<CellGrid cells=" + std::to_string(g.size()) +
                   " base_size=" + std::to_string(g.base_cell_size) + ">";
        });

    // Serialization functions
    m.def("save_grid", &trivima::save_grid, nb::arg("grid"), nb::arg("path"));
    m.def("load_grid", &trivima::load_grid, nb::arg("grid"), nb::arg("path"));
}

// Module entry point
NB_MODULE(trivima_native, m) {
    m.doc() = "Trivima native cell grid engine";
    bind_cell(m);
    bind_grid(m);
}
