#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "trivima/cell.h"

namespace nb = nanobind;

void bind_cell(nb::module_& m) {
    // CellType enum
    nb::enum_<trivima::CellType>(m, "CellType")
        .value("Empty", trivima::CellType::Empty)
        .value("Surface", trivima::CellType::Surface)
        .value("Solid", trivima::CellType::Solid)
        .value("Transparent", trivima::CellType::Transparent);

    // CellGeo — exposed as read/write properties
    nb::class_<trivima::CellGeo>(m, "CellGeo")
        .def(nb::init<>())
        .def_rw("density", &trivima::CellGeo::density)
        .def_rw("density_gx", &trivima::CellGeo::density_gx)
        .def_rw("density_gy", &trivima::CellGeo::density_gy)
        .def_rw("density_gz", &trivima::CellGeo::density_gz)
        .def_rw("density_integral", &trivima::CellGeo::density_integral)
        .def_rw("normal_x", &trivima::CellGeo::normal_x)
        .def_rw("normal_y", &trivima::CellGeo::normal_y)
        .def_rw("normal_z", &trivima::CellGeo::normal_z)
        .def_rw("normal_gx", &trivima::CellGeo::normal_gx)
        .def_rw("normal_gy", &trivima::CellGeo::normal_gy)
        .def_rw("normal_gz", &trivima::CellGeo::normal_gz)
        .def_prop_rw("cell_type",
            [](const trivima::CellGeo& g) { return g.type(); },
            [](trivima::CellGeo& g, trivima::CellType t) { g.set_type(t); })
        .def_rw("confidence", &trivima::CellGeo::confidence)
        .def_rw("collision_margin", &trivima::CellGeo::collision_margin)
        .def("is_solid", &trivima::CellGeo::is_solid)
        .def("is_empty", &trivima::CellGeo::is_empty)
        .def("is_low_confidence", &trivima::CellGeo::is_low_confidence)
        .def("__repr__", [](const trivima::CellGeo& g) {
            return "<CellGeo density=" + std::to_string(g.density) +
                   " type=" + std::to_string(g.cell_type_raw) +
                   " conf=" + std::to_string(g.confidence) + ">";
        });

    // CellVisual — key properties exposed
    nb::class_<trivima::CellVisual>(m, "CellVisual")
        .def(nb::init<>())
        .def_rw("albedo_r", &trivima::CellVisual::albedo_r)
        .def_rw("albedo_g", &trivima::CellVisual::albedo_g)
        .def_rw("albedo_b", &trivima::CellVisual::albedo_b)
        .def_rw("light_r", &trivima::CellVisual::light_r)
        .def_rw("light_g", &trivima::CellVisual::light_g)
        .def_rw("light_b", &trivima::CellVisual::light_b)
        .def_rw("light_a", &trivima::CellVisual::light_a)
        .def_rw("semantic_label", &trivima::CellVisual::semantic_label)
        .def_rw("roughness", &trivima::CellVisual::roughness)
        .def_rw("reflectance", &trivima::CellVisual::reflectance)
        .def("__repr__", [](const trivima::CellVisual& v) {
            return "<CellVisual albedo=(" +
                   std::to_string(v.albedo_r) + "," +
                   std::to_string(v.albedo_g) + "," +
                   std::to_string(v.albedo_b) + ") label=" +
                   std::to_string(v.semantic_label) + ">";
        });

    // CellKey
    nb::class_<trivima::CellKey>(m, "CellKey")
        .def(nb::init<>())
        .def_rw("level", &trivima::CellKey::level)
        .def_rw("x", &trivima::CellKey::x)
        .def_rw("y", &trivima::CellKey::y)
        .def_rw("z", &trivima::CellKey::z)
        .def("pack", &trivima::CellKey::pack)
        .def_static("unpack", &trivima::CellKey::unpack)
        .def("child", &trivima::CellKey::child)
        .def("parent", &trivima::CellKey::parent)
        .def("__repr__", [](const trivima::CellKey& k) {
            return "<CellKey level=" + std::to_string(k.level) +
                   " (" + std::to_string(k.x) + "," +
                   std::to_string(k.y) + "," +
                   std::to_string(k.z) + ")>";
        })
        .def("__eq__", &trivima::CellKey::operator==);

    // Constants
    m.attr("BASE_CELL_SIZE") = trivima::BASE_CELL_SIZE;
    m.attr("EYE_HEIGHT") = trivima::EYE_HEIGHT;
    m.attr("MAX_VISIBLE_CELLS") = trivima::MAX_VISIBLE_CELLS;
    m.attr("NEURAL_TEXTURE_DIM") = trivima::NEURAL_TEXTURE_DIM;
}
