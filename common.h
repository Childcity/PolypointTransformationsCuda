#pragma once

namespace geom {
using real = double;
using real3 = double3;
} // namespace geom

namespace consts {

constexpr geom::real reg_term = 1e-6;

}

namespace geom {

struct Plane
{
	int id;
	real a, b, c, d;
};

using PlaneList = std::vector<Plane>;
using BasisList = std::vector<real3>;

} // namespace geom

namespace gpu {

void deformPlanesPolypoint(
    __out geom::PlaneList &outPlanes,
    const geom::PlaneList &inPlanes,
    const geom::BasisList &origBasises,
    const geom::BasisList &resBasises);

} // namespace gpu
