#pragma once

namespace consts {

constexpr double reg_term = 1e-6;

}

namespace geom {

struct Plane
{
	int id;
	double a, b, c, d;
};

using PlaneList = std::vector<Plane>;
using BasisList = std::vector<double3>;

} // namespace geom

namespace gpu {

void deformPlanesPolypoint(
    __out geom::PlaneList &outPlanes,
    const geom::PlaneList &inPlanes,
    const geom::BasisList &origBasises,
    const geom::BasisList &resBasises);

} // namespace gpu
