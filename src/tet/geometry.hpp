#pragma once

#include <cmath>

#ifdef __CUDACC__
#define TET_HD __host__ __device__
#else
#define TET_HD
#endif

namespace tet {

struct Vec3 {
  double x = 0, y = 0, z = 0;
};

TET_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
TET_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
TET_HD inline Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
TET_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
TET_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
TET_HD inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
TET_HD inline double coord(Vec3 v, int d) { return d == 0 ? v.x : d == 1 ? v.y : v.z; }

// Zeroth and first moments of a region: volume = ∫ dV and first = ∫ (x, y, z) dV, so its centroid
// is first / volume.
struct Moments {
  double volume = 0;
  Vec3   first;
};

namespace detail {

// `out` from moments `m` taken about o, unless the volume is not above rel_tol * min_volume (then
// `out` is zero and the result false).
TET_HD inline bool finish_moments(const Moments& m, Vec3 o, double rel_tol, double min_volume, Moments& out) {
  out = Moments();
  if (m.volume <= rel_tol * min_volume) return false;
  out = {m.volume, m.first + m.volume * o};
  return true;
}

}  // namespace detail

// A tet, for the volumes of hexes (Hex::tets).
struct Tet {
  Vec3 v[4];

  // Signed volume; positive when (v1-v0, v2-v0, v3-v0) is right-handed.
  TET_HD double signed_volume() const { return dot(v[1] - v[0], cross(v[2] - v[0], v[3] - v[0])) / 6.0; }
};

}  // namespace tet
