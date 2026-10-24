#pragma once

// Hexahedral meshes, unstructured (vertices + 8-tuples) and Cartesian (implicit), and the moments
// of the overlap of two hexes or of a hex and a box.
//
// A hex's faces need not be planar, and the hex need not be convex: its geometry is that of its
// boundary triangles. Each face is split into 4 triangles around the face centre (the average of its
// corners), 24 in all; coned to the hex centre (the average of all 8 corners), they are the hex's 24
// tets. A tet that comes out inverted counts negatively, so the hex is the region its boundary
// triangles wind around either way. (Or, with HexSplit::diagonal, its 12 boundary triangles along a
// diagonal of each face, and the 6 tets they bound.) The moments of an overlap come from the
// boundary triangles directly (projection.hpp).

#include <array>
#include <random>
#include <vector>

#include "tet/projection.hpp"

namespace tet {

// A pair of overlapping elements: `a` indexes the first mesh, `b` the second.
struct Pair {
  int a, b;
  TET_HD bool operator<(const Pair& o) const { return a != o.a ? a < o.a : b < o.b; }
  TET_HD bool operator==(const Pair& o) const { return a == o.a && b == o.b; }
};

// The overlap of element `pair.a` of the first mesh with element `pair.b` of the second.
struct Overlap {
  Pair    pair;
  Moments moments;
};

// Corner k of face f, counterclockwise seen from outside. Corners 0-3 go counterclockwise around the
// bottom face (seen from above) and 4-7 sit above them, as in VTK.
TET_HD inline int hex_face_vertex(int f, int k) {
  static constexpr int F[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  return F[f][k];
}

// Axis-aligned box.
struct Box {
  Vec3 lo, hi;
};

TET_HD inline Box box_of(const Vec3* v, int n) {
  Box b{v[0], v[0]};
  for (int i = 1; i < n; ++i) {
    b.lo = {fmin(b.lo.x, v[i].x), fmin(b.lo.y, v[i].y), fmin(b.lo.z, v[i].z)};
    b.hi = {fmax(b.hi.x, v[i].x), fmax(b.hi.y, v[i].y), fmax(b.hi.z, v[i].z)};
  }
  return b;
}

TET_HD inline bool overlap_yz(const Box& a, const Box& b) {
  return a.lo.y <= b.hi.y && b.lo.y <= a.hi.y && a.lo.z <= b.hi.z && b.lo.z <= a.hi.z;
}
TET_HD inline bool overlap(const Box& a, const Box& b) { return a.lo.x <= b.hi.x && b.lo.x <= a.hi.x && overlap_yz(a, b); }

// How a hex's faces are cut into its boundary triangles. face_centers (the default, as above): into
// 4 triangles around the face centre, 24 in all, the surface of the 24 tets. diagonal: into 2 along
// the diagonal through corner 0 or 6, 12 in all, the surface of the 6 tets around the diagonal from
// corner 0 to 6 (the subzones of RAJAPerf's INTSC_HEXHEX). The two are the same solid when the faces
// are planar, and differ by the faces' curvature otherwise.
enum class HexSplit { face_centers, diagonal };
TET_HD constexpr int hex_triangles(HexSplit s) { return s == HexSplit::face_centers ? 24 : 12; }
TET_HD constexpr int hex_tets(HexSplit s) { return s == HexSplit::face_centers ? 24 : 6; }

// The 6 corners around the diagonal from corner 0 to 6, in order around it (counterclockwise seen
// from corner 0).
TET_HD inline int hex_diagonal_ring(int i) {
  static constexpr int R[7] = {1, 5, 4, 7, 3, 2, 1};
  return R[i];
}

// The corners of boundary triangle k < 12 of HexSplit::diagonal: corner 0 (k < 6) or 6 (k >= 6) and
// an edge of the ring, outward.
TET_HD inline void hex_diagonal_triangle(int k, int* c) {
  bool low = k < 6;
  c[0] = low ? 0 : 6, c[1] = hex_diagonal_ring(low ? k : 12 - k), c[2] = hex_diagonal_ring(low ? k + 1 : 11 - k);
}

struct Hex {
  Vec3 v[8];

  TET_HD Vec3 center() const {
    Vec3 c;
    for (int i = 0; i < 8; ++i) c = c + v[i];
    return 0.125 * c;
  }
  TET_HD Vec3 face_center(int f) const {
    return 0.25 * (v[hex_face_vertex(f, 0)] + v[hex_face_vertex(f, 1)] + v[hex_face_vertex(f, 2)] + v[hex_face_vertex(f, 3)]);
  }

  // Boundary triangle k < hex_triangles(split), outward, relative to o. face_centers: it joins face
  // k / 4's centre to its edge k % 4. diagonal: corner 0 (k < 6) or 6 (k >= 6) and an edge of the ring.
  template <HexSplit split = HexSplit::face_centers>
  TET_HD void triangle(int k, Vec3 o, Vec3* t) const {
    if constexpr (split == HexSplit::face_centers) {
      int f = k / 4, e = k % 4;
      t[0] = face_center(f) - o;
      t[1] = v[hex_face_vertex(f, e)] - o;
      t[2] = v[hex_face_vertex(f, (e + 1) % 4)] - o;
    } else {
      int c[3];
      hex_diagonal_triangle(k, c);
      for (int i = 0; i < 3; ++i) t[i] = v[c[i]] - o;
    }
  }

  // The hex_tets(split) tets that the boundary triangles bound (positively oriented unless inverted).
  // face_centers: each triangle coned to the centre. diagonal: the diagonal from corner 0 to 6 and an
  // edge of the ring.
  template <HexSplit split = HexSplit::face_centers>
  TET_HD void tets(Tet* t) const {
    if constexpr (split == HexSplit::face_centers) {
      Vec3 c = center();
      for (int f = 0; f < 6; ++f) {
        Vec3 fc = face_center(f);
        for (int e = 0; e < 4; ++e) t[4 * f + e] = Tet{{c, fc, v[hex_face_vertex(f, e)], v[hex_face_vertex(f, (e + 1) % 4)]}};
      }
    } else {
      for (int i = 0; i < 6; ++i) t[i] = Tet{{v[0], v[hex_diagonal_ring(i + 1)], v[hex_diagonal_ring(i)], v[6]}};
    }
  }

  TET_HD double volume(HexSplit split = HexSplit::face_centers) const {
    Tet t[24];
    if (split == HexSplit::face_centers)
      tets(t);
    else
      tets<HexSplit::diagonal>(t);
    double vol = 0;
    for (int k = 0; k < hex_tets(split); ++k) vol += t[k].signed_volume();
    return vol;
  }
};

// An unstructured hex mesh: vertex positions plus 8-tuples of vertex indices (corner order as above).
struct HexMesh {
  std::vector<Vec3> vertices;
  std::vector<std::array<int, 8>> hexes;

  size_t num_hexes() const { return hexes.size(); }

  Hex hex(size_t i) const {
    Hex h;
    for (int k = 0; k < 8; ++k) h.v[k] = vertices[hexes[i][k]];
    return h;
  }
};

// A Cartesian mesh: n[0] x n[1] x n[2] equal boxes filling [lo, hi], with nothing stored. Cell
// (i, j, k) has index i + n[0] (j + n[1] k).
struct CartesianMesh {
  Vec3 lo, hi;
  int  n[3];

  size_t num_cells() const { return size_t(n[0]) * n[1] * n[2]; }

  // Grid plane i along axis d (exactly lo and hi at the ends).
  TET_HD double plane(int d, int i) const {
    double l = coord(lo, d), h = coord(hi, d);
    return i == 0 ? l : i == n[d] ? h : l + (h - l) * i / n[d];
  }
  TET_HD Box cell(int i, int j, int k) const {
    return {{plane(0, i), plane(1, j), plane(2, k)}, {plane(0, i + 1), plane(1, j + 1), plane(2, k + 1)}};
  }
  TET_HD int index(int i, int j, int k) const { return i + n[0] * (j + n[1] * k); }

  // The cells i0..i1 along axis d that overlap [a, b] (none if i0 > i1).
  TET_HD void cell_range(int d, double a, double b, int& i0, int& i1) const {
    double l = coord(lo, d), h = coord(hi, d);
    i0 = 0;
    i1 = -1;
    if (b < l || a > h) return;
    double s = n[d] / (h - l);
    i0 = (int)fmax(0.0, fmin(n[d] - 1.0, floor((a - l) * s)));
    i1 = (int)fmax(0.0, fmin(n[d] - 1.0, floor((b - l) * s)));
    while (i0 > 0 && plane(d, i0) > a) --i0;  // in case the division rounded the wrong way
    while (i1 < n[d] - 1 && plane(d, i1 + 1) < b) ++i1;
  }
};

// n^3 hexes filling [-size/2, size/2]^2 x [0, size] (like make_grid_mesh), with every interior
// vertex moved by a random offset of up to `jitter` times the spacing in each coordinate, so that
// faces are not planar and cells need not be convex. Boundary vertices stay put, so the mesh fills
// the box exactly. Then rotated by `angle` about the z axis and shifted by `offset`.
inline HexMesh make_hex_grid_mesh(int n, double size = 1.0, double jitter = 0.0, unsigned seed = 1, Vec3 offset = {},
                                  double angle = 0.0) {
  HexMesh m;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-jitter, jitter);
  const double h = size / n, c = std::cos(angle), s = std::sin(angle);
  auto vid = [n](int i, int j, int k) { return (k * (n + 1) + j) * (n + 1) + i; };
  for (int k = 0; k <= n; ++k)
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i) {
        double x = i * h - size / 2, y = j * h - size / 2, z = k * h;
        if (i > 0 && i < n && j > 0 && j < n && k > 0 && k < n) {
          x += u(rng) * h;
          y += u(rng) * h;
          z += u(rng) * h;
        }
        m.vertices.push_back({c * x - s * y + offset.x, s * x + c * y + offset.y, z + offset.z});
      }
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        m.hexes.push_back({vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k), vid(i, j, k + 1),
                           vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)});
  return m;
}

// Moments of the overlap of two hexes, both with their faces split as `split` says: every pair of
// boundary triangles whose boxes can overlap (add_projected_pair).
template <HexSplit split = HexSplit::face_centers>
TET_HD inline Moments hex_hex_moments(const Hex& a, const Hex& b) {
  constexpr int nt = hex_triangles(split);
  Moments       m;
  Vec3          o = a.center();  // local origin, for conditioning
  Vec3          s[nt][3];
  Box           bs[nt];
  for (int j = 0; j < nt; ++j) {
    b.triangle<split>(j, o, s[j]);
    bs[j] = box_of(s[j], 3);
  }
  for (int i = 0; i < nt; ++i) {
    Vec3 t[3];
    a.triangle<split>(i, o, t);
    Box bt = box_of(t, 3);
    for (int j = 0; j < nt; ++j)
      if (bt.hi.x > bs[j].lo.x && overlap_yz(bt, bs[j])) add_projected_pair(t, s[j], 3, m);
  }
  m.first = m.first + m.volume * o;
  return m;
}

// The same, choosing the split at run time.
inline Moments hex_hex_moments(const Hex& a, const Hex& b, HexSplit split) {
  return split == HexSplit::diagonal ? hex_hex_moments<HexSplit::diagonal>(a, b) : hex_hex_moments(a, b);
}

// Moments of the overlap of a hex (its 24 triangles) and the box b: every triangle whose box can
// reach the box's (add_projected_box).
TET_HD inline Moments hex_box_moments(const Hex& a, const Box& b) {
  Moments m;
  Vec3    o = 0.5 * (b.lo + b.hi);  // local origin, for conditioning
  Box     rel{b.lo - o, b.hi - o};
  for (int f = 0; f < 6; ++f) {
    Vec3 fc = a.face_center(f) - o;
    for (int e = 0; e < 4; ++e) {
      Vec3 t[3] = {fc, a.v[hex_face_vertex(f, e)] - o, a.v[hex_face_vertex(f, (e + 1) % 4)] - o};
      Box  bt   = box_of(t, 3);
      if (bt.hi.x > rel.lo.x && overlap_yz(bt, rel)) add_projected_box(t, rel.lo, rel.hi, m);
    }
  }
  m.first = m.first + m.volume * o;
  return m;
}

}  // namespace tet
