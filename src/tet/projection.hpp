#pragma once

// Volume and first moments of the overlap of two solids straight from their boundary triangles,
// without decomposing them into convex pieces. What it computes is ∫ w_A w_B (1, x) dV, with w the
// winding numbers of the two closed, outward-oriented surfaces: the overlap of the solids they
// bound, convex or not (and for an element with inverted parts, the same signed overlap as its
// decomposition into signed tets).
//
// Integrate along x first. A ray parallel to x crosses triangle T of A at x = x_T(y, z), where w_A
// jumps by -σ_T, the sign of the x-component of T's outward normal. Both surfaces are closed, so
// the terms at x = ±∞ cancel and, over the triangles the ray crosses,
//   ∫ w_A w_B dx = -Σ_{T ∈ A, S ∈ B} σ_T σ_S max(x_T, x_S).
// Writing max(x_T, x_S) = x_S + (x_T - x_S)⁺, the x_S part sums to zero over T (A is closed), so
// integrating over the rays leaves only the parts where T is above S:
//   volume = -Σ_{T, S} σ_T σ_S ∬_{π(T) ∩ π(S), x_T > x_S} (x_T - x_S) dy dz
//   first  = -Σ_{T, S} σ_T σ_S ∬_{π(T) ∩ π(S), x_T > x_S} ((x_T² - x_S²) / 2, y (x_T - x_S), z (x_T - x_S)) dy dz
// with π the projection onto the yz-plane. Each term needs only the overlap of two projected
// polygons, cut where the planes of T and S cross. Every term is continuous in the vertex positions
// (a triangle parallel to x has no projected area, and the integrand vanishes along the cut), so
// coincident faces, edges and vertices need no tie-breaking.

#include "tet/error.hpp"
#include "tet/geometry.hpp"

namespace tet {

namespace detail {

constexpr int kMaxProjVerts = 9;  // a triangle cut by 6 lines (the box case)

// A convex polygon in the yz-plane.
struct ProjPoly {
  struct Vert {
    double y, z;
  } v[kMaxProjVerts];
  int n;
};

// The height x(y, z) of the plane of a polygon p, from its normal n: x = p0.x - (n.y (y - p0.y) +
// n.z (z - p0.z)) / n.x. Only for polygons not (nearly) parallel to x: theirs have (almost) no
// projected area, and dividing by their n.x would only amplify rounding errors (and decide
// differently on host and device), so they are skipped. Elsewhere, the error the division brings
// shrinks with the projected area it is weighted by.
struct PlaneX {
  Vec3   p0;
  double ry, rz;
  TET_HD double operator()(const ProjPoly::Vert& v) const { return p0.x - (ry * (v.y - p0.y) + rz * (v.z - p0.z)); }
};

// Whether a polygon with normal n is (nearly) parallel to x, and so skipped: |n.x| <= 1e-12 |n|, squared.
TET_HD inline bool parallel_to_x(Vec3 n) { return n.x * n.x <= 1e-24 * dot(n, n); }

// False if p is (nearly) parallel to x; n.x gives the orientation of its projection.
TET_HD inline bool plane_x(const Vec3* p, Vec3& n, PlaneX& h) {
  n = cross(p[1] - p[0], p[2] - p[0]);
  if (parallel_to_x(n)) return false;
  double r = 1 / n.x;
  h = {p[0], n.y * r, n.z * r};
  return true;
}

// Where the edge a -> b crosses the line where `dist` is 0, from the values da and db (of opposite
// signs) there. On a GPU, the reciprocal is the hardware's approximation and 3 fused multiply-adds
// (relative error about e^3 for its e of 2^-23, so within an ulp or so; cheaper and shorter than
// __drcp_rn, which also rounds correctly: on an A100 at 1200 MHz, INTSC_HEXHEX subzones 1.63 -> 1.59
// ms at a jitter of 0.3). ponytail: the approximation flushes a denormal da - db to 0, which would
// take distances below 1e-308 (coordinates near 1e-154); a guard would cost the time saved.
TET_HD inline ProjPoly::Vert crossing(const ProjPoly::Vert& a, const ProjPoly::Vert& b, double da, double db) {
#ifdef __CUDA_ARCH__
  double x = da - db, r;
  asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
  double e = fma(-x, r, 1.0);
  e = fma(e, e, e);
  r = fma(r, e, r);
  double t = da * r;
#else
  double t = da / (da - db);
#endif
  return {a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)};
}

// Keeps the part of p where the linear function `dist` (of a vertex) is >= 0, in place (so that a GPU
// lane needs room for one polygon, not two). Vertex i + 1 is read before the writes for edge i, and
// those stay below it: after edge i there are at most i + 2 vertices out when at most 2 edges cross,
// as for a convex polygon. Rounding can make a polygon that is nearly flat along the line cross it
// more often (coincident faces do): then the vertices not yet read move up by `s` to make room. As
// at most i + 1 + s are out before edge i, that only happens when edge i writes 2 (it keeps a and
// crosses) with exactly that many out.
template <class Dist>
TET_HD inline void clip_proj(ProjPoly& p, Dist dist) {
  const int m = p.n;
  if (!m) return;
  ProjPoly::Vert first = p.v[0], a = first;
  double         d0 = dist(first), da = d0;
  int            n = 0, s = 0;
  // The edge from vertex i (a) to b. The last edge, back to the first vertex, is peeled off the loop,
  // which then has no wrap-around to select (on an A100 at 1200 MHz, INTSC_HEXRECT 2.08 -> 1.95 ms,
  // INTSC_HEXHEX subzones 1.20 -> 1.13 ms at a jitter of 0.3).
  auto edge = [&](int i, const ProjPoly::Vert& b, double db) {
    bool ka = da >= 0, kb = db >= 0;
    ASSERT( n + ka + (ka != kb) <= kMaxProjVerts, "projected polygon exceeds fixed capacity" );
    if (ka) p.v[n++] = a;
    if (ka != kb) {
      if (ka && n == i + 2 + s && i + 2 < m) {  // the crossing would land on vertex i + 2, not yet read
        ASSERT( m + s < kMaxProjVerts, "projected polygon exceeds fixed capacity" );
        for (int j = m - 1 + s; j >= i + 2 + s; --j) p.v[j + 1] = p.v[j];
        ++s;
      }
      p.v[n++] = crossing(a, b, da, db);
    }
    a = b, da = db;
  };
  for (int i = 0; i + 1 < m; ++i) {
    ProjPoly::Vert b = p.v[i + 1 + s];
    edge(i, b, dist(b));
  }
  edge(m - 1, first, d0);  // the last edge, back to the first vertex
  p.n = n;
}

// The same for the triangle v (in registers rather than p), into p, for the first clip of a
// triangle: the loop unrolls and the 3 distances are independent of each other (on an A100 at
// 1200 MHz, INTSC_HEXHEX subzones 1.59 -> 1.54 ms at a jitter of 0.3, INTSC_HEXRECT 2.30 -> 2.18 ms).
template <class Dist>
TET_HD inline void clip_proj(const ProjPoly::Vert (&v)[3], Dist dist, ProjPoly& p) {
  double d[3] = {dist(v[0]), dist(v[1]), dist(v[2])};
  int    n    = 0;
  for (int k = 0; k < 3; ++k) {
    int l = k < 2 ? k + 1 : 0;
    if (d[k] >= 0) p.v[n++] = v[k];
    if ((d[k] >= 0) != (d[l] >= 0)) p.v[n++] = crossing(v[k], v[l], d[k], d[l]);
  }
  p.n = n;
}

// Adds sign * ∬ (xt - xs, (xt² - xs²) / 2, y (xt - xs), z (xt - xs)) over the triangle abc, whose
// orientation in the yz-plane gives the sign of its area, for linear functions xt and xs with the
// vertex values t* and s*.
TET_HD inline void add_tri_integrals(const ProjPoly::Vert& a, const ProjPoly::Vert& b, const ProjPoly::Vert& c, double ta,
                                     double tb, double tc, double sa, double sb, double sc, double sign, Moments& m) {
  // over a triangle of area A, ∬ f g = A / 12 (Σ f_i g_i + Σ f_i Σ g_i) for linear f and g
  // (multiplying by 1 / 6, 1 / 24: dividing by a constant other than a power of 2 is a full
  // division on a GPU)
  double A2 = sign * ((b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y));  // twice the area
  double ft = ta + tb + tc, fs = sa + sb + sc;
  double sy = a.y + b.y + c.y, sz = a.z + b.z + c.z;
  m.volume += A2 * (1.0 / 6) * (ft - fs);
  m.first.x += A2 * (1.0 / 48) * (ta * ta + tb * tb + tc * tc + ft * ft - sa * sa - sb * sb - sc * sc - fs * fs);
  m.first.y += A2 * (1.0 / 24) * (a.y * (ta - sa) + b.y * (tb - sb) + c.y * (tc - sc) + sy * (ft - fs));
  m.first.z += A2 * (1.0 / 24) * (a.z * (ta - sa) + b.z * (tb - sb) + c.z * (tc - sc) + sz * (ft - fs));
}

// The same over `p`, fanned into triangles from its first vertex, for linear functions xt and xs
// (of a vertex).
template <class Xt, class Xs>
TET_HD inline void add_proj_integrals(const ProjPoly& p, double sign, Xt xt, Xs xs, Moments& m) {
  if (p.n < 3) return;
  const ProjPoly::Vert& a = p.v[0];
  double ta = xt(a), tb = xt(p.v[1]), sa = xs(a), sb = xs(p.v[1]);
  for (int k = 1; k + 1 < p.n; ++k) {
    const ProjPoly::Vert& c = p.v[k + 1];
    double tc = xt(c), sc = xs(c);
    add_tri_integrals(a, p.v[k], c, ta, tb, tc, sa, sb, sc, sign, m);
    tb = tc, sb = sc;
  }
}

// Clips p to the half-plane left of the yz-edge a -> b (right of it if `sign` is negative).
TET_HD inline void clip_proj_edge(ProjPoly& p, double ay, double az, double by, double bz, double sign) {
  clip_proj(p, [&](const ProjPoly::Vert& v) { return sign * ((by - ay) * (v.z - az) - (bz - az) * (v.y - ay)); });
}
// The same for the triangle v, into p.
TET_HD inline void clip_proj_edge(const ProjPoly::Vert (&v)[3], double ay, double az, double by, double bz, double sign,
                                  ProjPoly& p) {
  clip_proj(v, [&](const ProjPoly::Vert& w) { return sign * ((by - ay) * (w.z - az) - (bz - az) * (w.y - ay)); }, p);
}

}  // namespace detail

// Adds the terms of the pair (T, S) to `m`: T a triangle of A, S a planar convex polygon of B (3 or
// 4 vertices), both outward-oriented, neither parallel to x, with their planes xt and xs (plane_x)
// and sx the sign of S's normal's x (for callers that pair each triangle with several). `p` is
// scratch space for the polygon (on a GPU, shared memory keeps it out of local memory).
TET_HD inline void add_projected_pair(const Vec3* t, const Vec3* s, int ns, const detail::PlaneX& xt, const detail::PlaneX& xs,
                                      double sx, Moments& m, detail::ProjPoly& p) {
  double smin = s[0].x;
  for (int i = 1; i < ns; ++i) smin = fmin(smin, s[i].x);
  if (fmax(t[0].x, fmax(t[1].x, t[2].x)) <= smin) return;  // T is nowhere above S
  const detail::ProjPoly::Vert tv[3] = {{t[0].y, t[0].z}, {t[1].y, t[1].z}, {t[2].y, t[2].z}};
  detail::clip_proj_edge(tv, s[0].y, s[0].z, s[1].y, s[1].z, sx, p);  // the first edge from registers
  for (int i = 1; i < ns && p.n >= 3; ++i) {  // π(S) is counterclockwise in (y, z) if sx > 0
    Vec3 a = s[i], b = s[i + 1 < ns ? i + 1 : 0];
    detail::clip_proj_edge(p, a.y, a.z, b.y, b.z, sx);
  }
  if (p.n < 3) return;
  // x_T - x_S as one linear function from T's corner p0: half the work of evaluating both planes
  const double c = xt.p0.x - xs({xt.p0.y, xt.p0.z}), dy = xt.ry - xs.ry, dz = xt.rz - xs.rz;
  detail::clip_proj(p, [&](const detail::ProjPoly::Vert& v) { return c - (dy * (v.y - xt.p0.y) + dz * (v.z - xt.p0.z)); });
  detail::add_proj_integrals(p, -sx, xt, xs, m);
}

// Whether the projections of T and S (as above) are certainly disjoint, so that the pair adds
// nothing: an edge of either has each of the other's corners strictly outside it. In floats, for
// speed: t and s are the yz corners, each within 2^-24 L of the exact ones (L the largest
// |coordinate|), and every value that decides a separation, or an orientation, must clear `margin` =
// 2^-18 L^2, beyond its rounding error (under about 50 2^-24 L^2). So what it finds, an exact test
// would find too. Every test is made, without branches: on a GPU, lanes that stop early only wait
// for the others.
TET_HD inline bool separated_yz(const float (*t)[2], const float (*s)[2], float margin) {
  bool sep = false;
  for (int pass = 0; pass < 2; ++pass) {
    const float(*u)[2] = pass ? s : t, (*w)[2] = pass ? t : s;
    float o  = (u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[1][1] - u[0][1]) * (u[2][0] - u[0][0]);
    bool  ok = fabsf(o) > margin;
    float so = o > 0 ? 1.f : -1.f;
    for (int e = 0; e < 3; ++e) {
      const float *a = u[e], *b = u[e + 1 < 3 ? e + 1 : 0];
      float        ey = so * (b[0] - a[0]), ez = so * (b[1] - a[1]);
      auto         side = [&](const float* v) { return ey * (v[1] - a[1]) - ez * (v[0] - a[0]); };
      sep |= ok & (fmaxf(side(w[0]), fmaxf(side(w[1]), side(w[2]))) < -margin);
    }
  }
  return sep;
}

// The same, any T and S (those parallel to x add nothing).
TET_HD inline void add_projected_pair(const Vec3* t, const Vec3* s, int ns, Moments& m, detail::ProjPoly& p) {
  Vec3           nt, n;
  detail::PlaneX xt, xs;
  if (detail::plane_x(t, nt, xt) && detail::plane_x(s, n, xs)) add_projected_pair(t, s, ns, xt, xs, n.x > 0 ? 1.0 : -1.0, m, p);
}

TET_HD inline void add_projected_pair(const Vec3* t, const Vec3* s, int ns, Moments& m) {
  detail::ProjPoly p;
  add_projected_pair(t, s, ns, m, p);
}

// The terms of triangle T of A against the box [lo, hi], whose only faces not parallel to x are
// x = lo (σ = -1) and x = hi (σ = +1). Their two pair terms combine to
//   σ_T ∬_{π(T) ∩ R} clamp(x_T - lo, 0, hi - lo) dy dz
// (R the box's yz-rectangle), and likewise for the first moments, so T is clipped to R only once,
// and triangles below the box drop out. The clamp is hi - lo where x_T >= lo, plus x_T - hi where
// also x_T < hi: the first over the part of π(T) ∩ R above x = lo, the second over the part of that
// below x = hi, clipped from it in place once the first is added. `p` is scratch space, as above.
TET_HD inline void add_projected_box(const Vec3* t, Vec3 lo, Vec3 hi, Moments& m, detail::ProjPoly& p) {
  if (fmax(t[0].x, fmax(t[1].x, t[2].x)) <= lo.x) return;
  Vec3           nt;
  detail::PlaneX xt;
  if (!detail::plane_x(t, nt, xt)) return;
  using V = detail::ProjPoly::Vert;
  auto    lo_x  = [&](const V&) { return lo.x; };
  auto    hi_x  = [&](const V&) { return hi.x; };
  const V tv[3] = {{t[0].y, t[0].z}, {t[1].y, t[1].z}, {t[2].y, t[2].z}};
  detail::clip_proj(tv, [&](const V& v) { return v.y - lo.y; }, p);  // the first line from registers
  detail::clip_proj(p, [&](const V& v) { return hi.y - v.y; });
  detail::clip_proj(p, [&](const V& v) { return v.z - lo.z; });
  detail::clip_proj(p, [&](const V& v) { return hi.z - v.z; });
  detail::clip_proj(p, [&](const V& v) { return xt(v) - lo.x; });  // the part of T above x = lo
  detail::add_proj_integrals(p, 1.0, hi_x, lo_x, m);
  // The integrand x_T - hi vanishes on the cut x = hi, so a triangle lying in that plane (where x_T is
  // exactly hi) adds nothing, whichever side it lands on.
  detail::clip_proj(p, [&](const V& v) { return hi.x - xt(v); });
  detail::add_proj_integrals(p, 1.0, xt, hi_x, m);
}

TET_HD inline void add_projected_box(const Vec3* t, Vec3 lo, Vec3 hi, Moments& m) {
  detail::ProjPoly p;
  add_projected_box(t, lo, hi, m, p);
}

}  // namespace tet
