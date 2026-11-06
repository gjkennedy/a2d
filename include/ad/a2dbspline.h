#ifndef A2D_BSPLINE_H
#define A2D_BSPLINE_H

#include <type_traits>

#include "../a2ddefs.h"
#include "../a2dobj.h"
#include "a2dsamples.h"

namespace A2D {

/*
  B-splines of order K on the open uniform knots of the unit interval

  A spline with M control points has the knot 0 repeated K times, the interior
  knots j / (M - K + 1), and the knot 1 repeated K times, the knot vector of
  the MBI package. The value at t sums the K basis functions that are nonzero
  on the span of t, and outside the unit interval the end spans extend the
  spline.

  The inverse of a coordinate spline with increasing coefficients returns the
  t where the spline takes the value x. An x below the first or above the last
  coefficient holds t at 0 or 1 with zero derivatives, as the clamped Newton
  iteration of MBI does. The derivatives of t are 1 / s' and -s'' / s'^3 at
  the root.
*/

// Knot i of the open uniform knot vector of M control points of order K
template <int K, int M>
A2D_FUNCTION double BSplineKnot(int i) {
  if (i < K) {
    return 0.0;
  } else if (i >= M) {
    return 1.0;
  }
  return double(i - K + 1) / double(M - K + 1);
}

// First control point of the span that holds t
template <int K, int M>
A2D_FUNCTION int BSplineSpan(double t) {
  constexpr int nspan = M - K + 1;
  if (!(t > 0.0)) {
    return 0;
  } else if (t >= 1.0) {
    return nspan - 1;
  }

  // The span from the uniform spacing is corrected against the knot values
  int s = static_cast<int>(t * nspan);
  if (s > nspan - 1) {
    s = nspan - 1;
  }
  while (s > 0 && t < BSplineKnot<K, M>(s + K - 1)) {
    s--;
  }
  while (s < nspan - 1 && t >= BSplineKnot<K, M>(s + K)) {
    s++;
  }
  return s;
}

// Values and first two derivatives of the K basis functions of span s
template <int K, int M>
A2D_FUNCTION void BSplineBasisFunctions(double t, int s, double N0[],
                                        double N1[], double N2[]) {
  constexpr int p = K - 1;
  constexpr int nd = p < 2 ? p : 2;
  const int i = s + p;

  // Basis functions of every degree with the knot differences below them
  double ndu[K][K], left[K], right[K];
  ndu[0][0] = 1.0;
  for (int j = 1; j <= p; j++) {
    left[j] = t - BSplineKnot<K, M>(i + 1 - j);
    right[j] = BSplineKnot<K, M>(i + j) - t;
    double saved = 0.0;
    for (int r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const double temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }

  double ders[3][K];
  for (int j = 0; j <= p; j++) {
    ders[0][j] = ndu[j][p];
    ders[1][j] = 0.0;
    ders[2][j] = 0.0;
  }

  // Derivatives from the differences of the lower degree functions
  double a[2][K];
  for (int r = 0; r <= p; r++) {
    int s1 = 0, s2 = 1;
    a[0][0] = 1.0;
    for (int k = 1; k <= nd; k++) {
      double d = 0.0;
      const int rk = r - k;
      const int pk = p - k;
      if (r >= k) {
        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
        d = a[s2][0] * ndu[rk][pk];
      }
      const int j1 = rk >= -1 ? 1 : -rk;
      const int j2 = r - 1 <= pk ? k - 1 : p - r;
      for (int j = j1; j <= j2; j++) {
        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
        d += a[s2][j] * ndu[rk + j][pk];
      }
      if (r <= pk) {
        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
        d += a[s2][k] * ndu[r][pk];
      }
      ders[k][r] = d;
      const int swap = s1;
      s1 = s2;
      s2 = swap;
    }
  }

  double factor = p;
  for (int k = 1; k <= nd; k++) {
    for (int j = 0; j <= p; j++) {
      ders[k][j] *= factor;
    }
    factor *= (p - k);
  }

  for (int j = 0; j <= p; j++) {
    N0[j] = ders[0][j];
    N1[j] = ders[1][j];
    N2[j] = ders[2][j];
  }
}

// Spline value at t with its first and second derivatives
template <int K, class Coef>
A2D_FUNCTION double BSplineValue(const Coef& coef, double t, double& d1,
                                 double& d2) {
  constexpr int M = Coef::size;
  static_assert(M >= K, "A B-spline needs at least as many points as its order");

  const int s = BSplineSpan<K, M>(t);
  double N0[K], N1[K], N2[K];
  BSplineBasisFunctions<K, M>(t, s, N0, N1, N2);

  double val = 0.0;
  d1 = 0.0;
  d2 = 0.0;
  for (int j = 0; j < K; j++) {
    const double c = RealPart(coef(s + j));
    val += N0[j] * c;
    d1 += N1[j] * c;
    d2 += N2[j] * c;
  }
  return val;
}

// Basis function index of M control points at t with its first and second derivatives
template <int K, int M>
A2D_FUNCTION double BSplineBasisValue(int index, double t, double& d1,
                                      double& d2) {
  static_assert(M >= K, "A B-spline needs at least as many points as its order");

  const int s = BSplineSpan<K, M>(t);
  d1 = 0.0;
  d2 = 0.0;
  if (index < s || index >= s + K) {
    return 0.0;
  }
  double N0[K], N1[K], N2[K];
  BSplineBasisFunctions<K, M>(t, s, N0, N1, N2);
  d1 = N1[index - s];
  d2 = N2[index - s];
  return N0[index - s];
}

// Parameter where the increasing coordinate spline takes the value x
template <int K, class Coef>
A2D_FUNCTION double BSplineInverseValue(const Coef& coef, double x, double& d1,
                                        double& d2) {
  constexpr int M = Coef::size;
  const double lo = RealPart(coef(0));
  const double hi = RealPart(coef(M - 1));
  d1 = 0.0;
  d2 = 0.0;
  if (!(x >= lo)) {
    return 0.0;
  } else if (!(x <= hi)) {
    return 1.0;
  }

  // Newton steps that fall outside the bracket of the root are bisected
  double a = 0.0, b = 1.0;
  double t = (x - lo) / (hi - lo);
  double s1 = 0.0, s2 = 0.0;
  for (int iter = 0; iter < 100; iter++) {
    const double f = BSplineValue<K>(coef, t, s1, s2) - x;
    if (f < 0.0) {
      a = t;
    } else if (f > 0.0) {
      b = t;
    } else {
      break;
    }
    double next = t - f / s1;
    if (!(s1 > 0.0) || !(next > a && next < b)) {
      next = 0.5 * (a + b);
    }
    const double step = next - t;
    t = next;
    if (absfunc(step) <= 4e-16 || b - a <= 4e-16) {
      break;
    }
  }

  BSplineValue<K>(coef, t, s1, s2);
  d1 = 1.0 / s1;
  d2 = -s2 * d1 * d1 * d1;
  return t;
}

// The spline of a vector table or of one column of a matrix table
template <int K, class Samples>
class BSplineEval {
 public:
  A2D_FUNCTION BSplineEval(const Samples& samples) : samples(samples) {}
  A2D_FUNCTION double operator()(double t, double& d1, double& d2) const {
    return BSplineValue<K>(samples, t, d1, d2);
  }

 private:
  Samples samples;
};

// One basis function of M control points
template <int K, int M>
class BSplineBasisEval {
 public:
  A2D_FUNCTION BSplineBasisEval(int index) : index(index) {}
  A2D_FUNCTION double operator()(double t, double& d1, double& d2) const {
    return BSplineBasisValue<K, M>(index, t, d1, d2);
  }

 private:
  int index;
};

// The inverse of a coordinate spline
template <int K, class Samples>
class BSplineInverseEval {
 public:
  A2D_FUNCTION BSplineInverseEval(const Samples& samples) : samples(samples) {}
  A2D_FUNCTION double operator()(double x, double& d1, double& d2) const {
    return BSplineInverseValue<K>(samples, x, d1, d2);
  }

 private:
  Samples samples;
};

template <class Eval, class A, class Ta, class T, bool CA>
class SplineExpr : public ADExpr<SplineExpr<Eval, A, Ta, T, CA>, T> {
 public:
  using expr_t = typename std::conditional<CA, const ADExpr<A, Ta>,
                                           ADExpr<A, Ta>>::type;
  using A_t = typename std::conditional<CA, A, A&>::type;
  A2D_FUNCTION SplineExpr(const Eval& fn, expr_t& a0)
      : fn(fn), a(a0.self()), val(0.0), bval(0.0), d1(0.0) {}
  A2D_FUNCTION void eval() {
    a.eval();
    double g1, g2;
    val = T(fn(RealPart(a.value()), g1, g2));
    d1 = T(g1);
  }
  A2D_FUNCTION void forward() {
    a.forward();
    bval = d1 * a.bvalue();
  }
  A2D_FUNCTION void reverse() {
    a.bvalue() += d1 * bval;
    a.reverse();
  }
  A2D_FUNCTION void bzero() {
    bval = T(0.0);
    a.bzero();
  }
  A2D_FUNCTION T& value() { return val; }
  A2D_FUNCTION const T& value() const { return val; }
  A2D_FUNCTION T& bvalue() { return bval; }
  A2D_FUNCTION const T& bvalue() const { return bval; }

 private:
  Eval fn;
  A_t a;
  T val, bval, d1;
};

template <class Eval, class A, class Ta, class T, bool CA>
class SplineExpr2 : public A2DExpr<SplineExpr2<Eval, A, Ta, T, CA>, T> {
 public:
  using expr_t = typename std::conditional<CA, const A2DExpr<A, Ta>,
                                           A2DExpr<A, Ta>>::type;
  using A_t = typename std::conditional<CA, A, A&>::type;
  A2D_FUNCTION SplineExpr2(const Eval& fn, expr_t& a0)
      : fn(fn),
        a(a0.self()),
        val(0.0),
        bval(0.0),
        pval(0.0),
        hval(0.0),
        d1(0.0),
        d2(0.0) {}
  A2D_FUNCTION void eval() {
    a.eval();
    double g1, g2;
    val = T(fn(RealPart(a.value()), g1, g2));
    d1 = T(g1);
    d2 = T(g2);
  }
  A2D_FUNCTION void reverse() {
    a.bvalue() += d1 * bval;
    a.reverse();
  }
  A2D_FUNCTION void hforward() {
    a.hforward();
    pval = d1 * a.pvalue();
  }
  A2D_FUNCTION void hreverse() {
    a.hvalue() += d1 * hval + d2 * bval * a.pvalue();
    a.hreverse();
  }
  A2D_FUNCTION void bzero() {
    bval = T(0.0);
    a.bzero();
  }
  A2D_FUNCTION void hzero() {
    hval = T(0.0);
    a.hzero();
  }
  A2D_FUNCTION T& value() { return val; }
  A2D_FUNCTION const T& value() const { return val; }
  A2D_FUNCTION T& bvalue() { return bval; }
  A2D_FUNCTION const T& bvalue() const { return bval; }
  A2D_FUNCTION T& pvalue() { return pval; }
  A2D_FUNCTION const T& pvalue() const { return pval; }
  A2D_FUNCTION T& hvalue() { return hval; }
  A2D_FUNCTION const T& hvalue() const { return hval; }

 private:
  Eval fn;
  A_t a;
  T val, bval, pval, hval, d1, d2;
};

template <class Eval, typename T,
          std::enable_if_t<is_passive_scalar_v<T>, int> = 0>
A2D_FUNCTION T spline_eval(const Eval& fn, const T x) {
  double d1, d2;
  return T(fn(RealPart(x), d1, d2));
}

template <class Eval, typename T, int N,
          std::enable_if_t<is_passive_scalar_v<T>, int> = 0>
A2D_FUNCTION ADScalar<T, N> spline_eval(const Eval& fn,
                                        const ADScalar<T, N>& x) {
  double d1, d2;
  ADScalar<T, N> out(T(fn(RealPart(x.value), d1, d2)));
  for (int i = 0; i < N; i++) {
    out.deriv[i] = d1 * x.deriv[i];
  }
  return out;
}

template <class Eval, class A, class Ta>
A2D_FUNCTION auto spline_eval(const Eval& fn, const ADExpr<A, Ta>& x) {
  using T = remove_cvref_t<Ta>;
  return SplineExpr<Eval, A, Ta, T, true>(fn, x);
}

template <class Eval, class A, class Ta>
A2D_FUNCTION auto spline_eval(const Eval& fn, ADExpr<A, Ta>& x) {
  using T = remove_cvref_t<Ta>;
  return SplineExpr<Eval, A, Ta, T, false>(fn, x);
}

template <class Eval, class A, class Ta>
A2D_FUNCTION auto spline_eval(const Eval& fn, const A2DExpr<A, Ta>& x) {
  using T = remove_cvref_t<Ta>;
  return SplineExpr2<Eval, A, Ta, T, true>(fn, x);
}

template <class Eval, class A, class Ta>
A2D_FUNCTION auto spline_eval(const Eval& fn, A2DExpr<A, Ta>& x) {
  using T = remove_cvref_t<Ta>;
  return SplineExpr2<Eval, A, Ta, T, false>(fn, x);
}

// Spline with the coefficients of a vector table
template <int K, class Table, class X>
A2D_FUNCTION auto bspline(const Table& coef, X&& t) {
  using Samples = VecSamples<Table>;
  return spline_eval(BSplineEval<K, Samples>(Samples(coef)),
                     static_cast<X&&>(t));
}

// Spline with the coefficients in one column of a matrix table
template <int K, class Table, class X>
A2D_FUNCTION auto bspline(const Table& coef, int column, X&& t) {
  using Samples = MatColumnSamples<Table>;
  return spline_eval(BSplineEval<K, Samples>(Samples(coef, column)),
                     static_cast<X&&>(t));
}

// Basis function index of a spline with M control points
template <int K, int M, class X>
A2D_FUNCTION auto bspline_basis(int index, X&& t) {
  return spline_eval(BSplineBasisEval<K, M>(index), static_cast<X&&>(t));
}

// Inverse of the coordinate spline with the coefficients of a vector table
template <int K, class Table, class X>
A2D_FUNCTION auto bspline_inverse(const Table& coef, X&& x) {
  using Samples = VecSamples<Table>;
  return spline_eval(BSplineInverseEval<K, Samples>(Samples(coef)),
                     static_cast<X&&>(x));
}

// Inverse of the coordinate spline in one column of a matrix table
template <int K, class Table, class X>
A2D_FUNCTION auto bspline_inverse(const Table& coef, int column, X&& x) {
  using Samples = MatColumnSamples<Table>;
  return spline_eval(BSplineInverseEval<K, Samples>(Samples(coef, column)),
                     static_cast<X&&>(x));
}

}  // namespace A2D

#endif  // A2D_BSPLINE_H
