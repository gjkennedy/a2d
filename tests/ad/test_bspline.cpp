#include <gtest/gtest.h>

#include <cmath>

#include "a2dcore.h"

using namespace A2D;

/*
  Tests of the B-splines of a2dbspline.h

  The spline evaluates on the real part of its argument, so the complex step of
  A2DTest does not apply. The basis functions are checked against the
  Cox-de Boor recursion on knot vectors written out by hand, and the
  expressions are checked against central differences.
*/

// Tolerance relative to the size of the reference value
static double Tol(double ref, double rtol) { return rtol * (1.0 + std::fabs(ref)); }

// The knot interval of t, the last one with a lower knot at or below t, so that
// the end intervals extend past the ends of the unit interval
static int ReferenceSpan(const double u[], int K, int M, double t) {
  int span = K - 1;
  for (int i = K - 1; i < M; i++) {
    if (u[i] <= t && u[i] < u[i + 1]) {
      span = i;
    }
  }
  return span;
}

// Derivative k of basis function i of degree p from the Cox-de Boor recursion,
// where the degree zero function of the knot interval span is one
static double CoxDeBoor(const double u[], int span, int i, int p, int k,
                        double t) {
  if (p == 0) {
    return (k == 0 && i == span) ? 1.0 : 0.0;
  }
  const double dl = u[i + p] - u[i];
  const double dr = u[i + p + 1] - u[i + 1];
  double val = 0.0;
  if (k == 0) {
    if (dl > 0.0) {
      val += (t - u[i]) / dl * CoxDeBoor(u, span, i, p - 1, 0, t);
    }
    if (dr > 0.0) {
      val += (u[i + p + 1] - t) / dr * CoxDeBoor(u, span, i + 1, p - 1, 0, t);
    }
  } else {
    if (dl > 0.0) {
      val += p / dl * CoxDeBoor(u, span, i, p - 1, k - 1, t);
    }
    if (dr > 0.0) {
      val -= p / dr * CoxDeBoor(u, span, i + 1, p - 1, k - 1, t);
    }
  }
  return val;
}

// Knots, basis values and derivatives against the recursion, at knots, inside
// spans, at the ends and past them, with the sums of a partition of unity
template <int K, int M>
static void CheckBasis(const double u[]) {
  for (int i = 0; i < M + K; i++) {
    EXPECT_EQ((BSplineKnot<K, M>(i)), u[i]) << "K " << K << " M " << M
                                            << " knot " << i;
  }

  const double ts[] = {-0.3, 0.0,       0.1,  0.25, 1.0 / 3.0, 0.4, 0.5,
                       0.6,  2.0 / 3.0, 0.75, 0.9,  1.0,       1.2};
  for (double t : ts) {
    const int span = ReferenceSpan(u, K, M, t);
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < M; i++) {
      double d1, d2;
      const double n0 = BSplineBasisValue<K, M>(i, t, d1, d2);
      const double r0 = CoxDeBoor(u, span, i, K - 1, 0, t);
      const double r1 = CoxDeBoor(u, span, i, K - 1, 1, t);
      const double r2 = CoxDeBoor(u, span, i, K - 1, 2, t);
      EXPECT_NEAR(n0, r0, Tol(r0, 1e-13))
          << "K " << K << " M " << M << " i " << i << " t " << t;
      EXPECT_NEAR(d1, r1, Tol(r1, 1e-13))
          << "K " << K << " M " << M << " i " << i << " t " << t;
      EXPECT_NEAR(d2, r2, Tol(r2, 1e-13))
          << "K " << K << " M " << M << " i " << i << " t " << t;
      sum0 += n0;
      sum1 += d1;
      sum2 += d2;
    }
    EXPECT_NEAR(sum0, 1.0, 1e-13) << "K " << K << " M " << M << " t " << t;
    EXPECT_NEAR(sum1, 0.0, 1e-12) << "K " << K << " M " << M << " t " << t;
    EXPECT_NEAR(sum2, 0.0, 1e-11) << "K " << K << " M " << M << " t " << t;
  }
}

TEST(test_bspline, BasisMatchesCoxDeBoor) {
  const double u1[] = {0.0, 0.25, 0.5, 0.75, 1.0};
  const double u2[] = {0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0};
  const double u3[] = {0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
  const double u3b[] = {0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0};
  const double u4[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  const double u4b[] = {0.0, 0.0,  0.0, 0.0, 0.25, 0.5,
                        0.75, 1.0, 1.0, 1.0, 1.0};
  const double u5[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5,
                       0.75, 1.0, 1.0, 1.0, 1.0, 1.0};
  CheckBasis<1, 4>(u1);
  CheckBasis<2, 5>(u2);
  CheckBasis<3, 5>(u3);
  CheckBasis<3, 6>(u3b);
  CheckBasis<4, 4>(u4);
  CheckBasis<4, 7>(u4b);
  CheckBasis<5, 8>(u5);
}

// With four points of order four the basis is the cubic Bernstein basis, here
// and on its polynomial extension
TEST(test_bspline, CubicBernsteinBasis) {
  const double ts[] = {-0.2, 0.0, 0.13, 0.5, 0.77, 1.0, 1.3};
  for (double t : ts) {
    const double s = 1.0 - t;
    const double b0[4] = {s * s * s, 3.0 * t * s * s, 3.0 * t * t * s,
                          t * t * t};
    const double b1[4] = {-3.0 * s * s, 3.0 - 12.0 * t + 9.0 * t * t,
                          6.0 * t - 9.0 * t * t, 3.0 * t * t};
    const double b2[4] = {6.0 * s, -12.0 + 18.0 * t, 6.0 - 18.0 * t, 6.0 * t};
    for (int i = 0; i < 4; i++) {
      double d1, d2;
      const double n0 = BSplineBasisValue<4, 4>(i, t, d1, d2);
      EXPECT_NEAR(n0, b0[i], 1e-14) << "i " << i << " t " << t;
      EXPECT_NEAR(d1, b1[i], 1e-13) << "i " << i << " t " << t;
      EXPECT_NEAR(d2, b2[i], 1e-13) << "i " << i << " t " << t;
    }
  }
}

// The Greville abscissae as coefficients reproduce s(t) = t, so the spline is
// the identity and so is its inverse
TEST(test_bspline, LinearPrecision) {
  const double u[] = {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0};
  Vec<double, 7> c;
  Mat<double, 7, 2> m;
  for (int i = 0; i < 7; i++) {
    c(i) = (u[i + 1] + u[i + 2] + u[i + 3]) / 3.0;
    m(i, 0) = 0.0;
    m(i, 1) = c(i);
  }
  VecSamples<Vec<double, 7>> coef(c);

  const double ts[] = {-0.2, 0.0, 0.1, 0.25, 0.6, 0.9, 1.0, 1.3};
  for (double t : ts) {
    double d1, d2;
    EXPECT_NEAR(BSplineValue<4>(coef, t, d1, d2), t, 1e-14) << "t " << t;
    EXPECT_NEAR(d1, 1.0, 1e-13) << "t " << t;
    EXPECT_NEAR(d2, 0.0, 1e-12) << "t " << t;
    EXPECT_NEAR(bspline<4>(c, t), t, 1e-14) << "t " << t;
    EXPECT_NEAR(bspline<4>(m, 1, t), t, 1e-14) << "t " << t;
  }

  const double xs[] = {0.0, 0.05, 0.3, 0.5, 0.8, 1.0};
  for (double x : xs) {
    double d1, d2;
    EXPECT_NEAR(BSplineInverseValue<4>(coef, x, d1, d2), x, 1e-14)
        << "x " << x;
    EXPECT_NEAR(d1, 1.0, 1e-12) << "x " << x;
    EXPECT_NEAR(d2, 0.0, 1e-10) << "x " << x;
    EXPECT_NEAR(bspline_inverse<4>(c, x), x, 1e-14) << "x " << x;
    EXPECT_NEAR(bspline_inverse<4>(m, 1, x), x, 1e-14) << "x " << x;
  }
}

// The inverse of a nonlinear increasing spline maps back to x, with the
// derivatives of the inverse against central differences
TEST(test_bspline, InverseRoundTrip) {
  const double v[7] = {0.0, 0.05, 0.3, 0.45, 0.9, 1.4, 2.0};
  Vec<double, 7> c;
  for (int i = 0; i < 7; i++) {
    c(i) = v[i];
  }
  VecSamples<Vec<double, 7>> coef(c);

  const double xs[] = {0.0, 0.01, 0.2, 0.45, 0.87, 1.5, 1.99, 2.0};
  for (double x : xs) {
    double d1, d2, s1, s2;
    const double t = BSplineInverseValue<4>(coef, x, d1, d2);
    EXPECT_GE(t, 0.0) << "x " << x;
    EXPECT_LE(t, 1.0) << "x " << x;
    EXPECT_NEAR(BSplineValue<4>(coef, t, s1, s2), x, 1e-14) << "x " << x;
    EXPECT_NEAR(d1 * s1, 1.0, 1e-13) << "x " << x;
  }

  // Interior points, where both one-sided steps stay in the range
  const double xi[] = {0.01, 0.2, 0.45, 0.87, 1.5, 1.99};
  for (double x : xi) {
    double d1, d2, p1, p2, m1, m2;
    BSplineInverseValue<4>(coef, x, d1, d2);
    const double h = 1e-6;
    const double tp = BSplineInverseValue<4>(coef, x + h, p1, p2);
    const double tm = BSplineInverseValue<4>(coef, x - h, m1, m2);
    EXPECT_NEAR(d1, (tp - tm) / (2.0 * h), Tol(d1, 1e-7)) << "x " << x;
    EXPECT_NEAR(d2, (p1 - m1) / (2.0 * h), Tol(d2, 1e-5)) << "x " << x;
  }
}

// Outside the range of the coefficients the inverse holds at an end with zero
// derivatives, in the plain function and through reverse mode
TEST(test_bspline, InverseOutsideRange) {
  const double v[7] = {0.0, 0.05, 0.3, 0.45, 0.9, 1.4, 2.0};
  Vec<double, 7> c;
  for (int i = 0; i < 7; i++) {
    c(i) = v[i];
  }
  VecSamples<Vec<double, 7>> coef(c);

  double d1, d2;
  EXPECT_EQ(BSplineInverseValue<4>(coef, -0.5, d1, d2), 0.0);
  EXPECT_EQ(d1, 0.0);
  EXPECT_EQ(d2, 0.0);
  EXPECT_EQ(BSplineInverseValue<4>(coef, 2.5, d1, d2), 1.0);
  EXPECT_EQ(d1, 0.0);
  EXPECT_EQ(d2, 0.0);

  const double xs[] = {-0.5, 2.5};
  const double ends[] = {0.0, 1.0};
  for (int k = 0; k < 2; k++) {
    ADObj<double> x, f;
    x.value() = xs[k];
    auto stack = MakeStack(Eval(bspline_inverse<4>(c, x), f));
    f.bvalue() = 1.0;
    stack.reverse();
    EXPECT_EQ(f.value(), ends[k]);
    EXPECT_EQ(x.bvalue(), 0.0);
  }
}

/*
  Splines of vector and matrix tables and of references to them inside an
  expression, with arguments inside the unit interval and past both of its
  ends, a basis function and an inverse
*/
class BSplineExpr : public ::testing::Test {
 protected:
  BSplineExpr() : vec_ref(vec.get_data()), mat_ref(mat.get_data()) {
    const double v[7] = {0.3, -1.2, 0.8, 2.5, -0.4, 0.6, 1.1};
    const double w[7] = {0.0, 0.05, 0.3, 0.45, 0.9, 1.4, 2.0};
    for (int i = 0; i < 7; i++) {
      vec(i) = v[i];
      inv(i) = w[i];
    }
    for (int i = 0; i < 6; i++) {
      mat(i, 0) = 2.0 * v[i];
      mat(i, 1) = v[i] * v[i] - 0.5;
    }
  }

  // At (a0, b0) the arguments are 0.2146, 0.348, 1.15, -0.12, 0.37 and 0.87,
  // each away from the knots
  template <class A, class B>
  auto func(A& a, B& b) const {
    return bspline<4>(vec, a * b) * b + bspline<3>(mat, 1, 0.4 * a + 0.2) +
           bspline<4>(vec_ref, a + b + 0.2) * a +
           bspline<3>(mat_ref, 0, b - 0.7) + bspline_basis<4, 7>(3, a) * b +
           bspline_inverse<4>(inv, a + 0.5) * b;
  }

  double value(double a, double b) const { return func(a, b); }

  void gradient(double a0, double b0, double& ga, double& gb) const {
    ADObj<double> a, b, f;
    a.value() = a0;
    b.value() = b0;
    auto stack = MakeStack(Eval(func(a, b), f));
    f.bvalue() = 1.0;
    stack.reverse();
    ga = a.bvalue();
    gb = b.bvalue();
  }

  const double a0 = 0.37, b0 = 0.58;
  const double pa = 0.6, pb = -1.3;

  Vec<double, 7> vec, inv;
  Mat<double, 6, 2> mat;
  VecRef<double, 7> vec_ref;
  MatRef<double, 6, 2> mat_ref;
};

TEST_F(BSplineExpr, ReverseMatchesDifferences) {
  double ga, gb;
  gradient(a0, b0, ga, gb);

  const double h = 1e-6;
  const double fa = (value(a0 + h, b0) - value(a0 - h, b0)) / (2.0 * h);
  const double fb = (value(a0, b0 + h) - value(a0, b0 - h)) / (2.0 * h);
  EXPECT_NEAR(ga, fa, Tol(fa, 1e-7));
  EXPECT_NEAR(gb, fb, Tol(fb, 1e-7));
}

TEST_F(BSplineExpr, ForwardMatchesReverse) {
  double ga, gb;
  gradient(a0, b0, ga, gb);

  ADObj<double> a, b, f;
  a.value() = a0;
  b.value() = b0;
  auto stack = MakeStack(Eval(func(a, b), f));
  a.bvalue() = pa;
  b.bvalue() = pb;
  stack.forward();

  const double directional = ga * pa + gb * pb;
  EXPECT_NEAR(f.value(), value(a0, b0), 1e-14);
  EXPECT_NEAR(f.bvalue(), directional, Tol(directional, 1e-13));
}

// The Hessian product with a second order seed, against central differences
// of the reverse mode gradient
TEST_F(BSplineExpr, HessianProductMatchesDifferences) {
  double ga, gb;
  gradient(a0, b0, ga, gb);

  const double hseed = 0.3;
  A2DObj<double> a, b, f;
  a.value() = a0;
  b.value() = b0;
  a.pvalue() = pa;
  b.pvalue() = pb;
  auto stack = MakeStack(Eval(func(a, b), f));
  f.bvalue() = 1.0;
  f.hvalue() = hseed;
  stack.hproduct();

  EXPECT_NEAR(f.value(), value(a0, b0), 1e-14);
  EXPECT_NEAR(a.bvalue(), ga, Tol(ga, 1e-14));
  EXPECT_NEAR(b.bvalue(), gb, Tol(gb, 1e-14));

  const double eps = 1e-6;
  double gap, gbp, gam, gbm;
  gradient(a0 + eps * pa, b0 + eps * pb, gap, gbp);
  gradient(a0 - eps * pa, b0 - eps * pb, gam, gbm);
  const double ha = (gap - gam) / (2.0 * eps) + hseed * ga;
  const double hb = (gbp - gbm) / (2.0 * eps) + hseed * gb;
  EXPECT_NEAR(a.hvalue(), ha, Tol(ha, 1e-7));
  EXPECT_NEAR(b.hvalue(), hb, Tol(hb, 1e-7));
}

// ADScalar forward mode in two directions gives the reverse mode gradient
TEST_F(BSplineExpr, ADScalarMatchesReverse) {
  double ga, gb;
  gradient(a0, b0, ga, gb);

  ADScalar<double, 2> a(a0), b(b0);
  a.deriv[0] = 1.0;
  b.deriv[1] = 1.0;
  ADScalar<double, 2> f = func(a, b);

  EXPECT_NEAR(f.value, value(a0, b0), 1e-14);
  EXPECT_NEAR(f.deriv[0], ga, Tol(ga, 1e-13));
  EXPECT_NEAR(f.deriv[1], gb, Tol(gb, 1e-13));
}

/*
  Coefficients and arguments that carry ADScalar derivatives, as the
  derivatives with respect to data in amigo give them. The spline is linear in
  its coefficients, so its derivative with respect to coefficient j is basis
  function j.
*/
class BSplineCoefficients : public ::testing::Test {
 protected:
  BSplineCoefficients() {
    const double v[7] = {0.3, -1.2, 0.8, 2.5, -0.4, 0.6, 1.1};
    const double w[7] = {0.0, 0.05, 0.3, 0.45, 0.9, 1.4, 2.0};
    for (int i = 0; i < 7; i++) {
      vec(i) = v[i];
      inv(i) = w[i];
    }
  }

  // Coefficients with coefficient j seeded
  void seed(int j, Vec<ADScalar<double, 1>, 7>& c,
            Vec<ADScalar<double, 1>, 7>& ci) const {
    for (int i = 0; i < 7; i++) {
      c(i) = vec(i);
      ci(i) = inv(i);
    }
    c(j).deriv[0] = 1.0;
    ci(j).deriv[0] = 1.0;
  }

  // Slope of the inverse with coefficient j of the coordinate spline shifted
  double inverse_slope(int j, double shift) const {
    Vec<double, 7> c(inv);
    c(j) += shift;
    VecSamples<Vec<double, 7>> coef(c);
    double d1, d2;
    BSplineInverseValue<4>(coef, x0, d1, d2);
    return d1;
  }

  // Root of the inverse with coefficient j of the coordinate spline shifted
  double inverse_root(int j, double shift) const {
    Vec<double, 7> c(inv);
    c(j) += shift;
    return bspline_inverse<4>(c, x0);
  }

  const double t0 = 0.37, x0 = 0.87, h = 1e-6;

  Vec<double, 7> vec, inv;
};

// Forward mode gives the basis function and the motion of the root
TEST_F(BSplineCoefficients, ForwardMatchesBasisAndDifferences) {
  for (int j = 0; j < 7; j++) {
    Vec<ADScalar<double, 1>, 7> c, ci;
    seed(j, c, ci);

    double b1, b2;
    const double basis = BSplineBasisValue<4, 7>(j, t0, b1, b2);
    ADScalar<double, 1> t(t0), x(x0);
    EXPECT_NEAR(bspline<4>(c, t).deriv[0], basis, 1e-14) << "j " << j;
    EXPECT_NEAR(bspline<4>(c, t0).deriv[0], basis, 1e-14) << "j " << j;

    const double fd = (inverse_root(j, h) - inverse_root(j, -h)) / (2.0 * h);
    EXPECT_NEAR(bspline_inverse<4>(ci, x).deriv[0], fd, Tol(fd, 1e-7))
        << "j " << j;
  }
}

// Reverse mode over ADScalar values gives the derivatives of the slopes with
// respect to each coefficient
TEST_F(BSplineCoefficients, ReverseOverForwardMatchesDifferences) {
  for (int j = 0; j < 7; j++) {
    Vec<ADScalar<double, 1>, 7> c, ci;
    seed(j, c, ci);

    double b1, b2;
    const double basis = BSplineBasisValue<4, 7>(j, t0, b1, b2);
    ADObj<ADScalar<double, 1>> t, f;
    t.value() = t0;
    auto stack = MakeStack(Eval(bspline<4>(c, t), f));
    f.bvalue() = 1.0;
    stack.reverse();
    EXPECT_NEAR(f.value().deriv[0], basis, 1e-14) << "j " << j;
    EXPECT_NEAR(t.bvalue().deriv[0], b1, Tol(b1, 1e-13)) << "j " << j;

    ADObj<ADScalar<double, 1>> x, g;
    x.value() = x0;
    auto istack = MakeStack(Eval(bspline_inverse<4>(ci, x), g));
    g.bvalue() = 1.0;
    istack.reverse();
    const double fd = (inverse_slope(j, h) - inverse_slope(j, -h)) / (2.0 * h);
    EXPECT_NEAR(x.bvalue().deriv[0], fd, Tol(fd, 1e-6)) << "j " << j;
  }
}

// An argument that carries a derivative moves the value along the slope and
// the slope along the curvature
TEST_F(BSplineCoefficients, ReverseOverForwardArgument) {
  VecSamples<Vec<double, 7>> coef(vec);
  double s1, s2;
  const double s0 = BSplineValue<4>(coef, t0, s1, s2);

  ADObj<ADScalar<double, 1>> t, f;
  t.value() = t0;
  t.value().deriv[0] = 1.0;
  auto stack = MakeStack(Eval(bspline<4>(vec, t), f));
  f.bvalue() = 1.0;
  stack.reverse();
  EXPECT_NEAR(f.value().value, s0, 1e-14);
  EXPECT_NEAR(f.value().deriv[0], s1, Tol(s1, 1e-14));
  EXPECT_NEAR(t.bvalue().value, s1, Tol(s1, 1e-14));
  EXPECT_NEAR(t.bvalue().deriv[0], s2, Tol(s2, 1e-13));

  double b1, b2;
  const double basis = BSplineBasisValue<4, 7>(3, t0, b1, b2);
  ADObj<ADScalar<double, 1>> u, g;
  u.value() = t0;
  u.value().deriv[0] = 1.0;
  auto bstack = MakeStack(Eval(bspline_basis<4, 7>(3, u), g));
  g.bvalue() = 1.0;
  bstack.reverse();
  EXPECT_NEAR(g.value().value, basis, 1e-14);
  EXPECT_NEAR(g.value().deriv[0], b1, Tol(b1, 1e-14));
  EXPECT_NEAR(u.bvalue().deriv[0], b2, Tol(b2, 1e-13));
}
