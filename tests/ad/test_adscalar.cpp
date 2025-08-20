#include <functional>
#include <iostream>
#include <vector>

#include "a2dcore.h"

using namespace A2D;

template <typename T> class ADScalarTest : public A2D::Test::A2DTest<T, T, T> {
public:
  using Input = VarTuple<T, T>;
  using Output = VarTuple<T, T>;

  std::string name() override { return "ADScalarTest"; }

  // Evaluate f(x) = x^2 using ADScalar
  Output eval(const Input &x) override {
    T val;
    x.get_values(val);
    ADScalar<T, 1> ad_x(val);
    ADScalar<T, 1> ad_y = ad_x * ad_x;
    return MakeVarTuple<T>(ad_y.value);
  }

  // Compute the derivative: df/dx = 2x using ADScalar
  void deriv(const Output &seed, const Input &x, Input &g) override {
    T val;
    x.get_values(val);
    T dy;
    seed.get_values(dy);
    ADScalar<T, 1> ad_x(val);
    ad_x.deriv[0] = dy;
    ADScalar<T, 1> ad_y = ad_x * ad_x;
    g.set_values(ad_y.deriv[0]);
  }

  // Compute the second derivative: d^2f/dx^2 = 2 using ADScalar
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) override {
    T val, pval, dy, ddy;
    x.get_values(val);
    p.get_values(pval);
    seed.get_values(dy);
    hval.get_values(ddy);
    ADScalar<T, 1> ad_x(val);
    ad_x.deriv[0] = pval;
    ADScalar<T, 1> ad_y = ad_x * ad_x;
    // Second derivative: d^2f/dx^2 * p = 2 * pval
    h.set_values(2.0 * pval * dy + 2.0 * ddy * val);
  }
};

template <typename T>
class ADScalarNestedTest : public A2D::Test::A2DTest<T, T, T> {
public:
  using Input = VarTuple<T, T>;
  using Output = VarTuple<T, T>;

  std::string name() override { return "ADScalarNestedTest"; }

  // Evaluate f(x) = x^2 using nested ADScalar
  Output eval(const Input &x) override {
    T val;
    x.get_values(val);
    ADScalar<T, 1> ad_x(val);
    ADScalar<ADScalar<T, 1>, 1> nested_x(ad_x);
    ADScalar<ADScalar<T, 1>, 1> nested_y = nested_x * nested_x;
    return MakeVarTuple<T>(nested_y.value.value); // Unwrap both layers
  }

  // Compute the derivative: df/dx = 2x using nested ADScalar
  void deriv(const Output &seed, const Input &x, Input &g) override {

    T val;
    x.get_values(val);
    T dy;
    seed.get_values(dy);

    ADScalar<T, 1> ad_x(val);
    ad_x.deriv[0] = dy;

    ADScalar<ADScalar<T, 1>, 1> nested_x(ad_x);

    ADScalar<ADScalar<T, 1>, 1> nested_y = nested_x * nested_x;

    g.set_values(
        nested_y.value.deriv[0]); // Unwrap derivative from inner ADScalar
  }

  // Compute the second derivative: d^2f/dx^2 = 2 using nested ADScalar
  void hprod(const Output &seed, const Output &hval, const Input &x,
             const Input &p, Input &h) override {
    T val, pval, dy, ddy;
    x.get_values(val);
    p.get_values(pval);
    seed.get_values(dy);
    hval.get_values(ddy);
    ADScalar<T, 1> ad_x(val);
    ad_x.deriv[0] = pval;
    ADScalar<ADScalar<T, 1>, 1> nested_x(ad_x);
    nested_x.deriv[0] = ADScalar<T, 1>(0.0);
    ADScalar<ADScalar<T, 1>, 1> nested_y = nested_x * nested_x;
    // Second derivative: d^2f/dx^2 * p = 2 * pval
    h.set_values(2.0 * pval * dy + 2.0 * ddy * val);
  }
};
bool ADScalarTestAll(bool component, bool write_output) {
  using Tc = A2D_complex_t<double>;

  bool passed = true;
  ADScalarTest<Tc> test;
  test.set_step_size(1e-30);

  ADScalarNestedTest<Tc> nested_test;
  nested_test.set_step_size(1e-30);

  passed = passed && Run(test, component, write_output);
  passed = passed && Run(nested_test, component, write_output);

  return passed;
}

int main(int argc, char *argv[]) {
  bool component = false;    // Default to a projection test
  bool write_output = false; // Don't write output;

  // Check for the write_output flag
  for (int i = 0; i < argc; i++) {
    std::string str(argv[i]);
    if (str.compare("--write_output") == 0) {
      write_output = true;
    }
    if (str.compare("--component") == 0) {
      component = true;
    }
  }

  bool passed = ADScalarTestAll(component, write_output);

  return passed ? 0 : 1;
}
