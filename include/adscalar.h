#ifndef A2D_ADSCALAR_H
#define A2D_ADSCALAR_H

#include <complex>
#include <type_traits>

#include "a2ddefs.h"
#include "a2dmath.h"

namespace A2D {

template <class T, int N>
class ADScalar {
 public:
  using type = T;

  A2D_FUNCTION ADScalar() {}

  // Value constructor (sets a value, zeros derivatives)
  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION ADScalar(const R value) : value(value), deriv{0.0} {}

  // Value and derivative constructor (sets both, and works regardless of the
  // type T)
  A2D_FUNCTION ADScalar(const T& value, const T d[]) : value(value) {
    for (int i = 0; i < N; i++) {
      deriv[i] = d[i];
    }
  }

  // Scalar-only version for ADScalar constructor for a common occasion
  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  ADScalar(const R val, const T d[]) : ADScalar(T(val), d) {}

  // Copy constructor (sets value and derivatives from another ADScalar)
  A2D_FUNCTION ADScalar(const ADScalar<T, N>& r) : value(r.value) {
    for (int i = 0; i < N; i++) {
      deriv[i] = r.deriv[i];
    }
  }

  // Conversion constructor
  //  - disabled when R == T, which forces copy constructor for same type copies
  //  - enabled when conversion or lifting makes sense
  template <typename R, std::enable_if_t<!std::is_same_v<R, T> &&
                                             (std::is_convertible_v<R, T> ||
                                              is_adscalar_v<T>),
                                         int> = 0>
  explicit ADScalar(const ADScalar<R, N>& r) {
    if constexpr (is_adscalar_v<T>) {
      // Lifting: occurs if T is an ADScalar<...> type
      // copy the entire ADScalar r into 'value' for the current ADScalar
      value = r;  // invokes inner ADScalar's conversion/copy
      for (int i = 0; i < N; ++i) {
        deriv[i] = T(0.0);  // outer derivatives zeroed
      }
    } else {
      // Componentwise: T is a plain scalar-like type (e.g. double)
      value = static_cast<T>(r.value);
      for (int i = 0; i < N; ++i) {
        deriv[i] = static_cast<T>(r.deriv[i]);
      }
    }
  }

  // Assignment operator
  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline ADScalar<T, N>& operator=(const R& r) {
    value = r;
    for (int i = 0; i < N; i++) {
      deriv[i] = 0.0;
    }
    return *this;
  }

  // Comparison operators
  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline bool operator<(const R& rhs) const {
    return value < rhs;
  }

  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline bool operator<=(const R& rhs) const {
    return value <= rhs;
  }

  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline bool operator>(const R& rhs) const {
    return value > rhs;
  }

  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline bool operator>=(const R& rhs) const {
    return value >= rhs;
  }

  template <typename R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline bool operator!=(const R& rhs) const {
    return value != rhs;
  }

  template <typename X, int M>
  A2D_FUNCTION inline bool operator<(const ADScalar<X, M>& rhs) const {
    return value < rhs.value;
  }

  template <typename X, int M>
  A2D_FUNCTION inline bool operator<=(const ADScalar<X, M>& rhs) const {
    return value <= rhs.value;
  }

  template <typename X, int M>
  A2D_FUNCTION inline bool operator>(const ADScalar<X, M>& rhs) const {
    return value > rhs.value;
  }

  template <typename X, int M>
  A2D_FUNCTION inline bool operator>=(const ADScalar<X, M>& rhs) const {
    return value >= rhs.value;
  }

  // Operator +=, -=, *=, /=
  A2D_FUNCTION inline ADScalar<T, N>& operator+=(const ADScalar<T, N>& r) {
    value += r.value;
    for (int i = 0; i < N; i++) {
      deriv[i] += r.deriv[i];
    }
    return *this;
  }

  template <class R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline ADScalar<T, N>& operator+=(const R& r) {
    value += r;
    return *this;
  }

  A2D_FUNCTION inline ADScalar<T, N>& operator-=(const ADScalar<T, N>& r) {
    value -= r.value;
    for (int i = 0; i < N; i++) {
      deriv[i] -= r.deriv[i];
    }
    return *this;
  }

  template <class R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline ADScalar<T, N>& operator-=(const R& r) {
    value -= r;
    return *this;
  }

  A2D_FUNCTION inline ADScalar<T, N>& operator*=(const ADScalar<T, N>& r) {
    for (int i = 0; i < N; i++) {
      deriv[i] = r.value * deriv[i] + value * r.deriv[i];
    }
    value *= r.value;
    return *this;
  }

  template <class R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline ADScalar<T, N>& operator*=(const R& r) {
    value *= r;
    for (int i = 0; i < N; i++) {
      deriv[i] = r * deriv[i];
    }
    return *this;
  }

  A2D_FUNCTION inline ADScalar<T, N>& operator/=(const ADScalar<T, N>& r) {
    T inv = T(1.0) / r.value;
    T inv2 = value * inv * inv;
    value *= inv;
    for (int i = 0; i < N; i++) {
      deriv[i] = inv * deriv[i] - inv2 * r.deriv[i];
    }
    return *this;
  }

  template <class R,
            std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
  A2D_FUNCTION inline ADScalar<T, N> operator/=(const R& r) {
    T inv = T(1.0) / r;
    value *= inv;
    for (int i = 0; i < N; i++) {
      deriv[i] = inv * deriv[i];
    }
    return *this;
  }

  A2D_FUNCTION inline ADScalar<T, N> operator-() const {
    T negderivs[N];
    for (int i = 0; i < N; i++) {
      negderivs[i] = -deriv[i];
    }
    return ADScalar<T, N>(-value, negderivs);
  }

  //  private:
  T value;
  T deriv[N];
};

// The real part of an ADScalar is the real part of its value
template <class X, int M>
A2D_FUNCTION inline double RealPart(const ADScalar<X, M>& a) {
  return RealPart(a.value);
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> operator+(const ADScalar<X, M>& l,
                                             const ADScalar<X, M>& r) {
  ADScalar<X, M> out(l.value + r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = l.deriv[i] + r.deriv[i];
  }
  return out;
}

template <class X, int M, class L,
          std::enable_if_t<is_passive_scalar_or_integral_v<L>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator+(const L& l,
                                             const ADScalar<X, M>& r) {
  return ADScalar<X, M>(r.value + l, r.deriv);
}

template <class X, int M, class R,
          std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator+(const ADScalar<X, M>& l,
                                             const R& r) {
  return ADScalar<X, M>(l.value + r, l.deriv);
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> operator-(const ADScalar<X, M>& l,
                                             const ADScalar<X, M>& r) {
  ADScalar<X, M> out(l.value - r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = l.deriv[i] - r.deriv[i];
  }
  return out;
}

template <class X, int M, class L,
          std::enable_if_t<is_passive_scalar_or_integral_v<L>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator-(const L& l,
                                             const ADScalar<X, M>& r) {
  ADScalar<X, M> out(l - r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = -r.deriv[i];
  }
  return out;
}

template <class X, int M, class R,
          std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator-(const ADScalar<X, M>& l,
                                             const R& r) {
  return ADScalar<X, M>(l.value - r, l.deriv);
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> operator*(const ADScalar<X, M>& l,
                                             const ADScalar<X, M>& r) {
  ADScalar<X, M> out(l.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = r.value * l.deriv[i] + r.deriv[i] * l.value;
  }
  return out;
}

template <class X, int M, class L,
          std::enable_if_t<is_passive_scalar_or_integral_v<L>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator*(const L& l,
                                             const ADScalar<X, M>& r) {
  ADScalar<X, M> out(l * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = r.deriv[i] * l;
  }
  return out;
}

template <class X, int M, class R,
          std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator*(const ADScalar<X, M>& l,
                                             const R& r) {
  ADScalar<X, M> out(l.value * r);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = l.deriv[i] * r;
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> operator/(const ADScalar<X, M>& l,
                                             const ADScalar<X, M>& r) {
  X inv = X(1.0) / r.value;
  X inv2 = l.value * inv * inv;
  ADScalar<X, M> out(inv * l.value);

  for (int i = 0; i < M; i++) {
    out.deriv[i] = inv * l.deriv[i] - inv2 * r.deriv[i];
  }
  return out;
}

template <class X, int M, class L,
          std::enable_if_t<is_passive_scalar_or_integral_v<L>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator/(const L& l,
                                             const ADScalar<X, M>& r) {
  X inv = X(1.0) / r.value;
  X inv2 = l * inv * inv;
  ADScalar<X, M> out(inv * l);

  for (int i = 0; i < M; i++) {
    out.deriv[i] = -inv2 * r.deriv[i];
  }
  return out;
}

template <class X, int M, class R,
          std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> operator/(const ADScalar<X, M>& l,
                                             const R& r) {
  X inv = X(1.0) / r;
  ADScalar<X, M> out(inv * l.value);

  for (int i = 0; i < M; i++) {
    out.deriv[i] = inv * l.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> fabs(const ADScalar<X, M>& r) {
  X scalar = X(1.0);
  if (r.value < 0.0) {
    scalar = X(-1.0);
  }
  ADScalar<X, M> out(A2D::fabs(r.value));
  for (int i = 0; i < M; i++) {
    out.deriv[i] = scalar * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> sqrt(const ADScalar<X, M>& r) {
  X value = A2D::sqrt(r.value);
  ADScalar<X, M> out(value);
  X inv = X(0.5) / value;
  for (int i = 0; i < M; i++) {
    out.deriv[i] = inv * r.deriv[i];
  }
  return out;
}

template <class X, int M, class R,
          std::enable_if_t<is_passive_scalar_or_integral_v<R>, int> = 0>
A2D_FUNCTION inline ADScalar<X, M> pow(const ADScalar<X, M>& r,
                                       const R& exponent) {
  X value = A2D::pow(r.value, exponent);
  ADScalar<X, M> out(value);
  X inv = exponent * value / r.value;
  for (int i = 0; i < M; i++) {
    out.deriv[i] = inv * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> exp(const ADScalar<X, M>& r) {
  X value = A2D::exp(r.value);
  ADScalar<X, M> out(value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = value * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> log(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::log(r.value));
  X inv = X(1.0) / r.value;
  for (int i = 0; i < M; i++) {
    out.deriv[i] = inv * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> sin(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::sin(r.value));
  X d = A2D::cos(r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> cos(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::cos(r.value));
  X d = -A2D::sin(r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> tan(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::tan(r.value));
  X t = A2D::cos(r.value);
  X d = X(1.0) / (t * t);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> asin(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::asin(r.value));
  X d = X(1.0) / A2D::sqrt(X(1.0) - r.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> acos(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::acos(r.value));
  X d = X(-1.0) / A2D::sqrt(X(1.0) - r.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> atan(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::atan(r.value));
  X d = X(1.0) / A2D::sqrt(X(1.0) + r.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> atan2(const ADScalar<X, M>& y,
                                         const ADScalar<X, M>& x) {
  ADScalar<X, M> out(A2D::atan2(y.value, x.value));
  X denom = x.value * x.value + y.value * y.value;
  X dx = -y.value / denom;
  X dy = x.value / denom;
  for (int i = 0; i < M; i++) {
    out.deriv[i] = dx * x.deriv[0] + dy * y.deriv[0];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> sinh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::sinh(r.value));
  X d = A2D::cosh(r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> cosh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::cosh(r.value));
  X d = A2D::sinh(r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> tanh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::tanh(r.value));
  const X t = A2D::cosh(r.value);
  X d = X(1.0) / (t * t);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> asinh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::asinh(r.value));
  X d = X(1.0) / A2D::sqrt(X(1.0) + r.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> acosh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::acosh(r.value));
  X d = X(1.0) / A2D::sqrt(r.value * r.value - X(1.0));
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

template <class X, int M>
A2D_FUNCTION inline ADScalar<X, M> atanh(const ADScalar<X, M>& r) {
  ADScalar<X, M> out(A2D::atanh(r.value));
  X d = X(1.0) / (X(1.0) - r.value * r.value);
  for (int i = 0; i < M; i++) {
    out.deriv[i] = d * r.deriv[i];
  }
  return out;
}

}  // namespace A2D

#endif  // A2D_ADSCALAR_H
