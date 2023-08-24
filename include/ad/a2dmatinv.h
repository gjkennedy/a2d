#ifndef A2D_MAT_INV_H
#define A2D_MAT_INV_H

#include "a2denum.h"
#include "a2dmat.h"
#include "a2dobjs.h"
#include "ad/core/a2dgemmcore.h"
#include "ad/core/a2dmatinvcore.h"

namespace A2D {

/*
  Compute Ainv = A^{-1} for small matrices

  dot{Ainv} = - A^{-1} * dot{A} * A^{-1}

  hA = A^{-T} * Ap^{T} * A^{-T} * Ainvb * A^{-T} +
      A^{-T} * Ainvb * A^{-T} * Ap^{T} * A^{-T} =
    = - (A^{-T} * Ap^{T} * Ab + Ab * Ap^{T} * A^{-T})
*/

template <typename T, int N>
KOKKOS_FUNCTION void MatInv(Mat<T, N, N>& A, Mat<T, N, N>& Ainv) {
  MatInvCore<T, N>(get_data(A), get_data(Ainv));
}

template <typename T, int N, ADorder order>
class MatInvExpr {
 private:
  using Atype = ADMatType<ADiffType::ACTIVE, order, Mat<T, N, N>>;

  static constexpr MatOp NORMAL = MatOp::NORMAL;
  static constexpr MatOp TRANSPOSE = MatOp::TRANSPOSE;

 public:
  KOKKOS_FUNCTION MatInvExpr(Atype& A, Atype& Ainv) : A(A), Ainv(Ainv) {
    MatInvCore<T, N>(get_data(A), get_data(Ainv));
  }

  template <ADorder forder>
  KOKKOS_FUNCTION void forward() {
    static_assert(
        !(order == ADorder::FIRST and forder == ADorder::SECOND),
        "Can't perform second order forward with first order objects");
    constexpr ADseed seed = conditional_value<ADseed, forder == ADorder::FIRST,
                                              ADseed::b, ADseed::p>::value;

    T temp[N * N];
    MatMatMultCore<T, N, N, N, N, N, N, NORMAL, NORMAL>(
        get_data(Ainv), GetSeed<seed>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, NORMAL, true>(
        T(-1.0), temp, get_data(Ainv), GetSeed<seed>::get_data(Ainv));
  }

  KOKKOS_FUNCTION void reverse() {
    T temp[N * N];
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, NORMAL>(
        get_data(Ainv), GetSeed<ADseed::b>::get_data(Ainv), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, true>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::b>::get_data(A));
  }

  KOKKOS_FUNCTION void hreverse() {
    static_assert(order == ADorder::SECOND,
                  "hreverse() can be called for only second order objects.");

    T temp[N * N];

    // - A^{-T} * Ap^{T} * Ab
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, TRANSPOSE>(
        get_data(Ainv), GetSeed<ADseed::p>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, NORMAL, true>(
        T(-1.0), temp, GetSeed<ADseed::b>::get_data(A),
        GetSeed<ADseed::h>::get_data(A));

    // - Ab * Ap^{T} * A^{-T}
    MatMatMultCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE>(
        GetSeed<ADseed::b>::get_data(A), GetSeed<ADseed::p>::get_data(A), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, true>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::h>::get_data(A));

    // - A^{-T} * Ainvh * A^{-T}
    MatMatMultCore<T, N, N, N, N, N, N, TRANSPOSE, NORMAL>(
        get_data(Ainv), GetSeed<ADseed::h>::get_data(Ainv), temp);
    MatMatMultScaleCore<T, N, N, N, N, N, N, NORMAL, TRANSPOSE, true>(
        T(-1.0), temp, get_data(Ainv), GetSeed<ADseed::h>::get_data(A));
  }

  Atype& A;
  Atype& Ainv;
};

template <typename T, int N>
KOKKOS_FUNCTION auto MatInv(ADMat<Mat<T, N, N>>& A,
                                ADMat<Mat<T, N, N>>& Ainv) {
  return MatInvExpr<T, N, ADorder::FIRST>(A, Ainv);
}

template <typename T, int N>
KOKKOS_FUNCTION auto MatInv(A2DMat<Mat<T, N, N>>& A,
                                A2DMat<Mat<T, N, N>>& Ainv) {
  return MatInvExpr<T, N, ADorder::SECOND>(A, Ainv);
}

}  // namespace A2D

#endif  // A2D_MAT_INV_H