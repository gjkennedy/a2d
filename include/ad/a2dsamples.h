#ifndef A2D_SAMPLES_H
#define A2D_SAMPLES_H

#include "../a2ddefs.h"
#include "../a2dobj.h"

namespace A2D {

/*
  Read access to a table of samples or coefficients

  The table is a Vec or VecRef, or one column of a Mat or MatRef. The accessor
  keeps a reference to the table, so the table must outlive any expression
  built on it.
*/

// The samples of a vector table
template <class Vtype>
class VecSamples {
 public:
  using type = typename Vtype::type;
  static constexpr int size = get_vec_size<Vtype>::size;
  A2D_FUNCTION VecSamples(const Vtype& table) : table(table) {}
  A2D_FUNCTION auto operator()(int i) const { return table(i); }

 private:
  const Vtype& table;
};

// The samples in one column of a matrix table
template <class Mtype>
class MatColumnSamples {
 public:
  using type = typename Mtype::type;
  static constexpr int size = get_matrix_rows<Mtype>::size;
  A2D_FUNCTION MatColumnSamples(const Mtype& table, int column)
      : table(table), column(column) {}
  A2D_FUNCTION auto operator()(int i) const { return table(i, column); }

 private:
  const Mtype& table;
  int column;
};

}  // namespace A2D

#endif  // A2D_SAMPLES_H
