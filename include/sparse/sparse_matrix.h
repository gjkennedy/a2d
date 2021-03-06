#ifndef A2D_SPARSE_MATRIX_H
#define A2D_SPARSE_MATRIX_H

#include "multiarray.h"

namespace A2D {

/**
 * @brief Block CSR matrix
 *
 * @tparam I index type
 * @tparam T data type
 * @tparam M number of rows for each block
 * @tparam N number of columns for each block
 */
template <typename I, typename T, index_t M, index_t N>
class BSRMat {
 public:
  /**
   * @brief Constructor
   *
   * @tparam VecType a vector type
   * @param nbrows number of rows of blocks
   * @param nbcols number of columns of blocks
   * @param nnz number of non-zero blocks, note that nnz(mat) = nnz * M * N
   * @param _rowp vector of row pointers
   * @param _cols vector of column indices
   */
  template <class VecType>
  BSRMat(index_t nbrows, index_t nbcols, index_t nnz, const VecType &_rowp,
         const VecType &_cols)
      : nbrows(nbrows), nbcols(nbcols), nnz(nnz), Avals(CLayout<M, N>(nnz)) {
    data_owner = true;
    rowp = new I[nbrows + 1];
    cols = new I[nnz];

    for (I i = 0; i < nbrows + 1; i++) {
      rowp[i] = _rowp[i];
    }

    for (I i = 0; i < nnz; i++) {
      cols[i] = _cols[i];
    }

    // Set the diagonal to NULL until factorization
    diag = NULL;

    // Set the permutation array and its inverse to NULL
    perm = NULL;
    iperm = NULL;

    // Set the color count to NULL
    num_colors = 0;
    color_count = NULL;
  }
  BSRMat(const BSRMat &src)
      : nbrows(src.nbrows), nbcols(src.nbcols), nnz(src.nnz), Avals(src.Avals) {
    data_owner = false;
  }
  ~BSRMat() {
    if (data_owner) {
      delete[] rowp;
      delete[] cols;
      if (diag) {
        delete[] diag;
      }
      if (perm) {
        delete[] perm;
      }
      if (iperm) {
        delete[] iperm;
      }
      if (color_count) {
        delete[] color_count;
      }
    }
  }

  // Zero the entries of the matrix
  void zero() { Avals.zero(); }

  /**
   * @brief Find the address of the column index given block indices (row, col)
   *
   * @param row block row index
   * @param col block column index
   * @return I* address of the block column index
   */
  I *find_column_index(I row, I col) {
    I jp_start = rowp[row];
    I jp_end = rowp[row + 1];

    for (I jp = jp_start; jp < jp_end; jp++) {
      if (cols[jp] == col) {
        return &cols[jp];
      }
    }

    return nullptr;
  }

  // Number of block rows and block columns
  index_t nbrows, nbcols;

  // Number of non-zero blocks
  index_t nnz;  // = rowp[nbrows];

  // rowp and cols array
  I *rowp;  // length: nbrows + 1
  I *cols;  // length: nnz = rowp[nbrows]

  // Pointer to the diagonal block
  I *diag;  // length: nbrows

  // permutation perm[new var] = old var
  I *perm;

  // Inverse permutation iperm[old var] = new var
  I *iperm;

  // When coloring is used, its ordering is stored in the permutation array
  I num_colors;    // Number of colors
  I *color_count;  // Number of nodes with this color

  // MultiArray data - length: nnz
  MultiArray<T, CLayout<M, N>> Avals;

 private:
  bool data_owner;
};

}  // namespace A2D

#endif  // A2D_SPARSE_MATRIX_H
