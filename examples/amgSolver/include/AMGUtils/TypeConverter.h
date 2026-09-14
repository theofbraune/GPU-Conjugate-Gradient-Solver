
#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
// ============================================================
//  AMGCL includes
// ============================================================
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

#include <algorithm>
#include <vector>

namespace AMGUtils {

// ============================================================
//  Helper: Eigen -> CRS conversion
// ============================================================

// Templated on both scalar type and Eigen storage order, so it accepts a
// ColMajor or RowMajor sparse matrix of any (floating-point) scalar type.
// Internally always normalizes to RowMajor before extracting CRS arrays.
template <typename Scalar, int Options>
void eigen_to_crs(const Eigen::SparseMatrix<Scalar, Options>& A_in, ptrdiff_t& rows,
                   std::vector<ptrdiff_t>& ptr, std::vector<ptrdiff_t>& col,
                   std::vector<Scalar>& val) {
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

    // Enforce RowMajor + compressed format
    SpMat A = A_in;
    A.makeCompressed();

    rows = A.rows();

    ptr.resize(rows + 1);
    col.resize(A.nonZeros());
    val.resize(A.nonZeros());

    std::copy(A.outerIndexPtr(), A.outerIndexPtr() + rows + 1, ptr.begin());
    std::copy(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros(), col.begin());
    std::copy(A.valuePtr(), A.valuePtr() + A.nonZeros(), val.begin());
}

template <typename Scalar>
void eigen_to_std(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& e, std::vector<Scalar>& v) {
    v.assign(e.data(), e.data() + e.size());
}

template <typename Scalar>
void std_to_eigen(const std::vector<Scalar>& v, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& e) {
    e = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(v.data(), v.size());
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor>
crs_to_eigen(ptrdiff_t rows, ptrdiff_t cols, const std::vector<ptrdiff_t>& ptr,
             const std::vector<ptrdiff_t>& col, const std::vector<Scalar>& val) {
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

    SpMat A(rows, cols);

    // Reserve exact number of nonzeros
    A.reserve(val.size());

    for (ptrdiff_t i = 0; i < rows; ++i) {
        for (ptrdiff_t k = ptr[i]; k < ptr[i + 1]; ++k) {
            A.insert(i, col[k]) = val[k];
        }
    }

    A.makeCompressed();
    return A;
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor>
crs_to_eigen_triplets(const amgcl::backend::crs<Scalar>& A) {
    using SpMat   = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using Triplet = Eigen::Triplet<Scalar>;

    std::vector<Triplet> triplets;
    triplets.reserve(A.nnz);

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(A.nrows); ++i) {
        for (ptrdiff_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j) {
            triplets.emplace_back(
                static_cast<int>(i),
                static_cast<int>(A.col[j]),
                A.val[j]
            );
        }
    }

    SpMat M(A.nrows, A.ncols);
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

} // namespace AMGUtils
