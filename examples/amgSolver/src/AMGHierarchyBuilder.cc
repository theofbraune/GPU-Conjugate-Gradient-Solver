#include <AMGUtils/AMGHierarchyBuilder.h>
#include <AMGUtils/TypeConverter.h>

#include <amgcl/adapter/eigen.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

#include <functional>
#include <memory>

namespace MGBuilder {

namespace {

using Backend = amgcl::backend::builtin<float>;
using CrsMatrix = amgcl::backend::crs<float>;

Eigen::SparseMatrix<float, Eigen::RowMajor> toEigen(const CrsMatrix &m) {
  return AMGUtils::crs_to_eigen_triplets(m);
}

std::shared_ptr<CrsMatrix>
toBackend(const Eigen::SparseMatrix<float, Eigen::RowMajor> &A) {
  ptrdiff_t rows;
  std::vector<ptrdiff_t> ptr, col;
  std::vector<float> val;
  AMGUtils::eigen_to_crs(A, rows, ptr, col, val);
  return std::make_shared<CrsMatrix>(rows, rows, ptr, col, val);
}

// Shared implementation for all three "full hierarchy" builders below.
// `configureParams` fills in whatever per-level parameters the given
// Coarsening strategy needs; Ruge-Stuben's is a no-op beyond eps_strong
// since it doesn't use near-null-space information.
template <typename Coarsening>
gpuSolver::AMGHierarchy buildHierarchyImpl(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &M0,
    const Eigen::MatrixXf &V, int maxLevels, int minDofs,
    const std::function<void(typename Coarsening::params &, int nrows,
                             const Eigen::MatrixXf &)> &configureParams) {
  gpuSolver::AMGHierarchy H;

  std::shared_ptr<CrsMatrix> Abackend = toBackend(A0);

  H.A.push_back(A0);
  H.M.push_back(M0);

  const int nRBM = 6;
  Eigen::MatrixXf nullspace;
  int nullspaceCols = nRBM;
  buildRigidBodyModes(V, nullspace, nullspaceCols);
  H.nullspaces.push_back(nullspace);

  for (int level = 0; level < maxLevels; ++level) {
    const int nrowsCurrent = static_cast<int>(Abackend->nrows);
    if (nrowsCurrent <= minDofs) {
      break;
    }

    typename Coarsening::params prm;
    configureParams(prm, nrowsCurrent, nullspace);

    Coarsening coarsening(prm);
    auto [Pptr, Rptr] = coarsening.transfer_operators(*Abackend);

    Eigen::SparseMatrix<float, Eigen::RowMajor> P = toEigen(*Pptr);
    Eigen::SparseMatrix<float, Eigen::RowMajor> R = toEigen(*Rptr);
    H.P.push_back(P);
    H.R.push_back(R);

    std::shared_ptr<CrsMatrix> AcoarseBackend =
        coarsening.coarse_operator(*Abackend, *Pptr, *Rptr);
    Eigen::SparseMatrix<float, Eigen::RowMajor> Acoarse =
        toEigen(*AcoarseBackend);
    Eigen::SparseMatrix<float, Eigen::RowMajor> Mcoarse =
        (R * (H.M.back() * P).eval()).eval();

    H.A.push_back(Acoarse);
    H.M.push_back(Mcoarse);

    // Restrict and re-orthonormalize the near-null space: R is not
    // orthogonal, so its columns lose orthonormality after restriction,
    // and amgcl needs them orthonormal for the next tentative prolongator.
    nullspace = R * nullspace;
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(nullspace);
    nullspace =
        qr.householderQ() * Eigen::MatrixXf::Identity(nullspace.rows(), nRBM);

    H.nullspaces.push_back(nullspace);

    Abackend = AcoarseBackend;
  }

  return H;
}

} // namespace

gpuSolver::AMGHierarchy buildAmgclScalarSmoothedAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A0,
    int maxLevels,
    int minDofs)
{
    using Coarsening =
        amgcl::coarsening::smoothed_aggregation<Backend>;

    gpuSolver::AMGHierarchy H;

    std::shared_ptr<CrsMatrix> Abackend =
        toBackend(A0);

    H.A.push_back(A0);

    Eigen::VectorXf nullspace =
        Eigen::VectorXf::Ones(A0.rows());

    nullspace.normalize();

    for (int level = 0;
         level < maxLevels;
         ++level)
    {
        const int nRows =
            static_cast<int>(Abackend->nrows);

        if (nRows <= minDofs)
        {
            break;
        }

        Coarsening::params prm;

        prm.aggr.block_size = 1;
        prm.aggr.eps_strong = 0.1f;

        prm.nullspace.cols = 1;
        prm.nullspace.B.resize(
            static_cast<std::size_t>(nRows)
        );

        for (int row = 0;
             row < nRows;
             ++row)
        {
            prm.nullspace.B[row] =
                nullspace[row];
        }

        Coarsening coarsening(prm);

        auto [Pptr, Rptr] =
            coarsening.transfer_operators(
                *Abackend
            );

        gpuSolver::AMGHierarchy::Matrix P =
            toEigen(*Pptr);

        gpuSolver::AMGHierarchy::Matrix R =
            toEigen(*Rptr);

        H.P.push_back(P);
        H.R.push_back(R);

        std::shared_ptr<CrsMatrix> AcoarseBackend =
            coarsening.coarse_operator(
                *Abackend,
                *Pptr,
                *Rptr
            );

        gpuSolver::AMGHierarchy::Matrix Acoarse =
            toEigen(*AcoarseBackend);

        H.A.push_back(Acoarse);

        // Constant near-null mode on the next level.
        nullspace =
            R * nullspace;

        const float norm =
            nullspace.norm();

        if (norm > 0.0f)
        {
            nullspace /= norm;
        }

        Abackend =
            AcoarseBackend;
    }

    return H;
}

void buildRigidBodyModes(const Eigen::MatrixXf &V_rest, Eigen::MatrixXf &modes,
                         int &cols) {
  const int nVerts = static_cast<int>(V_rest.rows());
  const Eigen::Vector3f center = V_rest.colwise().mean();
  cols = 6;
  modes = Eigen::MatrixXf::Zero(3 * nVerts, cols);

  for (int i = 0; i < nVerts; ++i) {
    const Eigen::Vector3f relP = V_rest.row(i).transpose() - center;

    // 3 translations
    modes(3 * i + 0, 0) = 1.0f;
    modes(3 * i + 1, 1) = 1.0f;
    modes(3 * i + 2, 2) = 1.0f;

    // 3 infinitesimal rotations (cross products with the basis axes)
    modes(3 * i + 1, 3) = -relP.z();
    modes(3 * i + 2, 3) = relP.y();

    modes(3 * i + 0, 4) = relP.z();
    modes(3 * i + 2, 4) = -relP.x();

    modes(3 * i + 0, 5) = -relP.y();
    modes(3 * i + 1, 5) = relP.x();
  }

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(modes);
  modes = qr.householderQ() * Eigen::MatrixXf::Identity(3 * nVerts, cols);
}

gpuSolver::AMGHierarchy buildAmgclRugeStubenHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &M0,
    const Eigen::MatrixXf &V, int maxLevels, int minDofs) {
  using Coarsening = amgcl::coarsening::ruge_stuben<Backend>;

  auto configureParams = [](Coarsening::params &prm, int /*nrows*/,
                            const Eigen::MatrixXf & /*nullspace*/) {
    // Ruge-Stuben is more aggressive than SA, so a higher threshold is fine.
    prm.eps_strong = 0.25f;
  };

  return buildHierarchyImpl<Coarsening>(A0, M0, V, maxLevels, minDofs,
                                        configureParams);
}

gpuSolver::AMGHierarchy buildAmgclAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &M0,
    const Eigen::MatrixXf &V, int maxLevels, int minDofs) {
  using Coarsening = amgcl::coarsening::aggregation<Backend>;
  const int nRBM = 6;

  auto configureParams = [nRBM](Coarsening::params &prm, int nrows,
                                const Eigen::MatrixXf &nullspace) {
    prm.aggr.block_size = 3;
    prm.aggr.eps_strong = 0.1f;
    prm.nullspace.cols = nRBM;
    prm.nullspace.B.resize(static_cast<size_t>(nRBM) * nrows);

    // Row-major / per-node interleaved packing: B[row * nRBM + col].
    // See caveat about this in the write-up before you rely on it.
    for (int r = 0; r < nrows; ++r) {
      for (int c = 0; c < nRBM; ++c) {
        prm.nullspace.B[c * nRBM + r] = nullspace(r, c);
      }
    }
  };

  return buildHierarchyImpl<Coarsening>(A0, M0, V, maxLevels, minDofs,
                                        configureParams);
}

gpuSolver::AMGHierarchy buildAmgclSmoothedAggregationHierarchy(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A0,
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &M0,
    const Eigen::MatrixXf &V, int maxLevels, int minDofs) {
  using Coarsening = amgcl::coarsening::smoothed_aggregation<Backend>;
  const int nRBM = 6;

  auto configureParams = [nRBM](Coarsening::params &prm, int nrows,
                                const Eigen::MatrixXf &nullspace) {
    prm.aggr.block_size = 3;
    prm.aggr.eps_strong = 0.1f;
    prm.nullspace.cols = nRBM;
    prm.nullspace.B.resize(static_cast<size_t>(nRBM) * nrows);

    // Column-major packing: B[col * nrows + row]. NOTE this differs from
    // buildAmgclAggregationHierarchy above -- see caveat about this.
    for (int c = 0; c < nRBM; ++c) {
      for (int r = 0; r < nrows; ++r) {
        prm.nullspace.B[c * nrows + r] = nullspace(r, c);
      }
    }
  };

  return buildHierarchyImpl<Coarsening>(A0, M0, V, maxLevels, minDofs,
                                        configureParams);
}

// --------------------------------------------------------
// Standalone restriction-operator chains.
// --------------------------------------------------------

namespace {

template <typename Coarsening>
std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>>
buildRestrictionChainImpl(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A, int maxVertices,
    const std::function<void(typename Coarsening::params &)> &configureParams) {
  typename Coarsening::params prm;
  configureParams(prm);
  Coarsening coarsening(prm);

  std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> restrictionMatrices;

  std::shared_ptr<CrsMatrix> Abackend = toBackend(A);
  auto [Pptr, Rptr] = coarsening.transfer_operators(*Abackend);

  Eigen::SparseMatrix<float, Eigen::RowMajor> R = toEigen(*Rptr);
  Eigen::SparseMatrix<float, Eigen::RowMajor> P = toEigen(*Pptr);
  restrictionMatrices.push_back(R);

  Eigen::SparseMatrix<float, Eigen::RowMajor> Acurrent = A;

  while (R.rows() > maxVertices) {
    Eigen::SparseMatrix<float, Eigen::RowMajor> Acoarse = R * (Acurrent * P);

    std::shared_ptr<CrsMatrix> AcoarseBackend = toBackend(Acoarse);
    auto [PptrCoarse, RptrCoarse] =
        coarsening.transfer_operators(*AcoarseBackend);

    R = toEigen(*RptrCoarse);
    P = toEigen(*PptrCoarse);

    if (R.rows() <= maxVertices) {
      break;
    }

    restrictionMatrices.push_back(R);
    Acurrent = Acoarse;
  }

  return restrictionMatrices;
}

} // namespace

std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>>
buildRestrictionMatricesSmoothedAggregation(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A, int blockSize,
    float epsStrong, int maxVertices) {
  using Coarsening = amgcl::coarsening::smoothed_aggregation<Backend>;

  auto configureParams = [blockSize, epsStrong](Coarsening::params &prm) {
    prm.aggr.block_size = blockSize;
    prm.aggr.eps_strong = epsStrong;
  };

  return buildRestrictionChainImpl<Coarsening>(A, maxVertices, configureParams);
}

std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>>
buildRestrictionMatricesRugeStuben(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> &A, float epsStrong,
    int maxVertices) {
  using Coarsening = amgcl::coarsening::ruge_stuben<Backend>;

  auto configureParams = [epsStrong](Coarsening::params &prm) {
    prm.eps_strong = epsStrong;
  };

  return buildRestrictionChainImpl<Coarsening>(A, maxVertices, configureParams);
}

} // namespace MGBuilder
