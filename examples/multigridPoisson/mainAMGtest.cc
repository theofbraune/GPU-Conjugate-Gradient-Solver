#include <GPUSolver/CGSolver.h>
#include <GPUSolver/Preconditioners/MetalJacobiPreconditioner.h>
#include <GPUSolver/Preconditioners/DampedJacobiPreconditioner.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>
#include <AMGUtils/AMGHierarchyBuilder.h>

#include <igl/cotmatrix.h>
#include <igl/grad.h>
#include <igl/read_triangle_mesh.h>

#include <ostream>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: surfacePoissonTest mesh.obj\n";

    return 1;
  }

  // --------------------------------------------------------
  // Load triangle mesh.
  // --------------------------------------------------------

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  if (!igl::read_triangle_mesh(argv[1], V, F)) {
    throw std::runtime_error("Could not load triangle mesh.");
  }

  const Eigen::Index n = V.rows();

  if (n < 4) {
    throw std::runtime_error("Mesh is too small.");
  }

  std::cout << "Loaded mesh with " << V.rows() << " vertices and " << F.rows()
            << " triangles.\n";

  // --------------------------------------------------------
  // Construct cotangent Laplacian.
  //
  // libigl's cotmatrix uses the convention that L is
  // negative semidefinite.
  //
  // Therefore
  //
  //     K = -L
  //
  // is positive semidefinite.
  // --------------------------------------------------------

  Eigen::SparseMatrix<double> LDouble;

  igl::cotmatrix(V, F, LDouble);

  Eigen::SparseMatrix<float, Eigen::RowMajor> K = (-LDouble).cast<float>();

  K.makeCompressed();

  // --------------------------------------------------------
  // Pick three distinct random vertices carrying charges.
  // --------------------------------------------------------

  std::mt19937 generator(42);

  std::uniform_int_distribution<Eigen::Index> distribution(0, n - 1);

  Eigen::Index chargeVertex0 = distribution(generator);

  Eigen::Index chargeVertex1 = distribution(generator);

  while (chargeVertex1 == chargeVertex0) {
    chargeVertex1 = distribution(generator);
  }

  Eigen::Index chargeVertex2 = distribution(generator);

  while (chargeVertex2 == chargeVertex0 || chargeVertex2 == chargeVertex1) {
    chargeVertex2 = distribution(generator);
  }

  // --------------------------------------------------------
  // Build the point-charge RHS.
  //
  // Let's deliberately choose charges which do not sum to
  // zero first, and then explicitly project onto the image
  // of the Laplacian.
  // --------------------------------------------------------

  Eigen::VectorXf b = Eigen::VectorXf::Zero(n);

  b[chargeVertex0] = 1.0f;
  b[chargeVertex1] = 0.7f;
  b[chargeVertex2] = -0.4f;

  std::cout << "\nCharge vertices:\n"
            << "  v0 = " << chargeVertex0 << "\n"
            << "  v1 = " << chargeVertex1 << "\n"
            << "  v2 = " << chargeVertex2 << "\n";

  std::cout << "\nCharge sum before projection = " << b.sum() << "\n";

  // --------------------------------------------------------
  // Project RHS onto im(K).
  //
  // For a connected closed surface
  //
  //     ker(K) = span{1}
  //
  // so we remove the constant component:
  //
  //     b <- b - mean(b) * 1.
  // --------------------------------------------------------

  const float meanCharge = b.mean();

  b.array() -= meanCharge;

  std::cout << "Charge sum after projection  = " << b.sum() << "\n";

  // --------------------------------------------------------
  // K is still singular because constants are in its kernel.
  //
  // To give CG a genuinely SPD matrix, fix one vertex:
  //
  //     u[pinnedVertex] = 0.
  //
  // We construct the reduced (n-1)x(n-1) system.
  // --------------------------------------------------------

  const Eigen::Index pinnedVertex = 0;

  std::vector<Eigen::Index> oldToReduced(static_cast<std::size_t>(n), -1);

  Eigen::Index reducedIndex = 0;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    oldToReduced[static_cast<std::size_t>(i)] = reducedIndex;

    ++reducedIndex;
  }

  // --------------------------------------------------------
  // Build reduced sparse matrix.
  // --------------------------------------------------------

  using Matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;

  std::vector<Eigen::Triplet<float>> triplets;

  triplets.reserve(static_cast<std::size_t>(K.nonZeros()));

  for (Eigen::Index row = 0; row < K.outerSize(); ++row) {
    for (Matrix::InnerIterator it(K, row); it; ++it) {
      const Eigen::Index i = it.row();

      const Eigen::Index j = it.col();

      if (i == pinnedVertex || j == pinnedVertex) {
        continue;
      }

      triplets.emplace_back(oldToReduced[static_cast<std::size_t>(i)],
                            oldToReduced[static_cast<std::size_t>(j)],
                            it.value());
    }
  }

  Matrix A(n - 1, n - 1);

  A.setFromTriplets(triplets.begin(), triplets.end());

  A.makeCompressed();

  std::vector<Matrix> restrictionOperators = MGBuilder::buildRestrictionMatricesSmoothedAggregation(A,1,0.01f,10000);

  std::cout<<" The vector of the restriction matrices is of size "<<restrictionOperators.size()<<std::endl;
  for(Matrix matrix_: restrictionOperators){
    std::cout<<" current matrix is of size: "<<matrix_.rows()<<" x "<<matrix_.cols()<<std::endl;
  }

  // --------------------------------------------------------
  // Reduced RHS.
  //
  // Since the pinned value is zero, no boundary contribution
  // needs to be added.
  // --------------------------------------------------------

  Eigen::VectorXf bReduced(n - 1);

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    bReduced[oldToReduced[static_cast<std::size_t>(i)]] = b[i];
  }

  // --------------------------------------------------------
  // Initial guess.
  // --------------------------------------------------------

  Eigen::VectorXf xReduced = Eigen::VectorXf::Zero(n - 1);

  // --------------------------------------------------------
  // Set up Metal backend.
  // --------------------------------------------------------

  gpuSolver::MetalContext context;

  gpuSolver::MetalBackend backend(context);

  // --------------------------------------------------------
  // Jacobi preconditioner.
  //
  // Adapt this one line to the exact constructor/API you
  // implemented.
  // --------------------------------------------------------

  std::size_t nRows = std::size_t(V.rows());

  float *diagonalValues = new float[nRows - 1];
  for (int i = 0; i < nRows - 1; i++) {
    diagonalValues[i] = A.coeff(i, i);
  }

  // gpuSolver::IdentityPreconditioner preconditioner;
  // gpuSolver::MetalJacobiPreconditioner preconditioner;
      // gpuSolver::MetalJacobiPreconditioner();
  gpuSolver::DampedJacobiPreconditioner preconditioner(2,0.6f);


  // --------------------------------------------------------
  // GPU PCG.
  // --------------------------------------------------------

  gpuSolver::CGSolver solver(backend, preconditioner, A);

  solver.setMaxIterations(200);

  solver.setTolerance(1e-6f);

  std::cout << "\nStarting GPU PCG...\n";

  solver.solve(bReduced, xReduced);

  std::cout << "GPU PCG finished.\n";
  std::size_t nIter = solver.getNbOfIterations();
  std::cout << "finished in " << nIter << " iterations" << std::endl;

  // --------------------------------------------------------
  // Reconstruct full potential.
  // --------------------------------------------------------

  Eigen::VectorXf potential = Eigen::VectorXf::Zero(n);

  potential[pinnedVertex] = 0.0f;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == pinnedVertex) {
      continue;
    }

    potential[i] = xReduced[oldToReduced[static_cast<std::size_t>(i)]];
  }

  // --------------------------------------------------------
  // Check residual against the ORIGINAL Laplacian equation.
  //
  //     r = K*u - b
  // --------------------------------------------------------

  const Eigen::VectorXf residual = K * potential - b;

  const float absoluteResidual = residual.norm();

  const float relativeResidual = absoluteResidual / b.norm();

  Eigen::ConjugateGradient<Matrix, Eigen::Lower | Eigen::Upper,
                           Eigen::DiagonalPreconditioner<float>>
      eigenCG;

  eigenCG.setTolerance(1e-6f);

  eigenCG.setMaxIterations(5000);

  eigenCG.compute(A);

  Eigen::VectorXf xEigen = eigenCG.solve(bReduced);

  std::cout << "Eigen iterations = " << eigenCG.iterations() << "\n";

  std::cout << "Eigen reported error = " << eigenCG.error() << "\n";

  const float differenceToEigen =
      (xReduced - xEigen).norm() / std::max(1.0f, xEigen.norm());

  std::cout << "Relative difference GPU/Eigen = " << differenceToEigen << "\n";

  Eigen::VectorXf eigenReducedResidual = A * xEigen - bReduced;

  const float eigenReducedRelativeResidual =
      eigenReducedResidual.norm() / bReduced.norm();

  std::cout << "Eigen actual reduced residual = "
            << eigenReducedRelativeResidual << std::endl;

  const float gpuTrueResidual =
      (A * xReduced - bReduced).norm() / bReduced.norm();

  const float eigenTrueResidual =
      (A * xEigen - bReduced).norm() / bReduced.norm();

  const float relativeSolutionDifference =
      (xReduced - xEigen).norm() / std::max(1.0f, xEigen.norm());

  std::cout << "\nSolver comparison:\n"
            << "  GPU iterations              = " << solver.getNbOfIterations()
            << "\n"
            << "  GPU true relative residual  = " << gpuTrueResidual << "\n"
            << "  Eigen iterations            = " << eigenCG.iterations()
            << "\n"
            << "  Eigen true relative residual= " << eigenTrueResidual << "\n"
            << "  GPU/Eigen solution diff     = " << relativeSolutionDifference
            << "\n";
  constexpr float residualSanityTolerance = 5e-4f;

  constexpr float solutionComparisonTolerance = 5e-4f;

  bool passed = true;

  if (gpuTrueResidual > residualSanityTolerance) {
    std::cerr << "[FAIL] GPU true residual is too large: " << gpuTrueResidual
              << "\n";

    passed = false;
  }

  if (relativeSolutionDifference > solutionComparisonTolerance) {
    std::cerr << "[FAIL] GPU solution differs too much from Eigen: "
              << relativeSolutionDifference << "\n";

    passed = false;
  }

  if (passed) {
    std::cout << "\n[PASS] Surface Poisson sanity check.\n";
  } else {
    std::cerr << "\n[FAIL] Surface Poisson sanity check.\n";

    return 1;
  }
  std::cout << "  requested CG tolerance      = " << solver.tolerance()
            << "\n";
  // --------------------------------------------------------
  // Compute electric field
  //
  //     E = -grad(u)
  //
  // For piecewise-linear vertex potentials, grad(u) is
  // piecewise constant on each triangle.
  // --------------------------------------------------------

  Eigen::SparseMatrix<double> gradientOperator;

  igl::grad(V, F, gradientOperator);

  const Eigen::VectorXd potentialDouble = potential.cast<double>();

  const Eigen::VectorXd gradientFlat = gradientOperator * potentialDouble;

  // libigl stores the gradient components in blocks:
  //
  //     [ gx ]
  //     [ gy ]
  //     [ gz ]
  //
  // each block has #F entries.
  //
  // Convert this to an #F x 3 matrix for Polyscope.
  // --------------------------------------------------------

  Eigen::MatrixXd electricField(F.rows(), 3);

  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    electricField(f, 0) = -gradientFlat[f];

    electricField(f, 1) = -gradientFlat[f + F.rows()];

    electricField(f, 2) = -gradientFlat[f + 2 * F.rows()];
  }

  // --------------------------------------------------------
  // Positions of the three original point charges.
  // --------------------------------------------------------

  Eigen::MatrixXd chargePositions(3, 3);

  chargePositions.row(0) = V.row(chargeVertex0);

  chargePositions.row(1) = V.row(chargeVertex1);

  chargePositions.row(2) = V.row(chargeVertex2);

  Eigen::VectorXd chargeValues(3);

  chargeValues << 1.0, 0.7, -0.4;

  // ========================================================
  // Polyscope visualization
  // ========================================================

  polyscope::init();

  // --------------------------------------------------------
  // Surface mesh.
  // --------------------------------------------------------

  polyscope::SurfaceMesh *psMesh =
      polyscope::registerSurfaceMesh("surface", V, F);

  // --------------------------------------------------------
  // Potential u.
  // --------------------------------------------------------

  polyscope::SurfaceVertexScalarQuantity *potentialQuantity =
      psMesh->addVertexScalarQuantity("potential", potential);

  potentialQuantity->setEnabled(true);

  // --------------------------------------------------------
  // Projected charge density.
  //
  // This is the actual RHS which entered the Poisson solve.
  // --------------------------------------------------------

  polyscope::SurfaceVertexScalarQuantity *chargeDensityQuantity =
      psMesh->addVertexScalarQuantity("projected charge density", b);

  chargeDensityQuantity->setEnabled(false);

  // --------------------------------------------------------
  // Electric field.
  //
  //     E = -grad(u)
  //
  // This is naturally a per-face vector field.
  // --------------------------------------------------------

  polyscope::SurfaceFaceVectorQuantity *electricFieldQuantity =
      psMesh->addFaceVectorQuantity("electric field", electricField);

  electricFieldQuantity->setEnabled(true);

  electricFieldQuantity->setVectorLengthScale(0.03);

  // --------------------------------------------------------
  // Original point charges.
  //
  // Show the three selected vertices as a separate point
  // cloud so they remain clearly visible on top of the mesh.
  // --------------------------------------------------------

  polyscope::PointCloud *chargeCloud =
      polyscope::registerPointCloud("point charges", chargePositions);

  chargeCloud->setPointRadius(0.015, true);

  // Store the signed charge values on the point cloud as
  // another scalar quantity.
  polyscope::PointCloudScalarQuantity *pointChargeQuantity =
      chargeCloud->addScalarQuantity("charge", chargeValues);

  pointChargeQuantity->setEnabled(true);

  // --------------------------------------------------------
  // Launch viewer.
  // --------------------------------------------------------

  polyscope::show();

  delete[] diagonalValues;
  return 0;
}
