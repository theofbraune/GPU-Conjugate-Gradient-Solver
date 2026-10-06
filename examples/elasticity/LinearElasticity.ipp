#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
// #include <glog/logging.h>
#include <iostream>
#include <memory>
#include <polyscope/point_cloud.h>
#include <polyscope/volume_mesh.h>

#include <igl/boundary_facets.h>
#include <igl/massmatrix.h>
#include <igl/volume.h>
#include <igl/voronoi_mass.h>

// LinearElasticitySimulator::LinearElasticitySimulator()

LinearElasticitySimulator::LinearElasticitySimulator(Eigen::MatrixXd V,
                                                     Eigen::MatrixXi F,
                                                     Eigen::MatrixXi T,
                                                     double lambda, double mu) {

  this->restPositions = V;
  int nVertices = V.rows();
  this->Tets = T;
  this->Faces = F;
  this->currentPosition = V;
  this->nextPositions = Eigen::MatrixXd::Zero(nVertices, 3);
  Eigen::VectorXd massVertices = Eigen::VectorXd::Zero(nVertices);
  // igl::voronoi_mass(V, T, massVertices);
  Eigen::VectorXd volumes;
  igl::volume(V, T, volumes);
  for (int idxTet = 0; idxTet < T.rows(); idxTet++) {
    double volCurr = volumes(idxTet);
    if (volCurr < 0) {
      // LOG(FATAL) << " inverted tet found";
      std::cerr << " inverted tet found " << std::endl;

    } else {
      int i0 = T(idxTet, 0);
      int i1 = T(idxTet, 1);
      int i2 = T(idxTet, 2);
      int i3 = T(idxTet, 3);
      massVertices(i0) += volCurr / 4.;
      massVertices(i1) += volCurr / 4.;
      massVertices(i2) += volCurr / 4.;
      massVertices(i3) += volCurr / 4.;
    }

    Eigen::Vector3d X0 = V.row(T(idxTet, 0));
    Eigen::Vector3d X1 = V.row(T(idxTet, 1));
    Eigen::Vector3d X2 = V.row(T(idxTet, 2));
    Eigen::Vector3d X3 = V.row(T(idxTet, 3));

    // Check orientation
    if ((X1 - X0).cross(X2 - X0).dot(X3 - X0) < 0) {
      std::swap(this->Tets(idxTet, 1), this->Tets(idxTet, 2)); // Fix winding
    }
  }

  // Use diagonal mass matrix in time integrator
  int sizeMat = 3 * nVertices;
  this->massMatrixStacked.resize(sizeMat, sizeMat);
  this->massMatrix.resize(nVertices, nVertices);
  std::vector<Eigen::Triplet<double>> tripsForStackedMass;
  std::vector<Eigen::Triplet<double>> tripsForMass;
  for (int i = 0; i < nVertices; i++) {

    double massVertex = massVertices(i);
    tripsForMass.emplace_back(i, i, massVertex);
    tripsForStackedMass.emplace_back(3 * i, 3 * i, massVertex);
    tripsForStackedMass.emplace_back(3 * i + 1, 3 * i + 1, massVertex);
    tripsForStackedMass.emplace_back(3 * i + 2, 3 * i + 2, massVertex);
  }
  this->massMatrixStacked.setFromTriplets(tripsForStackedMass.begin(),
                                          tripsForStackedMass.end());
  std::cout << " Size of the mass matrix stacked: " << massMatrixStacked.rows()
            << " x " << massMatrixStacked.cols() << std::endl;
  std::cout << " size of the standard mass matrix: " << massMatrix.rows()
            << " x " << massMatrix.cols() << std::endl;

  this->contactForces = Eigen::MatrixXd::Zero(nVertices, 3);
  this->gravitationalForces = Eigen::MatrixXd::Zero(nVertices, 3);
  this->elasticForces = Eigen::MatrixXd::Zero(nVertices, 3);
  this->elasticForcesConstrained = elasticForces;
  this->currentVelocity = Eigen::MatrixXd::Zero(nVertices, 3);
  this->lambda = lambda;
  this->mu = mu;

  igl::boundary_facets(this->Tets, this->boundaryFaces);

  this->constitutiveMatrix.setZero();
  constitutiveMatrix(0, 0) = lambda + 2 * mu;
  constitutiveMatrix(1, 1) = lambda + 2 * mu;
  constitutiveMatrix(2, 2) = lambda + 2 * mu;
  constitutiveMatrix(0, 1) = lambda;
  constitutiveMatrix(0, 2) = lambda;
  constitutiveMatrix(1, 2) = lambda;
  constitutiveMatrix(1, 0) = lambda;
  constitutiveMatrix(2, 0) = lambda;
  constitutiveMatrix(2, 1) = lambda;
  constitutiveMatrix(3, 3) = mu;
  constitutiveMatrix(4, 4) = mu;
  constitutiveMatrix(5, 5) = mu;

  this->precomputeGradientsShapeFunctionsPerTet();

  this->assembleHessianElasticForces();

  // precompute here the hessian

  //
  // LOG(INFO) << " Assembly done !";
}

void LinearElasticitySimulator::precomputeGradientsShapeFunctionsPerTet() {

  for (int idxTet = 0; idxTet < this->Tets.rows(); idxTet++) {

    int idx0 = Tets(idxTet, 0);
    int idx1 = Tets(idxTet, 1);
    int idx2 = Tets(idxTet, 2);
    int idx3 = Tets(idxTet, 3);

    Eigen::Vector3d X0 = restPositions.row(idx0);
    Eigen::Vector3d X1 = restPositions.row(idx1);
    Eigen::Vector3d X2 = restPositions.row(idx2);
    Eigen::Vector3d X3 = restPositions.row(idx3);

    Eigen::Matrix<double, 3, 3> Dm;
    Dm.col(0) = X1 - X0;
    Dm.col(1) = X2 - X0;
    Dm.col(2) = X3 - X0;
    Eigen::Matrix<double, 3, 3> B = Dm.inverse();
    double volume = std::abs(Dm.determinant()) / 6.;
    volumePerUndeformedTet.push_back(volume);

    // compute the gradient of the shape functions
    std::array<Eigen::Vector3d, 4> gradN;
    gradN[1] = B.row(0).transpose();
    gradN[2] = B.row(1).transpose();
    gradN[3] = B.row(2).transpose();
    gradN[0] = -(gradN[1] + gradN[2] + gradN[3]);

    gradientsShapeFunctionsPerTet.push_back(gradN);
  }
}

double LinearElasticitySimulator::computeEnergyPerTet(int idxTet) {

  int idx0 = Tets(idxTet, 0);
  int idx1 = Tets(idxTet, 1);
  int idx2 = Tets(idxTet, 2);
  int idx3 = Tets(idxTet, 3);

  Eigen::Vector3d X0 = restPositions.row(idx0);
  Eigen::Vector3d X1 = restPositions.row(idx1);
  Eigen::Vector3d X2 = restPositions.row(idx2);
  Eigen::Vector3d X3 = restPositions.row(idx3);

  Eigen::Matrix3d Dm;
  Dm.col(0) = X1 - X0;
  Dm.col(1) = X2 - X0;
  Dm.col(2) = X3 - X0;

  Eigen::Vector3d p0 = currentPosition.row(idx0);
  Eigen::Vector3d p1 = currentPosition.row(idx1);
  Eigen::Vector3d p2 = currentPosition.row(idx2);
  Eigen::Vector3d p3 = currentPosition.row(idx3);

  Eigen::Vector3d u0 = p0 - X0;
  Eigen::Vector3d u1 = p1 - X1;
  Eigen::Vector3d u2 = p2 - X2;
  Eigen::Vector3d u3 = p3 - X3;

  std::array<Eigen::Vector3d, 4> gradientsBasis =
      gradientsShapeFunctionsPerTet[idxTet];

  Eigen::Matrix3d gradu = Eigen::MatrixXd::Zero();

  gradu =
      u0 * gradientsBasis[0].transpose() + u1 * gradientsBasis[1].transpose() +
      u2 * gradientsBasis[2].transpose() + u3 * gradientsBasis[3].transpose();

  Eigen::Matrix3d epsilon = 0.5 * (gradu + gradu.transpose());
  double normEps = epsilon.norm();
  double traceEps = epsilon.trace();

  double strain = mu * normEps * normEps + 0.5 * lambda * traceEps * traceEps;
  double volume = std::abs(Dm.determinant()) / 6.0;
  double energy = volume * strain;
  // LOG(INFO) << " Energy per tet " << idxTet << " is " << energy;
  // LOG(INFO) << " gradu is \n" << gradu;
  // LOG(INFO) << " epsilon is \n" << epsilon;
  // LOG(INFO) <<" normEps "<<normEps;
  // LOG(INFO) <<" Tra eps "<<traceEps;
  // LOG(INFO) <<" lambda "<<lambda;
  // LOG(INFO) << " strain energy density is " << strain;
  // LOG(INFO) << " volume is " << volume;
  // std::cin.get();
  return energy;
}

double LinearElasticitySimulator::computeGlobalElasticEnergy() {

  Eigen::VectorXd displacement = this->getGlobalDisplacmentVectorStacked();
  double energy = 0.5 * (displacement.dot(this->hessian * displacement));
  this->elasticEnergy = energy;
  return energy;
}

Eigen::VectorXd LinearElasticitySimulator::computeGradientEnergyGlobal() {

  int nTets = Tets.rows();
  Eigen::VectorXd gradientElasticEnergy =
      Eigen::VectorXd::Zero(3 * currentPosition.rows());
  this->elasticForces = Eigen::MatrixXd::Zero(currentPosition.rows(), 3);

  Eigen::VectorXd displacement = this->getGlobalDisplacmentVectorStacked();
  Eigen::VectorXd gradientEnergy = this->hessian * displacement;
  for (int idxTet = 0; idxTet < nTets; idxTet++) {
    int idx0 = Tets(idxTet, 0);
    int idx1 = Tets(idxTet, 1);
    int idx2 = Tets(idxTet, 2);
    int idx3 = Tets(idxTet, 3);
    Eigen::Vector3d grad0 = gradientEnergy.segment(3 * idx0, 3);
    Eigen::Vector3d grad1 = gradientEnergy.segment(3 * idx1, 3);
    Eigen::Vector3d grad2 = gradientEnergy.segment(3 * idx2, 3);
    Eigen::Vector3d grad3 = gradientEnergy.segment(3 * idx3, 3);

    this->elasticForces.row(idx0) += -grad0.transpose();
    this->elasticForces.row(idx1) += -grad1.transpose();
    this->elasticForces.row(idx2) += -grad2.transpose();
    this->elasticForces.row(idx3) += -grad3.transpose();
  }

  return gradientEnergy;
}

Eigen::Matrix<double, 12, 12>
LinearElasticitySimulator::computeHessianPerTet(int idxTet) {
  Eigen::Matrix<double, 12, 12> hessTet;
  hessTet.setZero();

  double V = volumePerUndeformedTet[idxTet];
  std::array<Eigen::Vector3d, 4> gradN = gradientsShapeFunctionsPerTet[idxTet];

  // Loop over the 4 nodes of the tetrahedron
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      // Compute the 3x3 block H_ij
      // H_ij = V * ( mu * (gradNj . gradNi) * I + mu * (gradNj * gradNi^T) +
      // lambda * (gradNi * gradNj^T) )

      Eigen::Matrix3d Hij;
      double dot = gradN[j].dot(gradN[i]);

      Hij = mu * dot * Eigen::Matrix3d::Identity();
      Hij += mu * (gradN[j] * gradN[i].transpose());
      Hij += lambda * (gradN[i] * gradN[j].transpose());

      Hij *= V;

      // Place the 3x3 block into the 12x12 element Hessian
      hessTet.block<3, 3>(3 * i, 3 * j) = Hij;
    }
  }
  Eigen::Matrix<double, 12, 6> rigidModes;
  rigidModes.setZero();
  int i0 = this->Tets(idxTet, 0);
  int i1 = this->Tets(idxTet, 1);
  int i2 = this->Tets(idxTet, 2);
  int i3 = this->Tets(idxTet, 3);
  Eigen::Vector3d X0 = restPositions.row(i0);
  Eigen::Vector3d X1 = restPositions.row(i1);
  Eigen::Vector3d X2 = restPositions.row(i2);
  Eigen::Vector3d X3 = restPositions.row(i3);
  std::array<Eigen::Vector3d, 4> localX = {X0, X1, X2, X3};
  Eigen::Vector3d center = (X0 + X1 + X2 + X3) / 4.0;

  for (int i = 0; i < 4; ++i) {
    // 3 Translations: Unit displacement in x, y, z
    rigidModes(3 * i + 0, 0) = 1.0;
    rigidModes(3 * i + 1, 1) = 1.0;
    rigidModes(3 * i + 2, 2) = 1.0;

    // 3 Infinitesimal Rotations: omega x (X - center)
    Eigen::Vector3d r = localX[i] - center;
    rigidModes(3 * i + 1, 3) = -r.z();
    rigidModes(3 * i + 2, 3) = r.y();
    rigidModes(3 * i + 0, 4) = r.z();
    rigidModes(3 * i + 2, 4) = -r.x();
    rigidModes(3 * i + 0, 5) = -r.y();
    rigidModes(3 * i + 1, 5) = r.x();
  }

  // Compute residuals: R = Ke * V
  Eigen::Matrix<double, 12, 6> residuals = hessTet * rigidModes;
  double maxResidual = residuals.array().abs().maxCoeff();

  if (maxResidual > 1e-7) {
    std::cout << "Tet " << idxTet
              << " is NOT invariant! Max residual: " << maxResidual
              << std::endl;
    // Log individual mode norms to see if it's translation (0-2) or rotation
    // (3-5) failing
    for (int m = 0; m < 6; ++m) {
      std::cout << "  Mode " << m
                << " residual norm: " << residuals.col(m).norm() << std::endl;
    }
  }

  return hessTet;
}

void LinearElasticitySimulator::assembleHessianElasticForces() {

  this->hessian.resize(3 * this->restPositions.rows(),
                       3 * this->restPositions.rows());
  std::vector<Eigen::Triplet<double>> tripletsForGlobalHessian;

  for (int idxTet = 0; idxTet < this->Tets.rows(); idxTet++) {
    int i0 = this->Tets(idxTet, 0);
    int i1 = this->Tets(idxTet, 1);
    int i2 = this->Tets(idxTet, 2);
    int i3 = this->Tets(idxTet, 3);

    Eigen::Matrix<double, 12, 12> hessian = this->computeHessianPerTet(idxTet);
    std::vector<int> indices = {i0, i1, i2, i3};

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        Eigen::Matrix3d hessianij = hessian.block(3 * i, 3 * j, 3, 3);
        int idxi = indices[i];
        int idxj = indices[j];
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            double entry = hessianij(k, l);
            tripletsForGlobalHessian.emplace_back(3 * idxi + k, 3 * idxj + l,
                                                  entry);
          }
        }
      }
    }
  }

  this->hessian.setFromTriplets(tripletsForGlobalHessian.begin(),
                                tripletsForGlobalHessian.end(),
                                [](double a, double b) { return a + b; });

  Eigen::SparseMatrix<double, Eigen::RowMajor> hessianT =
      this->hessian.transpose();
  double maxDiff = (this->hessian - hessianT).norm();
  std::cout << " Max diff between hessian and its transpose is " << maxDiff
            << std::endl;
  // At the end of assembleHessianElasticForces
}

Eigen::Vector3d
LinearElasticitySimulator::getDisplacementVectorVertex(int idxVertex) {

  Eigen::Vector3d restPosition = this->restPositions.row(idxVertex);
  Eigen::Vector3d deformedPosition = this->currentPosition.row(idxVertex);
  return deformedPosition - restPosition;
}

Eigen::VectorXd LinearElasticitySimulator::getGlobalDisplacmentVectorStacked() {

  Eigen::VectorXd displacement =
      Eigen::VectorXd::Zero(3 * this->restPositions.rows());

  for (int i = 0; i < restPositions.rows(); i++) {
    Eigen::Vector3d disp = getDisplacementVectorVertex(i);
    displacement.segment(3 * i, 3) = disp;
  }

  return displacement;
}

void LinearElasticitySimulator::constrainSimulation(
    std::vector<int> indicesPinnedVertices) {
  this->indicesContrainedVertices = indicesPinnedVertices;

  int nFreeVerts =
      this->restPositions.rows() - indicesContrainedVertices.size();
  int nFullVertices = this->restPositions.rows();
  this->selectorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>(
      3 * nFreeVerts, 3 * nFullVertices);
  this->selectorMatrix.resize(3 * nFreeVerts, 3 * nFullVertices);
  std::vector<int> fullIndexToFreeVertex;
  int freeVertexCounter = 0;
  std::vector<Eigen::Triplet<double>> tripsForSelector;
  for (int i = 0; i < nFullVertices; i++) {
    auto pointerFind = std::find(indicesPinnedVertices.begin(),
                                 indicesPinnedVertices.end(), i);
    if (pointerFind == indicesPinnedVertices.end()) {
      fullIndexToFreeVertex.push_back(i);
      tripsForSelector.emplace_back(3 * freeVertexCounter, 3 * i, 1);
      tripsForSelector.emplace_back(3 * freeVertexCounter + 1, 3 * i + 1, 1);
      tripsForSelector.emplace_back(3 * freeVertexCounter + 2, 3 * i + 2, 1);
      freeVertexCounter++;
    }
  }
  std::cout << " The dof map has " << fullIndexToFreeVertex.size()
            << " entries " << std::endl;
  this->selectorMatrix.setFromTriplets(tripsForSelector.begin(),
                                       tripsForSelector.end());
  this->liftMatrix = this->selectorMatrix.transpose();

  this->hessianConstrained =
      selectorMatrix * (this->hessian * liftMatrix).eval();

  this->massMatrixStackedConstrained =
      selectorMatrix * (this->massMatrixStacked * liftMatrix).eval();

  /*
  // using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  // using Vec     = Eigen::VectorXd;
  //
  // SpMatRM restrict_matrix_rowmajor(const SpMatRM& A, const std::vector<int>&
  // dof_map, int n_free_dofs) {
  //     SpMatRM A_free(n_free_dofs, n_free_dofs);
  //     std::vector<Eigen::Triplet<double>> triplets;
  //     triplets.reserve(A.nonZeros());
  //
  //     // RowMajor ⇒ outerSize() iterates rows
  //     for (int i = 0; i < A.outerSize(); ++i) {
  //         for (SpMatRM::InnerIterator it(A, i); it; ++it) {
  //             int r = dof_map[it.row()];
  //             int c = dof_map[it.col()];
  //             if (r >= 0 && c >= 0) triplets.emplace_back(r, c, it.value());
  //         }
  //     }
  //
  //     A_free.setFromTriplets(triplets.begin(), triplets.end());
  //     return A_free;
  // }
  //
  // Vec lift_to_full(const Vec& x_free, const std::vector<int>& dof_map) {
  //     Vec x_full = Vec::Zero(dof_map.size());
  //
  //     for (int i = 0; i < (int)dof_map.size(); ++i) {
  //         int j = dof_map[i];
  //         if (j >= 0) x_full[i] = x_free[j];
  //     }
  //
  //     return x_full;
  // }
  //
  //
  // Eigen::MatrixXd build_VertsFree(const std::vector<VertexPtr>& vertices,
  // const std::vector<int>& indicesPinnedPositions, const std::vector<int>&
  // free_vertex_map) {
  //     const int n_vertices = vertices.size();
  //
  //     // mark pinned vertices
  //     std::vector<char> pinned_vertex(n_vertices, 0);
  //     for (int v : indicesPinnedPositions) pinned_vertex[v] = 1;
  //
  //     // build vertex map: full vertex index -> free vertex index
  //     // free_vertex_map.assign(n_vertices, -1);
  //     std::vector<int> freeVtxMap(n_vertices, -1);
  //     int n_free_vertices = 0;
  //     for (int v = 0; v < n_vertices; ++v) {
  //         if (!pinned_vertex[v]) {
  //             freeVtxMap[v] = n_free_vertices;
  //             ++n_free_vertices;
  //         }
  //     }
  //
  //     // fill VertsFree
  //     Eigen::MatrixXd VertsFree(n_free_vertices, 3);
  //     for (int v = 0; v < n_vertices; ++v) {
  //         int fv = freeVtxMap[v];
  //         if (fv >= 0) VertsFree.row(fv) = vertices[v]->position.transpose();
  //     }
  //
  //     return VertsFree;
  // }
  */
}

Eigen::VectorXd LinearElasticitySimulator::liftConstrainedToFull(
    const Eigen::VectorXd &vecConstrained) {

  Eigen::VectorXd lifted = this->liftMatrix * vecConstrained;
  return lifted;
}
