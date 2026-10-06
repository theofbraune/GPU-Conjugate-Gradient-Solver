#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

class ElasticitySimulator {
    // geometry
  protected:
    Eigen::MatrixXd restPositions;
    Eigen::MatrixXd currentPosition;

    Eigen::MatrixXd currentVelocity;
    Eigen::MatrixXi Tets;
    Eigen::MatrixXi Faces;
    Eigen::MatrixXi boundaryFaces;
    Eigen::SparseMatrix<double, Eigen::RowMajor> massMatrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor> massMatrixStacked;

    // Eigen::SparseMatrix<double, Eigen::RowMajor> massMatrixConstrained;
    Eigen::SparseMatrix<double, Eigen::RowMajor> massMatrixStackedConstrained;

    // elastic energy parameters
    Eigen::SparseMatrix<double, Eigen::RowMajor> hessian;
    Eigen::SparseMatrix<double, Eigen::RowMajor> hessianConstrained;
    Eigen::MatrixXd elasticForces;
    Eigen::MatrixXd elasticForcesConstrained;

    // constraint parameters
    std::vector<int> indicesContrainedVertices;
    Eigen::MatrixXd contactForces;
    Eigen::MatrixXd gravitationalForces;

    // time integrator

    Eigen::MatrixXd nextPositions;

    double elasticEnergy;
    Eigen::SparseMatrix<double, Eigen::RowMajor> selectorMatrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor> liftMatrix;


  public:
    virtual ~ElasticitySimulator()                                           = default;
    virtual void constrainSimulation(std::vector<int> indicesPinnedVertices) = 0;
    Eigen::SparseMatrix<double, Eigen::RowMajor> getHessian() const { return hessianConstrained; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getFullHessian() const { return hessian; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getMassMatrix() const { return massMatrix; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getMassMatrixStacked() const { return massMatrixStacked; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getMassMatrixStackedConstrained() const { return massMatrixStackedConstrained; }
    Eigen::MatrixXd getRestPositions() const { return restPositions; }
    Eigen::MatrixXd getCurrentPositions() const { return currentPosition; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getSelectorMatrix() const { return selectorMatrix; }
    Eigen::SparseMatrix<double, Eigen::RowMajor> getLiftMatrix() const { return liftMatrix; }
    virtual Eigen::VectorXd liftConstrainedToFull(const Eigen::VectorXd& vecConstrained) = 0;


    // we need a constructor that passes the rest tetmesh, the lame parameters
    // set pinned vertices
    // reserve a
    // set as methods evaluate energy, evaluate elastic forces, evaluate hessian
    // where we pass a new matrix of the new set of vertices compute contactForces
    // compute gravitational force
};
