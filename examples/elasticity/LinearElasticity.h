
#pragma once

#include "ElasticitySimulator.h"
#include <Eigen/Core>

class LinearElasticitySimulator : public ElasticitySimulator {

  // TinyAD::ScalarFunction<3,double, int>  energyFunction;
  double computeEnergyPerTet(int idxTet);
  Eigen::VectorXd computeGradientEnergyPerTet(int idxTet);
  Eigen::Matrix<double, 12,12> computeHessianPerTet(int idxTet);

  std::vector<std::array<Eigen::Vector3d,4>> gradientsShapeFunctionsPerTet;

  void precomputeGradientsShapeFunctionsPerTet();
  double mu;
  double lambda;

  Eigen::Vector<double, 12> gradientPerTet(int idxTet);
  Eigen::Matrix<double,6,6> constitutiveMatrix;
  std::vector<double> volumePerUndeformedTet;

  Eigen::Vector3d getDisplacementVectorVertex(int idxVertex);
  Eigen::MatrixXi boundaryFaces;
  std::vector<int> freeToConstrraintMap; // give the index of a free vertex, tells you 
  

public:
  LinearElasticitySimulator() = default;
  ~LinearElasticitySimulator() override = default;

  LinearElasticitySimulator(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXi T, double lambda, double mu);

  double computeGlobalElasticEnergy();
  void constrainSimulation(std::vector<int> indicesPinnedVertices) override;
  Eigen::VectorXd computeGradientEnergyGlobal();
  void assembleHessianElasticForces();
  Eigen::VectorXd getGlobalDisplacmentVectorStacked();
  Eigen::VectorXd liftConstrainedToFull(const Eigen::VectorXd& vecConstrained) override;



};

#include "LinearElasticity.ipp"
