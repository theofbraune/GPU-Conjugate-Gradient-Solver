#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// libigl includes
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/readMESH.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>

// polyscope includes
#include <Eigen/Core>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>

#include <GLFW/glfw3.h>
#include <glog/logging.h>
void emptyGLFWErrorCallback(int, const char *) {}

void callback() {}

Eigen::MatrixXd V;
Eigen::MatrixXi F, T;

polyscope::VolumeMesh *psMesh = nullptr;
polyscope::PointCloud *psPts = nullptr;

int main(int argc, char *argv[]) {
  std::string path_to_source = std::string(PROJECT_SOURCE_DIR);

  // Ensure the logs directory exists
  FLAGS_logtostderr = 1;
  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  std::string mesh_path;

  // check if a mesh was passed as argument
  if (argc > 1) {
    mesh_path = std::string(argv[1]);
  } else {
    mesh_path = path_to_source + "/data/tetmeshes/box.mesh";
  }
  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  igl::readMESH(mesh_path, V, T, F);

  double xMean = V.col(0).mean();
  double yMean = V.col(1).mean();
  double zMean = V.col(2).mean();
  V.col(0) = V.col(0).array() - xMean;
  V.col(1) = V.col(1).array() - yMean;
  V.col(2) = V.col(2).array() - zMean;
  double xMin = V.col(0).minCoeff();
  V.col(0) = V.col(0).array() - xMin;

  glfwSetErrorCallback(emptyGLFWErrorCallback);
  polyscope::state::userCallback = callback;
  psMesh = polyscope::registerTetMesh("tet mesh", V, T);
  psPts = polyscope::registerPointCloud("points", V);
  auto *psCompare = polyscope::registerPointCloud("pointsRest", V);

  Eigen::SparseMatrix<double> laplacian;
  Eigen::SparseMatrix<double> massMatrix;
  igl::cotmatrix(V, T, laplacian);
  igl::massmatrix(V, T, igl::MASSMATRIX_TYPE_DEFAULT, massMatrix);
  // testGravity(V, F, T);
  // testPull(V, F, T);
  polyscope::show();
  return 0;
}
