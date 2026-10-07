#pragma once

#include <Eigen/Core>

namespace MGBuilder {

Eigen::MatrixXf
buildRigidBodyModes3D(
    const Eigen::MatrixXf &vertices);

} // namespace MGBuilder
