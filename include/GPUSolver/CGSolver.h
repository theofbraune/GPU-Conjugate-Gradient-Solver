#pragma once

#include "GPUSolver/BackendEncoder.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cstddef>

namespace gpuSolver
{

class Backend;
class BackendEncoder;
class Preconditioner;
class HostCSRMatrix;
class DeviceCSRMatrix;

class CGSolver
{
public:
    using Matrix =
        Eigen::SparseMatrix<
            float,
            Eigen::RowMajor
        >;

private:
    Backend& backend_;
    Preconditioner& preconditioner_;

    Matrix matrix_;

    HostCSRMatrix* hostMatrix_;
    DeviceCSRMatrix* deviceMatrix_;

    std::size_t maxIterations_;
    float tolerance_;


public:
    CGSolver(
        Backend& backend,
        Preconditioner& preconditioner,
        const Matrix& matrix
    );

    ~CGSolver();

    CGSolver(
        const CGSolver&
    ) = delete;

    CGSolver& operator=(
        const CGSolver&
    ) = delete;

    void setMaxIterations(
        std::size_t maxIterations
    );

    void setTolerance(
        float tolerance
    );

    std::size_t maxIterations() const;

    float tolerance() const;

    void solve(
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x
    );
};

} // namespace gpuSolver


