#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <cstddef>

namespace gpuSolver
{

class Backend;
class Preconditioner;

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

    std::size_t maxIterations_;
    float tolerance_;

public:
    CGSolver(
        Backend& backend,
        Preconditioner& preconditioner
    );

    ~CGSolver() = default;

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

    Eigen::VectorXf solve(
        const Matrix& A,
        const Eigen::VectorXf& b
    );
};

} // namespace gpuSolverpragma once
