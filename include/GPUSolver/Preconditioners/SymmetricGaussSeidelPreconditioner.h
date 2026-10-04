#pragma once

#include "../Preconditioner.h"
#include "GPUSolver/Smoother.h"

#include <cstddef>
#include <vector>

namespace gpuSolver {

class Backend;
class BackendEncoder;
class DeviceCSRMatrix;
class DeviceVector;
class DeviceIndexVector;
class HostCSRMatrix;

class SymmetricGaussSeidelPreconditioner : public Preconditioner
{
private:
    
    std::size_t nbOfReps = 1;

    Smoother* smoother = nullptr;

    float omega_;

public:
    explicit SymmetricGaussSeidelPreconditioner(
        float omega = 1.0f, std::size_t nbOfIters = 1
    );

    ~SymmetricGaussSeidelPreconditioner() override;

    SymmetricGaussSeidelPreconditioner(
        const SymmetricGaussSeidelPreconditioner&
    ) = delete;

    SymmetricGaussSeidelPreconditioner& operator=(
        const SymmetricGaussSeidelPreconditioner&
    ) = delete;

    void initialize(
        Backend& backend,
        const HostCSRMatrix& hostMatrix,
        const DeviceCSRMatrix& deviceMatrix
    ) override;

    void apply(
        Backend& backend,
        BackendEncoder& encoder,
        const DeviceVector& input,
        DeviceVector& output
    ) override;

};

} // namespace gpuSolver
