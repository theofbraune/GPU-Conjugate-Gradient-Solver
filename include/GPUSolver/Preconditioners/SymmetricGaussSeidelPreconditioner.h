#pragma once

#include "../Preconditioner.h"

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
    // Borrowed from CGSolver or the multigrid level.
    const DeviceCSRMatrix* matrix_;

    // Owned GPU coloring.
    DeviceIndexVector* colorVertices_;

    // Host-side boundaries between color classes.
    std::vector<int> colorOffsets_;

    std::size_t numberOfColors_;

    float omega_;

public:
    explicit SymmetricGaussSeidelPreconditioner(
        float omega = 1.0f
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

    // Updates an existing solution without resetting it.
    void smooth(
        Backend& backend,
        BackendEncoder& encoder,
        const DeviceVector& rhs,
        DeviceVector& solution,
        std::size_t iterations
    );
};

} // namespace gpuSolver
