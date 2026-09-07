#pragma once

namespace gpuSolver
{

class Backend;
class BackendEncoder;
class DeviceVector;

class Preconditioner
{
public:
    virtual ~Preconditioner() = default;

    virtual void apply(
        Backend& backend,
        BackendEncoder& encoder,
        const DeviceVector& input,
        DeviceVector& output
    ) = 0;
};

} // namespace gpuSolver
