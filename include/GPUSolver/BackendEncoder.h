#pragma once

namespace gpuSolver
{

class BackendEncoder
{
public:
    virtual ~BackendEncoder() = default;

    BackendEncoder(
        const BackendEncoder&
    ) = delete;

    BackendEncoder& operator=(
        const BackendEncoder&
    ) = delete;

protected:
    BackendEncoder() = default;
};

} // namespace gpuSolver
