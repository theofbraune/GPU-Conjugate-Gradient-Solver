#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    gpuSolver::MetalContext context;
    gpuSolver::MetalBackend backend(context);

    std::vector<float> xHost = {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };

    std::vector<float> yHost = {
        10.0f,
        10.0f,
        10.0f,
        10.0f
    };

    std::size_t sizeXresult = 4;

    gpuSolver::DeviceVector x(context, xHost);
    gpuSolver::DeviceVector y(context, yHost);

    backend.scale(x, 2.0f);

    backend.axpy(
        3.0f,
        x,
        y
    );

    float* xResult = x.download();
    float* yResult = y.download();

    std::cout << "x after scale:\n";

    for (std::size_t i = 0; i < sizeXresult; ++i)
    {
        std::cout << xResult[i] << " ";
    }

    std::cout << "\n\n";

    std::cout << "y after axpy:\n";

    for (std::size_t i = 0; i <sizeXresult; ++i)
    {
        std::cout << yResult[i] << " ";
    }

    std::cout << "\n";

    return 0;
}
