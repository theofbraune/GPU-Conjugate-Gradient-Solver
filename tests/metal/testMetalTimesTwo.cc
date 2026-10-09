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

    // gpuSolver::DeviceVector x(context, xHost);
    // gpuSolver::DeviceVector y(context, yHost);
    gpuSolver::DeviceVector* xPtr = backend.createVector(sizeXresult,xHost.data());
    gpuSolver::DeviceVector* yPtr = backend.createVector(sizeXresult,yHost.data());


    backend.scale(*xPtr, 2.0f);

    backend.axpy(
        3.0f,
        *xPtr,
        *yPtr
    );

    float* xResult = xPtr->download();
    float* yResult = yPtr->download();

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

    delete xPtr;
    delete yPtr;

    return 0;
}
