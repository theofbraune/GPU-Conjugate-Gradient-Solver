#include <GPUSolver/metal/MetalContext.h>

#include <iostream>
#include <vector>

int main()
{
    gpuSolver::MetalContext context;

    std::vector<float> x{
        1.f,
        2.f,
        3.f,
        4.f,
        5.f
    };

    context.timesTwo(
        x.data(),
        x.size()
    );

    for (float value : x)
        std::cout << value << '\n';
}
