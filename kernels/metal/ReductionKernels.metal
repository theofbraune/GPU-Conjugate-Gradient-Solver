#include <metal_stdlib>

kernel void dotPartial(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],

    device float* partialSums [[buffer(2)]],

    constant uint& n [[buffer(3)]],

    threadgroup float* scratch [[threadgroup(0)]],

    uint globalId [[thread_position_in_grid]],
    uint localId [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]],
    uint threadsPerGroup [[threads_per_threadgroup]])
{
    float value = 0.0f;

    if (globalId < n)
    {
        value =
            x[globalId]
            * y[globalId];
    }

    scratch[localId] =
        value;

    threadgroup_barrier(
        metal::mem_flags::mem_threadgroup
    );

    // --------------------------------------------------------
    // Tree reduction inside this threadgroup.
    // --------------------------------------------------------

    for (uint stride = threadsPerGroup / 2;
         stride > 0;
         stride /= 2)
    {
        if (localId < stride)
        {
            scratch[localId] +=
                scratch[localId + stride];
        }

        threadgroup_barrier(
            metal::mem_flags::mem_threadgroup
        );
    }

    // Thread 0 owns the result of this threadgroup.
    if (localId == 0)
    {
        partialSums[groupId] =
            scratch[0];
    }
}


kernel void reduceSumPartial(
    device const float* input [[buffer(0)]],

    device float* partialSums [[buffer(1)]],

    constant uint& n [[buffer(2)]],

    threadgroup float* scratch [[threadgroup(0)]],

    uint globalId [[thread_position_in_grid]],
    uint localId [[thread_position_in_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]],
    uint threadsPerGroup [[threads_per_threadgroup]])
{
    float value = 0.0f;

    if (globalId < n)
    {
        value =
            input[globalId];
    }

    scratch[localId] =
        value;

    threadgroup_barrier(
        metal::mem_flags::mem_threadgroup
    );

    for (uint stride = threadsPerGroup / 2;
         stride > 0;
         stride /= 2)
    {
        if (localId < stride)
        {
            scratch[localId] +=
                scratch[localId + stride];
        }

        threadgroup_barrier(
            metal::mem_flags::mem_threadgroup
        );
    }

    if (localId == 0)
    {
        partialSums[groupId] =
            scratch[0];
    }
}
