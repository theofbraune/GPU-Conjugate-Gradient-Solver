#include <metal_stdlib>

kernel void blockGaussSeidelColor3x3(
    device const int* rowPtr          [[buffer(0)]],
    device const int* colPtr          [[buffer(1)]],
    device const float* values        [[buffer(2)]],
    device const int* colorBlocks     [[buffer(3)]],
    device const float* inverseBlocks [[buffer(4)]],
    device const float* rhs           [[buffer(5)]],
    device float* x                   [[buffer(6)]],
    constant uint& colorStart         [[buffer(7)]],
    constant uint& colorCount         [[buffer(8)]],
    constant float& omega             [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= colorCount) {
        return;
    }

    const uint block =
        colorBlocks[colorStart + tid];

    const uint row0 = 3 * block;
    const uint row1 = row0 + 1;
    const uint row2 = row0 + 2;

    float r0 = rhs[row0];
    float r1 = rhs[row1];
    float r2 = rhs[row2];

    const uint blockFirstColumn =
        3 * block;

    const uint blockLastColumn =
        blockFirstColumn + 3;

    // ----------------------------------------------------
    // Row 0
    // ----------------------------------------------------

    for (int entry = rowPtr[row0];
         entry < rowPtr[row0 + 1];
         ++entry) {

        const uint col = colPtr[entry];

        // Skip the complete diagonal 3x3 block.
        if (col >= blockFirstColumn &&
            col < blockLastColumn) {
            continue;
        }

        r0 -= values[entry] * x[col];
    }

    // ----------------------------------------------------
    // Row 1
    // ----------------------------------------------------

    for (int entry = rowPtr[row1];
         entry < rowPtr[row1 + 1];
         ++entry) {

        const uint col = colPtr[entry];

        if (col >= blockFirstColumn &&
            col < blockLastColumn) {
            continue;
        }

        r1 -= values[entry] * x[col];
    }

    // ----------------------------------------------------
    // Row 2
    // ----------------------------------------------------

    for (int entry = rowPtr[row2];
         entry < rowPtr[row2 + 1];
         ++entry) {

        const uint col = colPtr[entry];

        if (col >= blockFirstColumn &&
            col < blockLastColumn) {
            continue;
        }

        r2 -= values[entry] * x[col];
    }

    // ----------------------------------------------------
    // Apply inverse diagonal block.
    // ----------------------------------------------------

    const uint matrixBase =
        9 * block;

    const float new0 =
        inverseBlocks[matrixBase + 0] * r0 +
        inverseBlocks[matrixBase + 1] * r1 +
        inverseBlocks[matrixBase + 2] * r2;

    const float new1 =
        inverseBlocks[matrixBase + 3] * r0 +
        inverseBlocks[matrixBase + 4] * r1 +
        inverseBlocks[matrixBase + 5] * r2;

    const float new2 =
        inverseBlocks[matrixBase + 6] * r0 +
        inverseBlocks[matrixBase + 7] * r1 +
        inverseBlocks[matrixBase + 8] * r2;

    // Save old values before writing.
    const float old0 = x[row0];
    const float old1 = x[row1];
    const float old2 = x[row2];

    x[row0] =
        (1.0f - omega) * old0
        + omega * new0;

    x[row1] =
        (1.0f - omega) * old1
        + omega * new1;

    x[row2] =
        (1.0f - omega) * old2
        + omega * new2;
}
