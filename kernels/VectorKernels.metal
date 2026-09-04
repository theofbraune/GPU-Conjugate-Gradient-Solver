#include <metal_stdlib>

kernel void timesTwo(
    device float* x [[buffer(0)]],
    uint id [[thread_position_in_grid]]
){
    x[id] *=2.0f;
}


kernel void scale(
  device float* x [[buffer(0)]],
  constant float& alpha [[buffer(1)]],
  uint id[[thread_position_in_grid]]
){
  x[id] *=alpha;
}


kernel void axpy(
  device float* x [[buffer(0)]],
  device float* y [[buffer(1)]],
  constant float& alpha [[buffer(2)]],
  uint id[[thread_position_in_grid]]
){
  y[id] += alpha * x[id];
}


kernel void spmv(
  device const int* rowPtr [[buffer(0)]],
  device const int* colPtr [[buffer(1)]],
  device const float* valPtr [[buffer(2)]],
  device const float* x[[buffer(3)]],
  device float* y[[buffer(4)]],
  uint row[[thread_position_in_grid]]
){

  float sum = 0.0f;
  for(int k = rowPtr[row]; k < rowPtr[row+1]; k++){
    sum += valPtr[k] * x[colPtr[k]];
  }

  y[row] = sum;
}

kernel void spmvELL(
    device const int* ellColIdx [[buffer(0)]],
    device const float* ellValues [[buffer(1)]],

    device const int* overflowRowPtr [[buffer(2)]],
    device const int* overflowColIdx [[buffer(3)]],
    device const float* overflowValues [[buffer(4)]],

    device const float* x [[buffer(5)]],
    device float* y [[buffer(6)]],

    constant uint& nRows [[buffer(7)]],
    constant uint& ellWidth [[buffer(8)]],

    uint row [[thread_position_in_grid]])
{
    if (row >= nRows)
    {
        return;
    }

    float sum = 0.0f;

    // --------------------------------------------------------
    // Regular ELL part.
    //
    // Neighboring GPU threads read neighboring addresses.
    // --------------------------------------------------------

    for (uint slot = 0;
         slot < ellWidth;
         ++slot)
    {
        const uint index =
            slot * nRows + row;

        const int column =
            ellColIdx[index];

        sum +=
            ellValues[index]
            * x[column];
    }

    // --------------------------------------------------------
    // Rare overflow part.
    //
    // For rows without overflow:
    //
    // start == end
    //
    // so this loop executes zero times.
    // --------------------------------------------------------

    const int overflowStart =
        overflowRowPtr[row];

    const int overflowEnd =
        overflowRowPtr[row + 1];

    for (int k = overflowStart;
         k < overflowEnd;
         ++k)
    {
        sum +=
            overflowValues[k]
            * x[overflowColIdx[k]];
    }

    y[row] = sum;
}
