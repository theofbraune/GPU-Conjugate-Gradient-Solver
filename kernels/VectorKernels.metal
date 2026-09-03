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
