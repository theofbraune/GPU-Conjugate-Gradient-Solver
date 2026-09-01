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
