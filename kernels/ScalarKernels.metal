#include <metal_stdlib>

using namespace metal;

kernel void scalarSet(
    constant float& value [[buffer(0)]],
    device float* result  [[buffer(1)]])
{
    result[0] = value;
}


kernel void scalarCopy(
  device const float* value [[buffer(0)]],
  device float* output [[buffer(1)]]
){
  output[0] = value[0];
}


kernel void scalarDivide(
  device const float* numerator [[buffer(0)]],
  device const float* denominator [[buffer(1)]],
  device float* output [[buffer(2)]]
){
  output[0] = numerator[0]/denominator[0];
}


kernel void scalarMultiply(
  device const float* factor1 [[buffer(0)]],
  device const float* factor2 [[buffer(1)]],
  device float* output [[buffer(2)]]
){
  output[0] = factor1[0]*factor2[0];
}


kernel void scalarNegate(
  device const float* input[[buffer(0)]],
  device float* output [[buffer(1)]]
){
  output[0] = -input[0];
}


kernel void scalarSqrt(
  device const float* input[[buffer(0)]],
  device float* output [[buffer(1)]]
){
  output[0] = sqrt(input[0]);
}


