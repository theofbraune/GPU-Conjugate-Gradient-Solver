#include <metal_stdlib>


kernel void gaussSeidelColor(
    device const int* rowPtr        [[buffer(0)]],
    device const int* colPtr        [[buffer(1)]],
    device const float* values      [[buffer(2)]],
    device const int* colorVertices [[buffer(3)]],
    device const float* rhs         [[buffer(4)]],
    device float* x                 [[buffer(5)]],
    constant uint& colorStart       [[buffer(6)]],
    constant float& omega           [[buffer(7)]],
    constant uint& colorCount [[buffer(8)]],
    uint tid [[thread_position_in_grid]]){

  const uint row = colorVertices[colorStart+tid];

  float diagonal = 0.f; 
  float sum = rhs[row];

  for(int entry = rowPtr[row]; entry < rowPtr[row+1]; entry++){

    const uint col = colPtr[entry];
    const float val =values[entry];

    if(col==row){
      diagonal = val;
    }else{
      sum -= val * x[col];
    }
  }
  float newVal = sum / diagonal;
  x[row] = (1.0 - omega) * x[row] + omega * newVal;


}
