
#include <metal_stdlib>


kernel void invert3x3BlockDiagonal(
  device const float* inverseBlocksFlattened [[buffer(0)]],
  device const float* input [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant uint& numberOfBlocks              [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
){

 if (tid >= numberOfBlocks) {
        return;
    }

  const uint vectorBase = 3*tid;
  const uint matrixBase = 9*tid;

  float in0 = input[vectorBase];
  float in1 = input[vectorBase+1];
  float in2 = input[vectorBase+2];

  float a00 = inverseBlocksFlattened[matrixBase];
  float a01 = inverseBlocksFlattened[matrixBase+1];
  float a02 = inverseBlocksFlattened[matrixBase+2];

  float out0 = a00*in0 + a01*in1 + a02*in2;


  float a10 = inverseBlocksFlattened[matrixBase+3];
  float a11 = inverseBlocksFlattened[matrixBase+4];
  float a12 = inverseBlocksFlattened[matrixBase+5];

  float out1 = a10*in0 + a11*in1 + a12*in2;


  float a20 = inverseBlocksFlattened[matrixBase+6];
  float a21 = inverseBlocksFlattened[matrixBase+7];
  float a22 = inverseBlocksFlattened[matrixBase+8];

  float out2 = a20*in0 + a21*in1 + a22*in2;

  output[vectorBase] = out0;
  output[vectorBase+1] = out1;
  output[vectorBase+2] = out2;


}
