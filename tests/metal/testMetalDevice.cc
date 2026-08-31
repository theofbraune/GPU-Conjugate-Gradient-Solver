#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
int main(){


  MTL::Device* device = MTL::CreateSystemDefaultDevice();

  if(!device){
    std::cerr <<" No metal device found!"<<std::endl;
    return 1;
  }

  std::cout<<" Name of the apple device is "<<device->name()->utf8String()<<std::endl;

  device->release();

  return 0;
}
