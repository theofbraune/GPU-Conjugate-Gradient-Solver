#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Preconditioners/IdentityPreconditioner.h>

namespace gpuSolver{

void IdentityPreconditioner::apply(
    Backend& backend, 
    BackendEncoder& encoder, 
    const DeviceVector& input,
    DeviceVector& output
    ){


  backend.encodeCopy(encoder, input, output);
  

}

}
