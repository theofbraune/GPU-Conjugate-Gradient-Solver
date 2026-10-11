#include <GPUSolver/Backend.h>
#include <GPUSolver/BackendEncoder.h>
#include <GPUSolver/DeviceVector.h>
#include <GPUSolver/Preconditioners/JacobiPreconditioner.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace gpuSolver {

JacobiPreconditioner::JacobiPreconditioner(){
  this->diagonalValuesInvDevice = nullptr;
}

void JacobiPreconditioner::initialize(
    Backend &backend,
    const HostCSRMatrix &hostMatrix,
    const DeviceCSRMatrix &deviceMatrix)
{
    const std::size_t n =
        hostMatrix.rows();

    std::vector<float> inverseDiagonal(n);

    const int *rowPtr =
        hostMatrix.activeRowPtr();

    const int *colPtr =
        hostMatrix.activeColPtr();

    const float *values =
        hostMatrix.activeValPtr();

    for (std::size_t row = 0; row < n; ++row)
    {
        bool diagonalFound = false;

        for (std::size_t entry = rowPtr[row];
             entry < rowPtr[row + 1];
             ++entry)
        {
            if (colPtr[entry] == row)
            {
                const float diagonalValue =
                    values[entry];

                if (std::abs(diagonalValue) < 1e-12f)
                {
                    throw std::runtime_error(
                        "JacobiPreconditioner::initialize: "
                        "zero diagonal entry."
                    );
                }

                inverseDiagonal[row] =
                    1.0f / diagonalValue;

                diagonalFound = true;
                break;
            }
        }

        if (!diagonalFound)
        {
            throw std::runtime_error(
                "JacobiPreconditioner::initialize: "
                "missing diagonal entry."
            );
        }
    }

    delete diagonalValuesInvDevice;

    diagonalValuesInvDevice =
        backend.createVector(
            n,
            inverseDiagonal.data()
        );

    diagonalSize_ = n;

    (void)deviceMatrix;
}


void JacobiPreconditioner::apply(Backend &backend, BackendEncoder &encoder,
                                      const DeviceVector &input,
                                      DeviceVector &output) {
  if (diagonalValuesInvDevice == nullptr) {
    throw std::runtime_error(
        "JacobiPreconditioner::apply: preconditioner has no diagonal "
        "set.");
  }

  if (input.size() != diagonalSize_|| output.size() !=diagonalSize_) {
    throw std::runtime_error(
        "JacobiPreconditioner::apply: input/output size does not match "
        "the preconditioner's diagonal size.");
  }
  backend.encodeCopy(encoder, input, output);
  backend.encodeScaleVector(encoder, *this->diagonalValuesInvDevice, output);
}


JacobiPreconditioner::~JacobiPreconditioner() {
  delete diagonalValuesInvDevice;
}

} // namespace gpuSolver
