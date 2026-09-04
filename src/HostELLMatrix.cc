#include <GPUSolver/HostELLMatrix.h>
#include <GPUSolver/HostSparseMatrix.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace gpuSolver {

HostELLMatrix::HostELLMatrix(
    const HostCSRMatrix& csr,
    std::size_t ellWidth)
    : nRows_(csr.rows()),
      nCols_(csr.cols()),
      nnz_(csr.nnz()),
      ellWidth_(ellWidth),
      overflowNnz_(0),
      ellColIdx_(nullptr),
      ellValues_(nullptr),
      overflowRowPtr_(nullptr),
      overflowColIdx_(nullptr),
      overflowValues_(nullptr)
{
    if (nRows_ == 0 || nCols_ == 0)
    {
        throw std::runtime_error(
            "HostELLMatrix: matrix dimensions must be nonzero."
        );
    }

    if (ellWidth_ == 0)
    {
        throw std::runtime_error(
            "HostELLMatrix: ELL width must be nonzero."
        );
    }

    const int* rowPtr =
        csr.activeRowPtr();

    const int* colIdx =
        csr.activeColPtr();

    const float* values =
        csr.activeValPtr();

    if (rowPtr == nullptr ||
        colIdx == nullptr ||
        values == nullptr)
    {
        throw std::runtime_error(
            "HostELLMatrix: CSR storage pointers must not be null."
        );
    }

    // --------------------------------------------------------
    // 1. Determine how many entries do not fit into ELL.
    // --------------------------------------------------------

    for (std::size_t row = 0;
         row < nRows_;
         ++row)
    {
        const int rowLength =
            rowPtr[row + 1] - rowPtr[row];

        if (rowLength < 0)
        {
            throw std::runtime_error(
                "HostELLMatrix: invalid CSR row pointer."
            );
        }

        if (rowLength > static_cast<int>(ellWidth_))
        {
            overflowNnz_ +=
                static_cast<std::size_t>(
                    rowLength
                    - static_cast<int>(ellWidth_)
                );
        }
    }

    // --------------------------------------------------------
    // 2. Allocate ELL storage.
    //
    // Stored slot-major:
    //
    //     index = slot * nRows + row
    //
    // --------------------------------------------------------

    const std::size_t ellStorageSize =
        nRows_ * ellWidth_;

    ellColIdx_ =
        new int[ellStorageSize];

    ellValues_ =
        new float[ellStorageSize];

    // Padding entries.
    //
    // column = 0 is safe,
    // value = 0 means it contributes nothing.
    for (std::size_t i = 0;
         i < ellStorageSize;
         ++i)
    {
        ellColIdx_[i] = 0;
        ellValues_[i] = 0.0f;
    }

    // --------------------------------------------------------
    // 3. Allocate overflow CSR storage.
    // --------------------------------------------------------

    overflowRowPtr_ =
        new int[nRows_ + 1];

    overflowRowPtr_[0] = 0;

    if (overflowNnz_ > 0)
    {
        overflowColIdx_ =
            new int[overflowNnz_];

        overflowValues_ =
            new float[overflowNnz_];
    }

    // --------------------------------------------------------
    // 4. Convert active CSR -> ELL + overflow CSR.
    // --------------------------------------------------------

    std::size_t overflowPosition = 0;

    for (std::size_t row = 0;
         row < nRows_;
         ++row)
    {
        const int rowStart =
            rowPtr[row];

        const int rowEnd =
            rowPtr[row + 1];

        const int rowLength =
            rowEnd - rowStart;

        const std::size_t numberELL =
            std::min(
                ellWidth_,
                static_cast<std::size_t>(rowLength)
            );

        // ----------------------------------------------------
        // ELL part.
        // ----------------------------------------------------

        for (std::size_t slot = 0;
             slot < numberELL;
             ++slot)
        {
            const int csrIndex =
                rowStart + static_cast<int>(slot);

            const std::size_t ellIndex =
                slot * nRows_ + row;

            ellColIdx_[ellIndex] =
                colIdx[csrIndex];

            ellValues_[ellIndex] =
                values[csrIndex];
        }

        // ----------------------------------------------------
        // Overflow part.
        // ----------------------------------------------------

        for (int csrIndex =
                 rowStart + static_cast<int>(numberELL);
             csrIndex < rowEnd;
             ++csrIndex)
        {
            overflowColIdx_[overflowPosition] =
                colIdx[csrIndex];

            overflowValues_[overflowPosition] =
                values[csrIndex];

            ++overflowPosition;
        }

        overflowRowPtr_[row + 1] =
            static_cast<int>(overflowPosition);
    }

    if (overflowPosition != overflowNnz_)
    {
        throw std::runtime_error(
            "HostELLMatrix: inconsistent overflow size."
        );
    }
}


HostELLMatrix::~HostELLMatrix()
{
    delete[] ellColIdx_;
    delete[] ellValues_;

    delete[] overflowRowPtr_;
    delete[] overflowColIdx_;
    delete[] overflowValues_;
}


std::size_t HostELLMatrix::rows() const
{
    return nRows_;
}

std::size_t HostELLMatrix::cols() const
{
    return nCols_;
}

std::size_t HostELLMatrix::nnz() const
{
    return nnz_;
}

std::size_t HostELLMatrix::ellWidth() const
{
    return ellWidth_;
}

std::size_t HostELLMatrix::overflowNnz() const
{
    return overflowNnz_;
}

const int* HostELLMatrix::ellColIdx() const
{
    return ellColIdx_;
}

const float* HostELLMatrix::ellValues() const
{
    return ellValues_;
}

const int* HostELLMatrix::overflowRowPtr() const
{
    return overflowRowPtr_;
}

const int* HostELLMatrix::overflowColIdx() const
{
    return overflowColIdx_;
}

const float* HostELLMatrix::overflowValues() const
{
    return overflowValues_;
}

} // namespace gpuSolver


