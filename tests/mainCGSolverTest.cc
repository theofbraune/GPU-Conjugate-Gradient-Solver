#include <GPUSolver/CGSolver.h>
#include <GPUSolver/Preconditioners/IdentityPreconditioner.h>
#include <GPUSolver/metal/MetalBackend.h>
#include <GPUSolver/metal/MetalContext.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>

#include <iostream>
#include <stdexcept>
#include <string>


int main(int argc, char* argv[])
{
    // --------------------------------------------------------
    // 1. Pick mesh.
    // --------------------------------------------------------

    const std::string projectSource =
        std::string(PROJECT_SOURCE_DIR);

    std::string meshPath;

    if (argc > 1)
    {
        meshPath =
            std::string(argv[1]);
    }
    else
    {
        meshPath =
            projectSource
            + "/data/surfaceMeshes/david140k.obj";
    }


    // --------------------------------------------------------
    // 2. Load triangle mesh.
    // --------------------------------------------------------

    Eigen::MatrixXf V;
    Eigen::MatrixXi F;

    if (!igl::read_triangle_mesh(
            meshPath,
            V,
            F))
    {
        throw std::runtime_error(
            "Could not load triangle mesh."
        );
    }

    std::cout
        << "Rows of V: "
        << V.rows()
        << " x "
        << V.cols()
        << '\n';

    std::cout
        << "Rows of F: "
        << F.rows()
        << " x "
        << F.cols()
        << "\n\n";


    // --------------------------------------------------------
    // 3. Construct cotangent Laplacian.
    // --------------------------------------------------------

    Eigen::SparseMatrix<float> columnMajorLaplacian;

    igl::cotmatrix(
        V,
        F,
        columnMajorLaplacian
    );


    // CGSolver expects row-major sparse storage.
    using Matrix =
        gpuSolver::CGSolver::Matrix;

    Matrix A =
        columnMajorLaplacian;

    A.makeCompressed();


    std::cout
        << "Matrix statistics\n"
        << "-----------------\n"
        << "Rows:      "
        << A.rows()
        << '\n'
        << "Cols:      "
        << A.cols()
        << '\n'
        << "NNZ:       "
        << A.nonZeros()
        << '\n'
        << "NNZ / row: "
        << static_cast<double>(
               A.nonZeros()
           )
           / static_cast<double>(
               A.rows()
           )
        << "\n\n";


    // --------------------------------------------------------
    // 4. Create GPU infrastructure.
    // --------------------------------------------------------

    gpuSolver::MetalContext context;

    gpuSolver::MetalBackend backend(
        context
    );


    // --------------------------------------------------------
    // 5. Create first preconditioner.
    //
    // Identity means:
    //
    //     M^-1 r = r
    //
    // --------------------------------------------------------

    gpuSolver::IdentityPreconditioner preconditioner;


    // --------------------------------------------------------
    // 6. Construct CG solver.
    //
    // We do NOT solve anything yet.
    // This only tests the object/data setup.
    // --------------------------------------------------------

    gpuSolver::CGSolver solver(
        backend,
        preconditioner,
        A
    );


    solver.setMaxIterations(
        1000
    );

    solver.setTolerance(
        1e-6f
    );


    std::cout
        << "CG solver constructed successfully.\n";

    std::cout
        << "Maximum iterations: "
        << solver.maxIterations()
        << '\n';

    std::cout
        << "Tolerance: "
        << solver.tolerance()
        << '\n';


    return 0;
}
