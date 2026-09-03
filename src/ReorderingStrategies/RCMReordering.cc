#include <GPUSolver/ReorderingStrategies/RCMReordering.h>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>


namespace gpuSolver {

void RCMReordering::compute(std::size_t nRows, const int *rowPtr,
                            const int *colPtr, int *oldToNew,
                            int *newToOld) const {

  // first compute the degrees of the vertices, i.e the nonzero entries per row
  int *degrees = new int[nRows];
  bool *visited = new bool[nRows];

  for (int idxRow = 0; idxRow < nRows; idxRow++) {

    int idxNext = rowPtr[idxRow + 1];
    int idxNow = rowPtr[idxRow];

    int deg = 0;
    for (int idxCol = idxNow; idxCol < idxNext; idxCol++) {
      if (colPtr[idxCol] != idxRow) {
        deg++;
      }
    }

    degrees[idxRow] = deg;
    visited[idxRow] = false;
  }
  // now do the bfs
  std::queue<int> queueForBFS;
  int numberOrdered = 0;
  std::vector<int> neighbors;

  int *cmOrdering = new int[nRows];

  while (numberOrdered < nRows) {
    // here get the new start vertex with lowed degree. Can be that there are
    // different connected components, thus cannot just take the first and
    // start.

    int idxStartVertex = -1;
    int minDegreeUnvisited = std::numeric_limits<int>::max();
    for (int idxRow = 0; idxRow < nRows; idxRow++) {
      if (!visited[idxRow]) {
        int deg = degrees[idxRow];
        if (deg < minDegreeUnvisited) {
          minDegreeUnvisited = deg;
          idxStartVertex = idxRow;
        }
      }
    }

    queueForBFS.emplace(idxStartVertex);
    // cmOrdering[numberOrdered] = idxStartVertex;
    // numberOrdered++;

    visited[idxStartVertex] = true;

    while (!queueForBFS.empty()) {
      int currentVertex = queueForBFS.front();
      queueForBFS.pop();
      cmOrdering[numberOrdered] = currentVertex;
      numberOrdered++;
      neighbors.clear();
      int rowLimitLeft = rowPtr[currentVertex];
      int rowLimit = rowPtr[currentVertex + 1];

      for (int idxCol =rowLimitLeft; idxCol < rowLimit; idxCol++) {
        int colNumber = colPtr[idxCol];
        if (colNumber != currentVertex) {
          neighbors.push_back(colNumber); // means this are the neighbors of the
                                          // vertex # current vertex
        }
      }

      // now sort them so that the number of the degree is ascending
      std::sort(neighbors.begin(), neighbors.end(),
                [degrees](int a, int b) { return degrees[a] < degrees[b]; });

      for (int neighbor : neighbors) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queueForBFS.push(neighbor);
        }
      }
    }
  }

  // nere now reverse the list for the
  for (std::size_t newIndex = 0; newIndex < nRows; ++newIndex) {
    const std::size_t reversedIndex = nRows - 1 - newIndex;

    newToOld[newIndex] = cmOrdering[reversedIndex];
  }

  for (std::size_t newIndex = 0; newIndex < nRows; ++newIndex) {
    const int oldIndex = newToOld[newIndex];

    oldToNew[oldIndex] = static_cast<int>(newIndex);
  }

  delete[] degrees;
  delete[] visited;
  delete[] cmOrdering;
}

} // namespace gpuSolver
