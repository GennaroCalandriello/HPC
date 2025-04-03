#include <Eigen/Dense>
using namespace Eigen;

if (!snapshots.empty()) {
    const int N = 2 * dim * dim;
    const int M = snapshots.size();
    MatrixXd S(N, M);

    for (int i = 0; i < M; i++)
        S.col(i) = Map<VectorXd>(snapshots[i].data(), N);

    // Optionally center the data
    VectorXd mean = S.rowwise().mean();
    MatrixXd S_centered = S.colwise() - mean;

    // Perform POD via SVD
    JacobiSVD<MatrixXd> svd(S_centered, ComputeThinU);h
    MatrixXd U = svd.matrixU();             // Spatial modes
    VectorXd sigma = svd.singularValues();  // Energy content

    // Truncate to r modes
    int r = 10;
    MatrixXd POD_basis = U.leftCols(r); // shape: N × r

    std::cout << "POD basis shape: " << POD_basis.rows() << " x " << POD_basis.cols() << "\n";
    std::cout << "Energy captured: " << sigma.head(r).array().square().sum() / sigma.array().square().sum() << "\n";

    // Save POD_basis for offline ROM projection or transfer to device
}


static std::vector<std::vector<float>> snapshots; // store flattened velocity fields

if (framecount % SNAPSHOT_INTERVAL == 0 && snapshots.size() < MAX_SNAPSHOTS) {
    std::vector<float> snapshot(2 * dim * dim); // Vector2f = 2 components
    cudaMemcpy(snapshot.data(), dev_u, 2 * dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
    snapshots.push_back(snapshot);
}
