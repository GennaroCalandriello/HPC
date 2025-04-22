// pod_compute.cpp
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;

// Read snapshot matrix from CSV file into Eigen::MatrixXd
MatrixXd loadSnapshotCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }
    file.close();

    if (data.empty()) {
        std::cerr << "No data loaded from: " << filename << std::endl;
        exit(1);
    }

    const int rows = data.size();
    const int cols = data[0].size();
    MatrixXd S(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            S(i, j) = data[i][j];

    return S.transpose(); // Return (dofs x snapshots)
}

int main() {
    std::string inputFile = "snapshots.csv";
    MatrixXd S = loadSnapshotCSV(inputFile);
    std::cout << "Loaded snapshot matrix of size: " << S.rows() << " x " << S.cols() << "\n";

    // Optionally center the data
    VectorXd mean = S.rowwise().mean();
    MatrixXd S_centered = S.colwise() - mean;
    MatrixXd SST = S_centered * S_centered.transpose(); // Covariance matrix

    // Perform POD via SVD
    JacobiSVD<MatrixXd> svd(SST, ComputeThinU);
    VectorXd singularValues = svd.singularValues();
    MatrixXd POD_modes = svd.matrixU();

    std::cout << "Computed " << POD_modes.cols() << " POD modes." << std::endl;
    std::cout << "First 10 singular values: \n" << singularValues.head(10).transpose() << "\n";

    // Save singular values
    std::ofstream out("singular_values.txt");
    for (int i = 0; i < singularValues.size(); ++i)
        out << singularValues(i) << "\n";
    out.close();

    // Save first r POD modes
    int r = 10; // can be adjusted
    std::ofstream modeFile("pod_modes.txt");
    for (int i = 0; i < POD_modes.rows(); ++i) {
        for (int j = 0; j < std::min(r, (int)POD_modes.cols()); ++j) {
            modeFile << POD_modes(i, j);
            if (j < r - 1) modeFile << ",";
        }
        modeFile << "\n";
    }
    modeFile.close();

    std::cout << "Saved first " << r << " POD modes and singular values.\n";
    return 0;
}
