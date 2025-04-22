int main() {
  // 1) Parametri
  const int M = MAX_FRAMES / SNAPSHOT_INTERVAL;
  const int dim = DIM;
  const int gridPoints = dim * dim;

  std::cout << "Computing per‑mode differentials for M=" << M << "\n";

  // 2) Carico la raw matrix N×M
  const int N = 2 * gridPoints;
  std::vector<std::vector<double>> raw;
  loadMatrix("resultsdata/pod_modes.txt", raw);
  if ((int)raw.size() != N || (int)raw[0].size() != M) {
    std::cerr << "Dimensioni errate: " << raw.size() << "×" << raw[0].size()
              << "\n";
    return 1;
  }

  // 3) Pre‐allocazione host‐side per i risultati
  //    Tutte e M le modalità
  std::vector<std::vector<Vector2f>> all_phi(M), all_divphi(M), all_gradchi(M),
      all_divgradphi(M), all_divoutprodphi(M);

  // 4) Allocazione device‐side (una volta sola)
  Vector2f *d_mode, *d_divphi, *d_gradchi, *d_divgradphi, *d_divoutprodphi;
  CHECK_CUDA(cudaMalloc(&d_mode, gridPoints * sizeof(Vector2f)));
  CHECK_CUDA(cudaMalloc(&d_divphi, gridPoints * sizeof(Vector2f)));
  CHECK_CUDA(cudaMalloc(&d_gradchi, gridPoints * sizeof(Vector2f)));
  CHECK_CUDA(cudaMalloc(&d_divgradphi, gridPoints * sizeof(Vector2f)));
  CHECK_CUDA(cudaMalloc(&d_divoutprodphi, gridPoints * sizeof(Vector2f)));

  // 5) Loop su ogni mode m
  for (int m = 0; m < M; ++m) {
    // 5a) Costruisco in host il vettore mode[m] di lunghezza gridPoints
    std::vector<Vector2f> h_mode(gridPoints);
    for (int k = 0; k < gridPoints; ++k) {
      float x = float(raw[k][m]);
      float y = float(raw[k + gridPoints][m]);
      h_mode[k] = Vector2f(x, y);
    }

    // 5b) Copio su GPU
    CHECK_CUDA(cudaMemcpy(d_mode, h_mode.data(), gridPoints * sizeof(Vector2f),
                          cudaMemcpyHostToDevice));

    // 5c) Chiamo il kernel che riempie d_divphi, d_gradchi, d_divgradphi,
    // d_divoutprodphi
    //    firma: galerkin_kernel(d_mode, d_divphi, d_gradchi, d_divgradphi,
    //    d_divoutprodphi, /*...*/);
    dim3 threads(16, 16), blocks((dim + 15) / 16, (dim + 15) / 16);
    galerkin_kernel<<<blocks, threads>>>(
        d_mode, d_divphi, d_gradchi, d_divgradphi, d_divoutprodphi,
        /* altri param: */ /* d_p, d_c, rdx, timestep, dim */
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5d) Copio indietro su host i risultati in vettori temporanei
    std::vector<Vector2f> h_divphi(gridPoints), h_gradchi(gridPoints),
        h_divgradphi(gridPoints), h_divoutprodphi(gridPoints);
    CHECK_CUDA(cudaMemcpy(h_divphi.data(), d_divphi,
                          gridPoints * sizeof(Vector2f),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gradchi.data(), d_gradchi,
                          gridPoints * sizeof(Vector2f),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_divgradphi.data(), d_divgradphi,
                          gridPoints * sizeof(Vector2f),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_divoutprodphi.data(), d_divoutprodphi,
                          gridPoints * sizeof(Vector2f),
                          cudaMemcpyDeviceToHost));

    // 5e) Spingo nei vettori all_ con push_back
    all_phi[m] = std::move(h_mode);
    all_divphi[m] = std::move(h_divphi);
    all_gradchi[m] = std::move(h_gradchi);
    all_divgradphi[m] = std::move(h_divgradphi);
    all_divoutprodphi[m] = std::move(h_divoutprodphi);
  }

  // 6) Pulizia device
  cudaFree(d_mode);
  cudaFree(d_divphi);
  cudaFree(d_gradchi);
  cudaFree(d_divgradphi);
  cudaFree(d_divoutprodphi);

  std::cout << "Fatte tutte le M modalità. Lunghezza all_phi: "
            << all_phi.size() << "\n";
  // ora all_phi[i] contiene il Vector2f[gridPoints] per la modalità i
  // idem per all_divphi, all_gradchi, ...

  return 0;
}