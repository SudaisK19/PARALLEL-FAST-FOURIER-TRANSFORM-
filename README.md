# Parallel Implementation of FFT for Image Processing

This repository showcases the **Parallel and Distributed Computing (PDC)** project, which focuses on optimizing the Fast Fourier Transform (FFT) for image processing using **OpenMP** and **MPI**. The project highlights the benefits of parallelization in shared and distributed-memory environments.

---

## Project Overview

The goal of this project is to enhance the computational efficiency of the FFT algorithm, which is widely used in signal and image processing. By leveraging parallel programming techniques, the project demonstrates how to handle large datasets effectively.

---

## Features

1. **Serial Implementation**:
   - Baseline performance measurement for sequential execution.
   - Acts as a reference for evaluating parallel efficiency.

2. **Parallel Implementation (OpenMP)**:
   - Uses OpenMP directives to parallelize tasks in a shared-memory environment.
   - Optimizations include `#pragma omp parallel`, loop collapsing, and task division.

3. **Parallel Implementation (MPI)**:
   - Distributes computations across multiple nodes using MPI.
   - Employs efficient communication mechanisms like `MPI_Bcast`, `MPI_Scatterv`, and `MPI_Gatherv`.

4. **Performance Metrics**:
   - Execution time for various image sizes (32x32 to 4096x4096).
   - Speedup and efficiency comparisons between OpenMP and MPI.

---

## Key Insights

- **OpenMP**:
  - Excels in smaller datasets with low communication overhead.
  - Ideal for shared-memory environments (e.g., multicore processors).

- **MPI**:
  - Handles large datasets efficiently in distributed-memory systems.
  - Better scalability for massive workloads.

- **Speedup Trends**:
  - Parallel benefits increase with image size.
  - OpenMP achieves higher speedup for smaller images, while MPI performs better for larger datasets.

---

## Graphical Results

This repository includes graphical comparisons of:
- Execution times (serial vs. parallel).
- Speedup and efficiency trends for OpenMP and MPI.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd parallel-fft-pdc
## For Openmp 
gcc -fopenmp fft_openmp.c -o fft_openmp
./fft_openmp
## For MPI
mpicc fft_mpi.c -o fft_mpi
mpirun -np <number_of_processes> ./fft_mpi

### Key Notes:
1. Replace `<repository-url>` with the actual GitHub URL of your repository.
2. Add any actual paths to code files or result images under the **Outputs** or **Graphical Results** sections.
3. Modify any part based on your finalized project details.

Let me know if you need further customization!


