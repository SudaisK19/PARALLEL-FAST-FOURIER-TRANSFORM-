#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cmath>

using namespace std;
using Complex = complex<double>;
using CVector = vector<Complex>;

#define ROOT 0

// Function to perform FFT on 1D data
CVector FFT(const CVector& a) {
    int n = a.size();
    if (n == 1) return a;

    Complex wn = polar(1.0, 2 * M_PI / n); // Primitive nth root of unity
    Complex w = 1;

    CVector a_even(n / 2), a_odd(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }

    CVector y_even = FFT(a_even);
    CVector y_odd = FFT(a_odd);

    CVector y(n);
    for (int k = 0; k < n / 2; ++k) {
        y[k] = y_even[k] + w * y_odd[k];
        y[k + n / 2] = y_even[k] - w * y_odd[k];
        w *= wn;
    }

    return y;
}

// Function to transpose a 2D vector
void transpose(vector<CVector>& data) {
    vector<CVector> transposed(data[0].size(), CVector(data.size()));
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            transposed[j][i] = data[i][j];
        }
    }
    data = transposed;
}

int main(int argc, char** argv) {
    int num_threads;
    cout << "Enter the number of threads: ";
    cin >> num_threads;
    MPI_Init(&argc, &argv);

    int rank, size=num_threads;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;
    cv::Mat image;
    vector<CVector> image_data;

    if (rank == ROOT) {
        // Load the image on the root process
        image = cv::imread("/mnt/c/Users/S Link Solutions/Downloads/PDCPRJECT_FFT/PDCPRJECT_FFT/512.jpg", cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Could not open or find the image." << endl;
            MPI_Finalize();
            return -1;
        }

        rows = image.rows;
        cols = image.cols;

        // Convert image to 2D vector
        image_data.resize(rows, CVector(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                image_data[i][j] = image.at<uchar>(i, j);
            }
        }
        cout << "ROOT: Loaded and initialized image of size " << rows << "x" << cols << endl;
    }
    
    double start_time = MPI_Wtime(); // Start timer
    // Broadcast dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Divide rows among processes
    int rows_per_process = rows / size;
    int remaining_rows = rows % size;

    int local_rows = rows_per_process + (rank < remaining_rows ? 1 : 0);
    vector<CVector> local_data(local_rows, CVector(cols));

    // Scatter rows to all processes
    vector<int> send_counts(size), displacements(size);
    if (rank == ROOT) {
        for (int i = 0; i < size; ++i) {
            send_counts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * cols;
            displacements[i] = (i > 0) ? displacements[i - 1] + send_counts[i - 1] : 0;
        }
        cout << "ROOT: Send counts and displacements calculated for scatter." << endl;
        cout << "Send counts: ";
        for (int i = 0; i < size; ++i) cout << send_counts[i] << " ";
        cout << endl;
        cout << "Displacements: ";
        for (int i = 0; i < size; ++i) cout << displacements[i] << " ";
        cout << endl;
    }

    vector<double> flat_image;
    if (rank == ROOT) {
        flat_image.resize(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                flat_image[i * cols + j] = real(image_data[i][j]);
            }
        }
        cout << "Root process initialized flat_image with size: " << flat_image.size() << endl;
    }

    vector<double> local_flat_data(local_rows * cols);
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Rank " << rank << ": local_rows = " << local_rows << ", expected data size = " << send_counts[rank] << endl;
    MPI_Scatterv(flat_image.data(), send_counts.data(), displacements.data(), MPI_DOUBLE,
                 local_flat_data.data(), local_rows * cols, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    cout << "Rank " << rank << " completed MPI_Scatterv." << endl;

    // Reconstruct local 2D data
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            local_data[i][j] = local_flat_data[i * cols + j];
        }
    }

    // Perform FFT on each row locally
    for (int i = 0; i < local_rows; ++i) {
        local_data[i] = FFT(local_data[i]);
    }

    // Flatten local data for gathering
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            local_flat_data[i * cols + j] = real(local_data[i][j]);
        }
    }

    vector<double> gathered_flat_image;
    if(rank==ROOT){
       gathered_flat_image.resize(rows * cols);
    }   
    MPI_Gatherv(local_flat_data.data(), local_rows * cols, MPI_DOUBLE,
                gathered_flat_image.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    cout << "Rank " << rank << ": Completed MPI_Gatherv." << endl;

    vector<CVector> gathered_image;
    if (rank == ROOT) {
        // Reconstruct 2D data from gathered data
        gathered_image.resize(rows, CVector(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                gathered_image[i][j] = gathered_flat_image[i * cols + j];
            }
        }

        // Perform column-wise FFT
        transpose(gathered_image);
    }   
    // Calculate columns per process
    int cols_per_process = cols / size;
    int remaining_cols = cols % size;
    int local_cols = cols_per_process + (rank < remaining_cols ? 1 : 0);
    // Prepare for scatter
    vector<int> column_send_counts(size),column_displacements(size);
    if (rank == ROOT) {
        for (int i = 0; i < size; ++i) {
             column_send_counts[i] = (cols_per_process + (i < remaining_cols ? 1 : 0)) * rows;
             column_displacements[i] = (i > 0) ? column_displacements[i - 1] + column_send_counts[i - 1] : 0;
        }
     }
     vector<double> flat_columns;
     if(rank==ROOT){   
        // Flatten columns for scattering (ROOT only)
        flat_columns.resize(rows*cols);
        for (int i = 0; i < rows; ++i) {
             for (int j = 0; j < cols; ++j) {
                  flat_columns[i * cols + j] = real(gathered_image[i][j]);
             }
        }
     }
     // Local storage for scattered columns
     vector<double> local_flat_columns(local_cols * rows);

     // Scatter the columns
     MPI_Scatterv(flat_columns.data(), column_send_counts.data(), column_displacements.data(),
             MPI_DOUBLE, local_flat_columns.data(), local_cols * rows, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

     // Reconstruct local column data
     vector<CVector> local_columns(local_cols, CVector(rows));
     for (int i = 0; i < local_cols; ++i) {
          for (int j = 0; j < rows; ++j) {
               local_columns[i][j] = local_flat_columns[i * rows + j];
          }
     }
     // Perform FFT on each local column
     for (int i = 0; i < local_cols; ++i) {
          local_columns[i] = FFT(local_columns[i]);
     } 
     // Flatten processed columns for gathering
     for (int i = 0; i < local_cols; ++i) {
          for (int j = 0; j < rows; ++j) {
               local_flat_columns[i * rows + j] = real(local_columns[i][j]);
          }
     }
     // Gather processed columns back to the root process
     vector<double> gathered_flat_columns;
     if (rank == ROOT) {
         gathered_flat_columns.resize(rows * cols);
     }
     MPI_Gatherv(local_flat_columns.data(), local_cols * rows, MPI_DOUBLE,
            gathered_flat_columns.data(), column_send_counts.data(), column_displacements.data(),
            MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
     
     if (rank == ROOT) {
         // Reconstruct final image
         vector<CVector> final_image(cols, CVector(rows));
         for (int i = 0; i < cols; ++i) {
             for (int j = 0; j < rows; ++j) {
                 final_image[i][j] = gathered_flat_columns[i * rows + j];
             }
         }

         // Transpose back to original orientation
         transpose(final_image); // Restore to rows x cols
           
         // Compute magnitude and apply logarithmic scaling
         cv::Mat magnitudeImage(rows, cols, CV_64F);
         for (int i = 0; i < rows; ++i) {
             for (int j = 0; j < cols; ++j) {
                 double mag = abs(final_image[i][j]);
                 magnitudeImage.at<double>(i, j) = mag;
             }
          }
        
          magnitudeImage += cv::Scalar::all(1);
          cv::log(magnitudeImage, magnitudeImage);

          // Normalize for display
          cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);
          cv::Mat displayMagnitude;
          magnitudeImage.convertTo(displayMagnitude, CV_8UC1,255);
       
          double end_time = MPI_Wtime(); // End timer
          cout << "Total Parallel Execution Time: " << end_time - start_time << " seconds." << endl;
          // Display the image
          cv::imshow("Original Image", image);
          cv::imshow("FFT Magnitude", magnitudeImage);
          cv::waitKey(0);
    }

    MPI_Finalize();
    return 0;
}

