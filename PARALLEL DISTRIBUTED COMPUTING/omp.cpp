#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <omp.h>

using namespace std;
using Complex = complex<double>;
using CVector = vector<Complex>;

// Implement the FFT algorithm
CVector FFT(const CVector& a) {
    int n = a.size();
    if (n == 1) {
        return a;
    }

    Complex wn = polar(1.0, 2 * M_PI / n); // Primitive nth root of unity
    Complex w = 1;

    CVector a_even(n / 2), a_odd(n / 2);
    #pragma omp parallel for
    for (int i = 0; i < n / 2; ++i) {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }

    CVector y_even, y_odd;
    #pragma omp parallel sections
    {
        #pragma omp section
        y_even = FFT(a_even);
        #pragma omp section
        y_odd = FFT(a_odd);
    }

    CVector y(n);
    for (int k = 0; k < n / 2; ++k) {
        y[k] = y_even[k] + w * y_odd[k];
        y[k + n / 2] = y_even[k] - w * y_odd[k];
        w *= wn; // Sequential dependency, cannot parallelize
    }

    return y;
}

// Function to transpose a 2D vector
void transpose(vector<CVector> &data) {
    vector<CVector> transposed(data[0].size(), CVector(data.size()));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            transposed[j][i] = data[i][j];
        }
    }
    data = transposed;
}

int main() {
    int num_threads;
    cout << "Enter the number of threads: ";
    cin >> num_threads;
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Load an image in grayscale (I/O time excluded from computation timing)
    cv::Mat image = cv::imread("/home/vboxuser/Downloads/512.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image." << endl;
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;
    vector<CVector> image_data(rows, CVector(cols));

    // Start timing after image input
    double start_parallel = omp_get_wtime();

    // Single parallel region for multiple loops
    #pragma omp parallel
    {
        // Convert image data to complex numbers
        #pragma omp for collapse(2)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                image_data[i][j] = image.at<uchar>(i, j);
            }
        }

        // Apply FFT to each row
        #pragma omp for
        for (int i = 0; i < rows; ++i) {
            image_data[i] = FFT(image_data[i]);
        }

        // Transpose the image data
        #pragma omp single
        transpose(image_data); // Transpose is done once in a single thread

        // Apply FFT to each column (now rows after transpose)
        #pragma omp for
        for (int i = 0; i < cols; ++i) {
            image_data[i] = FFT(image_data[i]);
        }

        // Transpose back to original orientation
        #pragma omp single
        transpose(image_data); // Transpose back is done once in a single thread
    }

    cv::Mat magnitudeImage(rows, cols, CV_64F);

    // Magnitude calculation and scaling
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double mag = abs(image_data[i][j]);
            magnitudeImage.at<double>(i, j) = mag;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            magnitudeImage.at<double>(i, j) = log(1 + magnitudeImage.at<double>(i, j));
        }
    }

    cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);
    cv::Mat displayMagnitude;
    magnitudeImage.convertTo(displayMagnitude, CV_8UC1, 255);

    double end_parallel = omp_get_wtime();
    cout << "Total Execution Time (excluding I/O): " << end_parallel - start_parallel << " seconds" << endl;

    // Display the original and the FFT magnitude image
    cv::imshow("Original Image", image);
    cv::imshow("FFT Magnitude", displayMagnitude);
    cv::waitKey(0);

    return 0;
}


