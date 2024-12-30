#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <time.h>
#include <chrono>

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
void transpose(vector<CVector> &data) {
    vector<CVector> transposed(data[0].size(), CVector(data.size()));
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[0].size(); j++) {
            transposed[j][i] = data[i][j];
        }
    }
    data = transposed;
}

int main() {
	using namespace std::chrono;
	auto start = high_resolution_clock::now();
    // Load an image in grayscale
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/32.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/64.jpg", cv::IMREAD_GRAYSCALE);
//	cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/128.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/256.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/512.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/1024.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/2048.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat image = cv::imread("/mnt/c/Users/sooraj/OneDrive/Pictures/new/updated/4096.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image." << endl;
        return -1;
    }

    // Convert the image to a 2D vector of complex numbers (for FFT)
    int rows = image.rows;
    int cols = image.cols;
    vector<CVector> image_data(rows, CVector(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            image_data[i][j] = image.at<uchar>(i, j);
        }
    }

    // Apply FFT to each row
    for (int i = 0; i < rows; ++i) {
        image_data[i] = FFT(image_data[i]);
    }

    // Apply FFT to each column (transpose first)
   // Transpose the image data
    transpose(image_data);

    // Apply FFT to each column (now rows after transpose)
    for (int i = 0; i < cols; ++i) {
        image_data[i] = FFT(image_data[i]);
    }

    // Transpose back to original orientation
    transpose(image_data);

    // Calculate magnitude and scale
    cv::Mat magnitudeImage(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double mag = abs(image_data[i][j]);
            magnitudeImage.at<double>(i, j) = mag;
        }
    }

    // Switch to logarithmic scale to enhance visibility
    magnitudeImage += cv::Scalar::all(1);
    cv::log(magnitudeImage, magnitudeImage);

    // Normalize the magnitude image to [0,1], then scale to [0,255] and convert to 8-bit
    cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);
    cv::Mat displayMagnitude;
    magnitudeImage.convertTo(displayMagnitude, CV_8UC1, 255);
    cout << fixed << setprecision(10);// Set fixed-point notation with 10 decimal places
	 // Print the complex values after Fourier Transform
    for (const auto& row : image_data) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
	auto stop = high_resolution_clock::now();
	duration<double> time_taken = stop - start;
    cout<<"Total Execution Time: "<< time_taken.count() <<" seconds";
    // Display the original and the FFT magnitude image
    cv::imshow("Original Image", image);
    cv::imshow("FFT Magnitude", magnitudeImage);
    cv::waitKey(0);

    return 0;
}
