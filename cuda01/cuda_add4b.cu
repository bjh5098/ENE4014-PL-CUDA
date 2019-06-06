#include <iostream>
#include <math.h>
#include <stdio.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;
  //for (int i = index; i < n; i += stride) 
  //  y[i] = x[i] + y[i];
  if (index < n)
    y[index] = x[index] + y[index];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;
  float *d_x, *d_y;

  x = new float[N];
  y = new float[N];

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc((void**)&d_x, N*sizeof(float));
  cudaMalloc((void**)&d_y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  std::cout << "numBlocks:" << numBlocks << std::endl;
  for (int i=0; i<100000; i++) {
      add<<<numBlocks, blockSize>>>(N, d_x, d_y);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();
  }

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  
  free(x);
  free(y);
  return 0;
}
