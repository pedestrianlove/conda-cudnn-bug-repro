#include <cudnn.h>
#include <stdio.h>

int main() {
  cudnnHandle_t handle;
  cudnnStatus_t status = cudnnCreate(&handle);

  if (status != CUDNN_STATUS_SUCCESS) {
    printf("CUDNN initialization failed: %s\n", cudnnGetErrorString(status));
    return 1;
  }

  printf("CUDNN initialized successfully.\n");

  // Clean up
  cudnnDestroy(handle);
  return 0;
}
