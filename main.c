#include <stdio.h>
#include <utils.h>

int main() {
  cl_int status;

  cl_device_id dev_id;
  status = find_device(&dev_id);
  if (status != CL_SUCCESS) {
    printf("failed to find device !\n");
    return EXIT_FAILURE;
  }

  size_t val_size;
  status = clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, NULL, &val_size);
  if (status != CL_SUCCESS) {
    printf("failed to get device name !\n");
    return EXIT_FAILURE;
  }

  void *dev_name = malloc(val_size);
  status = clGetDeviceInfo(dev_id, CL_DEVICE_NAME, val_size, dev_name, NULL);
  if (status != CL_SUCCESS) {
    printf("failed to get device name !\n");
    return EXIT_FAILURE;
  }

  printf("running on %s\n", (char *)dev_name);

  cl_context ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);
  if (status != CL_SUCCESS) {
    printf("failed to create context !\n");
    return EXIT_FAILURE;
  }

  cl_command_queue c_queue =
      clCreateCommandQueueWithProperties(ctx, dev_id, NULL, &status);
  if (status != CL_SUCCESS) {
    printf("failed to create command queue !\n");
    return EXIT_FAILURE;
  }

  cl_program prgm;
  status = build_kernel(ctx, dev_id, "kernel.cl", &prgm);
  if (status != CL_SUCCESS) {
    printf("failed to compile kernel !\n");
    return EXIT_FAILURE;
  }

  // releasing all OpenCL resources !
  clReleaseProgram(prgm);
  clReleaseCommandQueue(c_queue);
  clReleaseContext(ctx);
  clReleaseDevice(dev_id);

  free(dev_name);

  return EXIT_SUCCESS;
}
