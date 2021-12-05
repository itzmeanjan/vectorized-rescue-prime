#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

int main() {
  cl_int status;

  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS) {
    printf("failed to get number of platforms !\n");
    return EXIT_FAILURE;
  }

  if (num_platforms == 0) {
    printf("no OpenCL platform found !\n");
    return EXIT_SUCCESS;
  }

  printf("number of platforms: %u\n", num_platforms);

  cl_platform_id *platform_ids = malloc(sizeof(cl_platform_id) * num_platforms);
  status = clGetPlatformIDs(num_platforms, platform_ids, 0);
  if (status != CL_SUCCESS) {
    printf("failed to get platform ids !\n");
    return EXIT_FAILURE;
  }

  cl_uint num_devices;
  status = clGetDeviceIDs(*(platform_ids + 0), CL_DEVICE_TYPE_ALL, 0, NULL,
                          &num_devices);
  if (status != CL_SUCCESS) {
    printf("failed to get list of devices !\n");
    return EXIT_FAILURE;
  }

  if (num_devices == 0) {
    printf("no device found in platform !\n");
    return EXIT_SUCCESS;
  }

  printf("number of devices: %u\n", num_devices);

  cl_device_id *device_ids = malloc(sizeof(cl_device_id) * num_devices);
  status = clGetDeviceIDs(*(platform_ids + 0), CL_DEVICE_TYPE_ALL, num_devices,
                          device_ids, 0);
  if (status != CL_SUCCESS) {
    printf("failed to get device ids !\n");
    return EXIT_FAILURE;
  }

  size_t val_size;
  status =
      clGetDeviceInfo(*(device_ids + 0), CL_DEVICE_NAME, 0, NULL, &val_size);
  if (status != CL_SUCCESS) {
    printf("failed to get device name !\n");
    return EXIT_FAILURE;
  }

  void *device_name = malloc(val_size);
  status = clGetDeviceInfo(*(device_ids + 0), CL_DEVICE_NAME, val_size,
                           device_name, NULL);
  if (status != CL_SUCCESS) {
    printf("failed to get device name !\n");
    return EXIT_FAILURE;
  }

  printf("running on %s\n", (char *)device_name);

  free(device_name);
  free(device_ids);
  free(platform_ids);

  return 0;
}
