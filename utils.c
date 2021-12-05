#include <utils.h>

cl_int find_device(cl_device_id *device_id) {
  // just reset all bytes for safety !
  memset(device_id, 0, sizeof(cl_device_id));

  cl_int status;

  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS) {
    return status;
  }

  if (num_platforms == 0) {
    return CL_DEVICE_NOT_FOUND;
  }

  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (status != CL_SUCCESS) {
    return status;
  }

  // preferred device is either CPU, GPU
  cl_device_type dev_type = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU;

  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_uint num_devices;
    status = clGetDeviceIDs(*(platforms + i), dev_type, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
      continue;
    }

    if (num_devices == 0) {
      continue;
    }

    cl_device_id *devices = malloc(sizeof(cl_device_id) * num_devices);
    status =
        clGetDeviceIDs(*(platforms + i), dev_type, num_devices, devices, NULL);
    if (status != CL_SUCCESS) {
      free(devices);
      continue;
    }

    *device_id = *devices;
    free(devices);
    return CL_SUCCESS;
  }

  free(platforms);

  return CL_DEVICE_NOT_FOUND;
}

cl_int build_kernel(cl_context ctx, cl_device_id dev_id, char *kernel,
                    cl_program *prgm) {
  cl_int status;

  FILE *fd = fopen(kernel, "r");
  fseek(fd, 0, SEEK_END);
  size_t size = ftell(fd);
  fseek(fd, 0, SEEK_SET);

  char *kernel_src = malloc(sizeof(char) * size);
  size_t n = fread(kernel_src, sizeof(char), size, fd);

  assert(n == size);
  fclose(fd);

  cl_program prgm_ = clCreateProgramWithSource(
      ctx, 1, (const char **)&kernel_src, &size, &status);
  if (status != CL_SUCCESS) {
    return status;
  }

  status = clBuildProgram(prgm_, 1, &dev_id, NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    return status;
  }

  *prgm = prgm_;
  free(kernel_src);

  return CL_SUCCESS;
}
