#include <utils.h>

cl_int
find_device(cl_device_id* device_id)
{
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

  cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * num_platforms);
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

    cl_device_id* devices = malloc(sizeof(cl_device_id) * num_devices);
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

cl_int
build_kernel(cl_context ctx,
             cl_device_id dev_id,
             char* kernel,
             cl_program* prgm)
{
  cl_int status;

  FILE* fd = fopen(kernel, "r");
  fseek(fd, 0, SEEK_END);
  size_t size = ftell(fd);
  fseek(fd, 0, SEEK_SET);

  char* kernel_src = malloc(sizeof(char) * size);
  size_t n = fread(kernel_src, sizeof(char), size, fd);

  assert(n == size);
  fclose(fd);

  cl_program prgm_ = clCreateProgramWithSource(
    ctx, 1, (const char**)&kernel_src, &size, &status);
  if (status != CL_SUCCESS) {
    return status;
  }

  *prgm = prgm_;
  free(kernel_src);

  status = clBuildProgram(*prgm, 1, &dev_id, "-cl-std=CL2.0 -w", NULL, NULL);
  if (status != CL_SUCCESS) {
    return status;
  }

  return CL_SUCCESS;
}

cl_int
show_build_log(cl_device_id dev_id, cl_program prgm)
{
  cl_int status;

  size_t log_size;
  status = clGetProgramBuildInfo(
    prgm, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (status != CL_SUCCESS) {
    return status;
  }

  void* log = malloc(log_size);
  status = clGetProgramBuildInfo(
    prgm, dev_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
  if (status != CL_SUCCESS) {
    free(log);
    return status;
  }

  printf("\nkernel build log:\n%s\n\n", (char*)log);
  free(log);

  return CL_SUCCESS;
}

void
random_field_elements(cl_ulong* in, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    *(in + i) = (cl_ulong)rand();
  }
}

cl_int
device_memory_base_address_alignment(cl_device_id dev_id,
                                     size_t* mem_base_addr_align)
{
  cl_int status;
  cl_uint mem_base_addr_align_;
  // this value is in terms of bits
  status = clGetDeviceInfo(dev_id,
                           CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                           sizeof(cl_uint),
                           &mem_base_addr_align_,
                           NULL);
  check(status);

  // so converting into bytes
  *mem_base_addr_align = mem_base_addr_align_ >> 3;
  return status;
}
