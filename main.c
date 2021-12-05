// #include <rescue_prime.h>
#include <stdio.h>
#include <test_rescue_prime.h>
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
  }

  status = show_build_log(dev_id, prgm);
  if (status != CL_SUCCESS) {
    printf("failed to obtain kernel build log !\n");
    return EXIT_FAILURE;
  }

  cl_kernel krnl_0 = clCreateKernel(prgm, "test_apply_sbox", &status);
  if (status != CL_SUCCESS) {
    printf("failed to create `test_apply_sbox` kernel !\n");
    return EXIT_FAILURE;
  }

  status = test_apply_sbox(ctx, c_queue, krnl_0);
  check(status);

  cl_kernel krnl_1 = clCreateKernel(prgm, "test_apply_inv_sbox", &status);
  if (status != CL_SUCCESS) {
    printf("failed to create `test_apply_inv_sbox` kernel !\n");
    return EXIT_FAILURE;
  }

  status = test_apply_inv_sbox(ctx, c_queue, krnl_1);
  check(status);

  cl_kernel krnl_2 =
      clCreateKernel(prgm, "test_apply_rescue_permutation", &status);
  if (status != CL_SUCCESS) {
    printf("failed to create `test_apply_rescue_permutation` kernel !\n");
    return EXIT_FAILURE;
  }

  status = test_apply_rescue_permutation(ctx, c_queue, krnl_2);
  check(status);

  cl_kernel krnl_3 = clCreateKernel(prgm, "test_apply_mds", &status);
  if (status != CL_SUCCESS) {
    printf("failed to create `test_apply_mds` kernel !\n");
    return EXIT_FAILURE;
  }

  status = test_apply_mds(ctx, c_queue, krnl_3);
  check(status);

  cl_kernel krnl_4 = clCreateKernel(prgm, "test_reduce_sum_vec2", &status);
  if (status != CL_SUCCESS) {
    printf("failed to create `test_reduce_sum_vec2` kernel !\n");
    return EXIT_FAILURE;
  }

  status = test_reduce_sum_vec2(ctx, c_queue, krnl_4);
  check(status);

  // cl_kernel krnl_3 = clCreateKernel(prgm, "hash_elements", &status);
  // if (status != CL_SUCCESS) {
  //   printf("failed to create rescue prime hash kernel !\n");
  //   return EXIT_FAILURE;
  // }

  // status = hash_elements(ctx, c_queue, krnl_3);
  // check(status);

  // releasing all OpenCL resources !
  clReleaseKernel(krnl_0);
  clReleaseKernel(krnl_1);
  clReleaseKernel(krnl_2);
  clReleaseKernel(krnl_3);
  clReleaseKernel(krnl_4);
  clReleaseProgram(prgm);
  clReleaseCommandQueue(c_queue);
  clReleaseContext(ctx);
  clReleaseDevice(dev_id);

  free(dev_name);

  return EXIT_SUCCESS;
}
