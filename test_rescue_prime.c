#include <stdio.h>
#include <test_rescue_prime.h>

cl_int test_apply_sbox(cl_context ctx, cl_command_queue cq, cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {1ull << 10, 1ull << 11, 1ull << 12, 1ull << 13,
                         1ull << 20, 1ull << 21, 1ull << 22, 1ull << 23,
                         1ull << 60, 1ull << 61, 1ull << 62, 1ull << 63,
                         0ull,       0ull,       0ull,       0ull};
  uint64_t out_arr[16] = {0ull};
  uint64_t exp_out_arr[16] = {274877906880ull,
                              35184372080640ull,
                              4503599626321920ull,
                              576460752169205760ull,
                              18446726477228539905ull,
                              18444492269600899073ull,
                              18158513693262872577ull,
                              18446744060824649731ull,
                              68719476736ull,
                              8796093022208ull,
                              1125899906842624ull,
                              144115188075855872ull,
                              0ull,
                              0ull,
                              0ull,
                              0ull};

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_ulong) * 16,
                                 NULL, &status);
  cl_mem out_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * 16,
                                  NULL, &status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &out_buf);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, sizeof(in_arr), in_arr,
                                0, NULL, &evt_0);

  size_t global_size[] = {1};
  size_t local_size[] = {1};

  cl_event evt_1;
  status = clEnqueueNDRangeKernel(cq, krnl, 1, NULL, global_size, local_size, 1,
                                  &evt_0, &evt_1);

  cl_event evt_2;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, sizeof(out_arr),
                               out_arr, 1, &evt_1, &evt_2);

  status = clWaitForEvents(1, &evt_2);

  for (size_t i = 0; i < 16; i++) {
    assert(out_arr[i] == exp_out_arr[i]);
  }

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);

  return status;
}
