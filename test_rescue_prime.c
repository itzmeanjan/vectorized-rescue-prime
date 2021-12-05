#include <rescue_prime_constants.h>
#include <stdio.h>
#include <test_rescue_prime.h>

const uint64_t MOD = 18446744069414584321ull;

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

  printf("passed apply_sbox tests !\n");

  return status;
}

cl_int test_apply_inv_sbox(cl_context ctx, cl_command_queue cq,
                           cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {1ull << 10, 1ull << 11, 1ull << 12, 1ull << 13,
                         1ull << 20, 1ull << 21, 1ull << 22, 1ull << 23,
                         1ull << 60, 1ull << 61, 1ull << 62, 1ull << 63,
                         0ull,       0ull,       0ull,       0ull};
  uint64_t out_arr[16] = {0ull};
  uint64_t exp_out_arr[16] = {18446743794536677441ull,
                              536870912ull,
                              4503599626321920ull,
                              18446735273321562113ull,
                              18446726477228539905ul,
                              8ull,
                              288230376151711744ull,
                              18446744069414453249ull,
                              68719476736ull,
                              576460752169205760ull,
                              18445618169507741697ull,
                              512ull,
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
    assert(out_arr[i] % MOD == exp_out_arr[i]);
  }

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);

  printf("passed apply_inv_sbox tests !\n");

  return status;
}

cl_int test_apply_rescue_permutation(cl_context ctx, cl_command_queue cq,
                                     cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {0ull, 1ull, 2ull,  3ull,  4ull, 5ull, 6ull, 7ull,
                         8ull, 9ull, 10ull, 11ull, 0ull, 0ull, 0ull, 0ull};
  uint64_t out_arr[16] = {0ull};
  // uint64_t exp_out_arr[16] = {18446743794536677441ull,
  //                             536870912ull,
  //                             4503599626321920ull,
  //                             18446735273321562113ull,
  //                             18446726477228539905ul,
  //                             8ull,
  //                             288230376151711744ull,
  //                             18446744069414453249ull,
  //                             68719476736ull,
  //                             576460752169205760ull,
  //                             18445618169507741697ull,
  //                             512ull,
  //                             0ull,
  //                             0ull,
  //                             0ull,
  //                             0ull};

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_ulong) * 16,
                                 NULL, &status);
  cl_mem out_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * 16,
                                  NULL, &status);
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &out_buf);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &mds_buf);
  status = clSetKernelArg(krnl, 3, sizeof(cl_mem), &ark1_buf);
  status = clSetKernelArg(krnl, 4, sizeof(cl_mem), &ark2_buf);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, sizeof(in_arr), in_arr,
                                0, NULL, &evt_0);

  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_1);

  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_2);

  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_3);

  // creating dependency in compute pipeline !
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3};

  size_t global_size[] = {1};
  size_t local_size[] = {1};

  cl_event evt_4;
  status = clEnqueueNDRangeKernel(cq, krnl, 1, NULL, global_size, local_size, 4,
                                  evts, &evt_4);

  cl_event evt_5;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, sizeof(out_arr),
                               out_arr, 1, &evt_4, &evt_5);

  status = clWaitForEvents(1, &evt_5);

  // Not yet in a state to be in form of test case
  // with assertions !
  //
  // WIP
  //
  printf("apply rescue permutation :\n");
  for (size_t i = 0; i < 16; i++) {
    // assert(out_arr[i] % MOD == exp_out_arr[i]);
    printf("%lu\t", out_arr[i]);
  }
  printf("\n\n");

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseEvent(evt_4);
  clReleaseEvent(evt_5);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(mds_buf);
  clReleaseMemObject(ark1_buf);
  clReleaseMemObject(ark2_buf);

  // printf("passed apply_rescue_permutation tests !\n");

  return status;
}

cl_int test_apply_mds(cl_context ctx, cl_command_queue cq, cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {1ull, 1ull, 2ull,  3ull,  4ull, 5ull, 6ull, 7ull,
                         8ull, 9ull, 10ull, 11ull, 0ull, 0ull, 0ull, 0ull};
  uint64_t out_arr[16] = {0ull};
  // uint64_t exp_out_arr[16] = {18446743794536677441ull,
  //                             536870912ull,
  //                             4503599626321920ull,
  //                             18446735273321562113ull,
  //                             18446726477228539905ul,
  //                             8ull,
  //                             288230376151711744ull,
  //                             18446744069414453249ull,
  //                             68719476736ull,
  //                             576460752169205760ull,
  //                             18445618169507741697ull,
  //                             512ull,
  //                             0ull,
  //                             0ull,
  //                             0ull,
  //                             0ull};

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_ulong) * 16,
                                 NULL, &status);
  cl_mem out_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * 16,
                                  NULL, &status);
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &out_buf);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &mds_buf);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, sizeof(in_arr), in_arr,
                                0, NULL, &evt_0);

  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_1);

  // creating dependency in compute pipeline !
  cl_event evts[] = {evt_0, evt_1};

  size_t global_size[] = {1};
  size_t local_size[] = {1};

  cl_event evt_2;
  status = clEnqueueNDRangeKernel(cq, krnl, 1, NULL, global_size, local_size, 2,
                                  evts, &evt_2);

  cl_event evt_3;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, sizeof(out_arr),
                               out_arr, 1, &evt_2, &evt_3);

  status = clWaitForEvents(1, &evt_3);

  // Not yet in a state to be in form of test case
  // with assertions !
  //
  // WIP
  //
  printf("apply mds :\n");
  for (size_t i = 0; i < 16; i++) {
    // assert(out_arr[i] % MOD == exp_out_arr[i]);
    printf("%lu\t", out_arr[i]);
  }
  printf("\n\n");

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(mds_buf);

  // printf("passed apply_mds tests !\n");

  return status;
}
