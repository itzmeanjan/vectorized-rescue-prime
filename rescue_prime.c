#include <rescue_prime.h>
#include <rescue_prime_constants.h>

cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl) {
  cl_int status;

  const size_t in_width = 8ul;
  const size_t out_width = 4ul;

  cl_ulong in_arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  cl_ulong out_arr[4] = {0, 0, 0, 0};

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 sizeof(cl_ulong) * in_width, NULL, &status);
  check(status);
  cl_mem in_size_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
  check(status);
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  check(status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  check(status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);
  check(status);
  cl_mem out_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  sizeof(cl_ulong) * out_width, NULL, &status);
  check(status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  check(status);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &in_size_buf);
  check(status);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(krnl, 3, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(krnl, 4, sizeof(cl_mem), &ark2_buf);
  check(status);
  status = clSetKernelArg(krnl, 5, sizeof(cl_mem), &out_buf);
  check(status);

  cl_event evt_0;
  status =
      clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, sizeof(cl_ulong) * in_width,
                           in_arr, 0, NULL, &evt_0);
  check(status);

  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, in_size_buf, CL_FALSE, 0, sizeof(size_t),
                                &in_width, 0, NULL, &evt_1);
  check(status);

  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_2);
  check(status);

  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_3);
  check(status);

  cl_event evt_4;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_4);
  check(status);

  size_t global_size[] = {1};
  size_t local_size[] = {1};
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3, evt_4};

  cl_event evt_5;
  status = clEnqueueNDRangeKernel(cq, krnl, 1, NULL, global_size, local_size, 5,
                                  evts, &evt_5);
  check(status);

  cl_event evt_6;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0,
                               sizeof(cl_ulong) * out_width, out_arr, 1, &evt_5,
                               &evt_6);
  check(status);

  status = clWaitForEvents(1, &evt_6);
  check(status);

  status = clReleaseEvent(evt_0);
  check(status);
  status = clReleaseEvent(evt_1);
  check(status);
  status = clReleaseEvent(evt_2);
  check(status);
  status = clReleaseEvent(evt_3);
  check(status);
  status = clReleaseEvent(evt_4);
  check(status);
  status = clReleaseEvent(evt_5);
  check(status);
  status = clReleaseEvent(evt_6);
  check(status);

  status = clReleaseMemObject(in_buf);
  check(status);
  status = clReleaseMemObject(in_size_buf);
  check(status);
  status = clReleaseMemObject(mds_buf);
  check(status);
  status = clReleaseMemObject(ark1_buf);
  check(status);
  status = clReleaseMemObject(ark2_buf);
  check(status);
  status = clReleaseMemObject(out_buf);
  check(status);

  return CL_SUCCESS;
}

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
  uint64_t exp_out_arr[16] = {10809974140050983728ull,
                              6938491977181280539ull,
                              8834525837561071698ull,
                              6854417192438540779ull,
                              4476630872663101667ull,
                              6292749486700362097ull,
                              18386622366690620454ull,
                              10614098972800193173ull,
                              7543273285584849722ull,
                              9490898458612615694ull,
                              9030271581669113292ull,
                              10101107035874348250ull,
                              0ull,
                              0ull,
                              0ull,
                              0ull};

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

  for (size_t i = 0; i < 16; i++) {
    assert(out_arr[i] % MOD == exp_out_arr[i] % MOD);
  }

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

  printf("passed apply_rescue_permutation tests !\n");

  return status;
}

cl_int test_apply_mds(cl_context ctx, cl_command_queue cq, cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {0ull, 1ull, 2ull,  3ull,  4ull, 5ull, 6ull, 7ull,
                         8ull, 9ull, 10ull, 11ull, 0ull, 0ull, 0ull, 0ull};
  uint64_t out_arr[16] = {0ull};
  uint64_t exp_out_arr[16] = {8268579649362235275ull,
                              2236502997719307940ull,
                              4445585223683938180ull,
                              8490351819144058838ull,
                              17912450758129541069ull,
                              12381447012212465193ull,
                              6444916863184583255ull,
                              5403602327365240081ull,
                              7656905977925454065ull,
                              12880871053868334997ull,
                              13669293285556299269ull,
                              2401034710645280649ull,
                              0ull,
                              0ull,
                              0ull,
                              0ull};

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

  for (size_t i = 0; i < 16; i++) {
    assert(out_arr[i] % MOD == exp_out_arr[i] % MOD);
  }

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(mds_buf);

  printf("passed apply_mds tests !\n");

  return status;
}

cl_int test_reduce_sum_vec2(cl_context ctx, cl_command_queue cq,
                            cl_kernel krnl) {
  cl_int status;

  uint64_t in_arr[16] = {1ull << 10, 1ull << 11, 1ull << 12, 1ull << 13,
                         1ull << 20, 1ull << 21, 1ull << 22, 1ull << 23,
                         1ull << 60, 1ull << 61, 1ull << 62, 1ull << 63,
                         MOD - 1ull, 1ull << 63, 0xffffffff, MOD - 1ull};
  uint64_t out_arr[8] = {0ull};
  uint64_t exp_out_arr[8] = {3072ull,
                             12288ull,
                             3145728ull,
                             12582912ull,
                             3458764513820540928ull,
                             13835058055282163712ull,
                             9223372036854775807ull,
                             18446744073709551615ull};

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_ulong) * 16,
                                 NULL, &status);
  cl_mem out_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * 8,
                                  NULL, &status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &out_buf);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, sizeof(in_arr), in_arr,
                                0, NULL, &evt_0);

  size_t global_size[] = {8};
  size_t local_size[] = {8};

  cl_event evt_1;
  status = clEnqueueNDRangeKernel(cq, krnl, 1, NULL, global_size, local_size, 1,
                                  &evt_0, &evt_1);

  cl_event evt_2;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, sizeof(out_arr),
                               out_arr, 1, &evt_1, &evt_2);

  status = clWaitForEvents(1, &evt_2);

  for (size_t i = 0; i < 8; i++) {
    assert(out_arr[i] % MOD == exp_out_arr[i] % MOD);
  }

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);

  printf("passed reduce_sum_vec2 tests !\n");

  return status;
}
