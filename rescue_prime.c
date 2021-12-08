#include <rescue_prime.h>
#include <rescue_prime_constants.h>

cl_int bench_hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                           size_t glb_sz_x, size_t glb_sz_y, size_t loc_sz_x,
                           size_t loc_sz_y) {
  cl_int status;

  const size_t in_width = 8ul;
  const size_t out_width = 4ul;
  const size_t in_size = glb_sz_x * glb_sz_y * in_width * sizeof(cl_ulong);
  const size_t out_size = glb_sz_x * glb_sz_y * out_width * sizeof(cl_ulong);

  cl_ulong *in_arr = malloc(in_size);
  cl_ulong *out_arr = malloc(out_size);

  random_field_elements(in_arr, in_size / sizeof(cl_ulong));

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &status);
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
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &status);
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
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, in_size, in_arr, 0,
                                NULL, &evt_0);
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

  size_t global_size[] = {glb_sz_x, glb_sz_y};
  size_t local_size[] = {loc_sz_x, loc_sz_y};
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3, evt_4};

  cl_event evt_5;
  status = clEnqueueNDRangeKernel(cq, krnl, 2, NULL, global_size, local_size, 5,
                                  evts, &evt_5);
  check(status);

  cl_event evt_6;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out_arr, 1,
                               &evt_5, &evt_6);
  check(status);

  status = clWaitForEvents(1, &evt_6);
  check(status);

  cl_ulong start, end;
  status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, NULL);
  check(status);
  status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &end, NULL);
  check(status);

  // kernel execution time in nanoseconds, obtained
  // by enabling profiling in command queue
  double ts = (double)(end - start);

  printf("%15s\t\t%5lu x %5lu\t\t%20.2f ms\t\t%15.2f hashes/ sec\n",
         "hash_elements", glb_sz_x, glb_sz_y, ts * 1e-6,
         ((double)(glb_sz_x * glb_sz_y) / (double)ts) * 1e9);

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

  free(in_arr);
  free(out_arr);

  return CL_SUCCESS;
}

cl_int bench_merge(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                   size_t global_size_x, size_t global_size_y,
                   size_t local_size_x, size_t local_size_y) {
  cl_int status;

  const size_t in_width = 8ul;
  const size_t out_width = 4ul;
  const size_t in_size =
      global_size_x * global_size_y * in_width * sizeof(cl_ulong);
  const size_t out_size =
      global_size_x * global_size_y * out_width * sizeof(cl_ulong);

  cl_ulong *in_arr = malloc(in_size);
  cl_ulong *out_arr = malloc(out_size);

  random_field_elements(in_arr, in_size / sizeof(cl_ulong));

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &status);
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
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &status);
  check(status);

  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  check(status);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(krnl, 3, sizeof(cl_mem), &ark2_buf);
  check(status);
  status = clSetKernelArg(krnl, 4, sizeof(cl_mem), &out_buf);
  check(status);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, in_size, in_arr, 0,
                                NULL, &evt_0);
  check(status);

  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_1);
  check(status);

  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_2);
  check(status);

  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_3);
  check(status);

  size_t global_size[] = {global_size_x, global_size_y};
  size_t local_size[] = {local_size_x, local_size_y};
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3};

  cl_event evt_4;
  status = clEnqueueNDRangeKernel(cq, krnl, 2, NULL, global_size, local_size, 4,
                                  evts, &evt_4);
  check(status);

  cl_event evt_5;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out_arr, 1,
                               &evt_4, &evt_5);
  check(status);

  status = clWaitForEvents(1, &evt_5);
  check(status);

  cl_ulong start, end;
  status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &start, NULL);
  check(status);
  status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &end, NULL);
  check(status);

  // kernel execution time in nanoseconds, obtained
  // by enabling profiling in command queue
  //
  // make sure
  // https://github.com/itzmeanjan/vectorized-rescue-prime/blob/54df2cd08de2e3d56c7a6e0202981c489ff0ee63/main.c#L35-L44
  // stays as it's
  double ts = (double)(end - start);

  printf("%15s\t\t%5lu x %5lu\t\t%20.2f ms\t\t%15.2f merges/ sec\n", "merge",
         global_size_x, global_size_y, ts * 1e-6,
         ((double)(global_size_x * global_size_y) / (double)ts) * 1e9);

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

  status = clReleaseMemObject(in_buf);
  check(status);
  status = clReleaseMemObject(mds_buf);
  check(status);
  status = clReleaseMemObject(ark1_buf);
  check(status);
  status = clReleaseMemObject(ark2_buf);
  check(status);
  status = clReleaseMemObject(out_buf);
  check(status);

  free(in_arr);
  free(out_arr);

  return CL_SUCCESS;
}

cl_int build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                          cl_kernel merge_krnl_0, cl_kernel merge_krnl_1,
                          cl_kernel tip_krnl, cl_ulong *in, cl_ulong *out,
                          const size_t leave_count, const size_t wg_size) {
  // leave count of merkle tree should be power of 2
  assert((leave_count) & (leave_count - 1ul) == 0);
  // intermediate nodes of tree those can be computed in parallel
  //
  // to be specific
  // https://github.com/novifinancial/winterfell/blob/377e916c47fab3d9fa173b2f6123c7b713ffce03/crypto/src/merkle/mod.rs#L326-L329
  // section
  const size_t itmd_par_node_cnt = leave_count >> 1;
  const size_t subtree_cnt = itmd_par_node_cnt >> 1;
  assert(itmd_par_node_cnt >= wg_size);
  // input/ output both are 4-field element wide
  // rescue prime hash digests, stored in consequtive memory locations
  const size_t io_width = 4ul;
  const size_t in_size = io_width * leave_count * sizeof(cl_ulong);
  const size_t out_size = io_width * leave_count * sizeof(cl_ulong);
  const size_t itmd_size_0 = io_width * itmd_par_node_cnt * sizeof(cl_ulong);
  const size_t itmd_size_1 = io_width * subtree_cnt * sizeof(cl_ulong);

  cl_int status;

  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  check(status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  check(status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);
  check(status);

  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &status);
  check(status);
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &status);
  check(status);

  cl_buffer_region out_buf_reg_0;
  out_buf_reg_0.origin = itmd_size_0;
  out_buf_reg_0.size = itmd_size_0;

  cl_mem out_buf_0 =
      clCreateSubBuffer(out_buf, CL_MEM_WRITE_ONLY,
                        CL_BUFFER_CREATE_TYPE_REGION, &out_buf_reg_0, &status);
  check(status);

  cl_mem in_buf_0 =
      clCreateSubBuffer(out_buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                        &out_buf_reg_0, &status);
  check(status);

  cl_buffer_region out_buf_reg_1;
  out_buf_reg_1.origin = itmd_size_1;
  out_buf_reg_1.size = itmd_size_1;

  cl_mem out_buf_1 =
      clCreateSubBuffer(out_buf, CL_MEM_WRITE_ONLY,
                        CL_BUFFER_CREATE_TYPE_REGION, &out_buf_reg_1, &status);
  check(status);

  cl_buffer_region in_out_buf_reg;
  in_out_buf_reg.origin = 0;
  in_out_buf_reg.size = itmd_size_1;

  cl_mem in_out_buf =
      clCreateSubBuffer(out_buf, CL_MEM_READ_WRITE,
                        CL_BUFFER_CREATE_TYPE_REGION, &in_out_buf_reg, &status);
  check(status);

  status = clSetKernelArg(merge_krnl_0, 0, sizeof(cl_mem), &in_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_0, 1, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_0, 2, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_0, 3, sizeof(cl_mem), &ark2_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_0, 4, sizeof(cl_mem), &out_buf_0);
  check(status);

  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, in_size, in, 0, NULL,
                                &evt_0);
  check(status);

  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_1);
  check(status);

  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_2);
  check(status);

  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_3);
  check(status);

  size_t global_size_0[] = {1, itmd_par_node_cnt};
  size_t local_size_0[] = {1, wg_size};
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3};

  cl_event evt_4;
  status = clEnqueueNDRangeKernel(cq, merge_krnl_0, 2, NULL, global_size_0,
                                  local_size_0, 4, evts, &evt_4);
  check(status);

  status = clSetKernelArg(merge_krnl_1, 0, sizeof(cl_mem), &in_buf_0);
  check(status);
  status = clSetKernelArg(merge_krnl_1, 1, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_1, 2, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_1, 3, sizeof(cl_mem), &ark2_buf);
  check(status);
  status = clSetKernelArg(merge_krnl_1, 4, sizeof(cl_mem), &out_buf_1);
  check(status);

  size_t global_size_1[] = {1, subtree_cnt};
  size_t local_size_1[] = {1, wg_size};

  cl_event evt_5;
  status = clEnqueueNDRangeKernel(cq, merge_krnl_1, 2, NULL, global_size_1,
                                  local_size_1, 1, &evt_4, &evt_5);
  check(status);

  cl_mem num_subtrees_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
  check(status);

  status = clSetKernelArg(tip_krnl, 0, sizeof(cl_mem), &in_buf_0);
  check(status);
  status = clSetKernelArg(tip_krnl, 1, sizeof(cl_mem), &num_subtrees_buf);
  check(status);
  status = clSetKernelArg(tip_krnl, 2, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(tip_krnl, 3, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(tip_krnl, 4, sizeof(cl_mem), &ark2_buf);
  check(status);

  cl_event evt_6;
  status = clEnqueueWriteBuffer(cq, num_subtrees_buf, CL_FALSE, 0,
                                sizeof(size_t), &subtree_cnt, 0, NULL, &evt_6);
  check(status);

  size_t global_size_2[] = {1};
  size_t local_size_2[] = {1};
  cl_event evts_[] = {evt_5, evt_6};

  cl_event evt_7;
  status = clEnqueueNDRangeKernel(cq, tip_krnl, 1, NULL, global_size_2,
                                  local_size_2, 2, evts_, &evt_7);
  check(status);

  cl_event evt_8;
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out, 1,
                               &evt_7, &evt_8);
  check(status);

  status = clWaitForEvents(1, &evt_8);
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

// This function is used for computing (global_size_x x global_size_y) rescue
// prime hashes, each of input of width N, and produces same number of outputs,
// each of width 4
//
// Note when I mentioned about width of N or 4, I mean input/ output
// will have those many 64-bit prime field elements
//
// local_size_{x, y} denotes work-group size vertically/ horizontally
//
// I'm going to use this function for testing `merge` kernel
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L378
cl_int calculate_hash(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                      cl_ulong *input, size_t input_width, cl_ulong *output,
                      size_t global_size_x, size_t global_size_y,
                      size_t local_size_x, size_t local_size_y) {
  cl_int status;

  // input is supplied to kernel by this buffer
  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 sizeof(cl_ulong) * input_width *
                                     global_size_x * global_size_y,
                                 NULL, &status);

  // buffer for keeping width of input to hash_elements kernel, stored in
  // constant memory
  cl_mem in_width_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);

  // Following three buffers will be storing rescue prime hash constants
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);

  // output to be placed here, after kernel completes hash computation
  cl_mem out_buf = clCreateBuffer(
      ctx, CL_MEM_WRITE_ONLY,
      sizeof(cl_ulong) * 4 * global_size_x * global_size_y, NULL, &status);

  // input being copied to device memory
  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0,
                                sizeof(cl_ulong) * input_width * global_size_x *
                                    global_size_y,
                                input, 0, NULL, &evt_0);
  // input width being copied to device memory
  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, in_width_buf, CL_FALSE, 0, sizeof(size_t),
                                &input_width, 0, NULL, &evt_1);

  // scheduling rescue prime constant copying
  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_2);
  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_3);
  cl_event evt_4;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_4);

  // setting kernel arguments for
  // https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L320
  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &in_width_buf);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &mds_buf);
  status = clSetKernelArg(krnl, 3, sizeof(cl_mem), &ark1_buf);
  status = clSetKernelArg(krnl, 4, sizeof(cl_mem), &ark2_buf);
  status = clSetKernelArg(krnl, 5, sizeof(cl_mem), &out_buf);

  // preparing for creating dependency in compute execution graph
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3, evt_4};
  size_t global_size[] = {global_size_x, global_size_y};
  size_t local_size[] = {local_size_x, local_size_y};

  // kernel being dispatched for execution on device
  cl_event evt_5;
  status = clEnqueueNDRangeKernel(cq, krnl, 2, NULL, global_size, local_size, 5,
                                  evts, &evt_5);

  // hash output being copied back to host
  cl_event evt_6;
  status =
      clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0,
                          sizeof(cl_ulong) * 4 * global_size_x * global_size_y,
                          output, 1, &evt_5, &evt_6);

  status = clWaitForEvents(1, &evt_6);

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseEvent(evt_4);
  clReleaseEvent(evt_5);
  clReleaseEvent(evt_6);

  clReleaseMemObject(in_buf);
  clReleaseMemObject(in_width_buf);
  clReleaseMemObject(mds_buf);
  clReleaseMemObject(ark1_buf);
  clReleaseMemObject(ark2_buf);
  clReleaseMemObject(out_buf);

  return status;
}

// Merges two rescue prime hashes into single one by dispatching
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L378
// kernel (global_size_x x global_size_y)-many times
//
// It should produce same number of rescue prime digests, each of width 4,
// while in input each was of width 8, because two input rescue prime digests to
// be merged
cl_int merge(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
             cl_ulong *input, cl_ulong *output, size_t global_size_x,
             size_t global_size_y, size_t local_size_x, size_t local_size_y) {
  cl_int status;

  // input is supplied to kernel by this buffer
  cl_mem in_buf = clCreateBuffer(
      ctx, CL_MEM_READ_ONLY,
      sizeof(cl_ulong) * 8 * global_size_x * global_size_y, NULL, &status);

  // Following three buffers will be storing rescue prime hash constants
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);

  // output to be placed here, after kernel completes hash computation
  cl_mem out_buf = clCreateBuffer(
      ctx, CL_MEM_WRITE_ONLY,
      sizeof(cl_ulong) * 4 * global_size_x * global_size_y, NULL, &status);

  // input being copied to device memory
  cl_event evt_0;
  status =
      clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0,
                           sizeof(cl_ulong) * 8 * global_size_x * global_size_y,
                           input, 0, NULL, &evt_0);

  // scheduling rescue prime constant copying
  cl_event evt_1;
  status = clEnqueueWriteBuffer(cq, mds_buf, CL_FALSE, 0, sizeof(MDS), MDS, 0,
                                NULL, &evt_1);
  cl_event evt_2;
  status = clEnqueueWriteBuffer(cq, ark1_buf, CL_FALSE, 0, sizeof(ARK1), ARK1,
                                0, NULL, &evt_2);
  cl_event evt_3;
  status = clEnqueueWriteBuffer(cq, ark2_buf, CL_FALSE, 0, sizeof(ARK2), ARK2,
                                0, NULL, &evt_3);

  // setting kernel arguments for
  // https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L320
  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &mds_buf);
  status = clSetKernelArg(krnl, 2, sizeof(cl_mem), &ark1_buf);
  status = clSetKernelArg(krnl, 3, sizeof(cl_mem), &ark2_buf);
  status = clSetKernelArg(krnl, 4, sizeof(cl_mem), &out_buf);

  // preparing for creating dependency in compute execution graph
  cl_event evts[] = {evt_0, evt_1, evt_2, evt_3};
  size_t global_size[] = {global_size_x, global_size_y};
  size_t local_size[] = {local_size_x, local_size_y};

  // kernel being dispatched for execution on device
  cl_event evt_4;
  status = clEnqueueNDRangeKernel(cq, krnl, 2, NULL, global_size, local_size, 4,
                                  evts, &evt_4);

  // hash output being copied back to host
  cl_event evt_5;
  status =
      clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0,
                          sizeof(cl_ulong) * 4 * global_size_x * global_size_y,
                          output, 1, &evt_4, &evt_5);

  status = clWaitForEvents(1, &evt_5);

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseEvent(evt_4);
  clReleaseEvent(evt_5);

  clReleaseMemObject(in_buf);
  clReleaseMemObject(mds_buf);
  clReleaseMemObject(ark1_buf);
  clReleaseMemObject(ark2_buf);
  clReleaseMemObject(out_buf);

  return status;
}

cl_int test_merge(cl_context ctx, cl_command_queue cq, cl_kernel hash_krnl,
                  cl_kernel merge_krnl) {
  cl_int status;

  cl_ulong *in = malloc(sizeof(cl_ulong) * 16);
  cl_ulong *out = malloc(sizeof(cl_ulong) * 16);

  // prepare random 16 field elements
  random_field_elements(in, 16);

  // Hash 8 consequtive elements together, twice i.e. using two work-items
  status = calculate_hash(ctx, cq, hash_krnl, in, 8, out, 1, 2, 1, 2);
  // Then merge 8 consequtive elements together, such that they are
  // interpreted to be two rescue prime hash digests concatenated
  // one after another
  //
  // Do this thing twice i.e. using two work-items
  status = merge(ctx, cq, merge_krnl, in, out + 8, 1, 2, 1, 2);

  // it should produce 8 hash digests, each of width 4 field elements 👇
  //
  // out[0:4] produced from first work-item of `calculate_hash` function above
  // out[4:8] produced from second work-item of `calculate_hash` function above
  //
  // out[8:12] produced from first work-item of `merge` function above
  // out[12:16] produced from second work-item of `merge` function above
  //
  // This is how input and output are associated
  //
  // out[0:4] = hash_elements(in[0:8])
  // out[4:8] = hash_elements(in[8:16])
  //
  // out[8:12] = merge(in[0:8])
  // out[12:16] = merge(in[8:16])
  //
  // so followings should be passing !
  assert(out[0] == out[8]);
  assert(out[1] == out[9]);
  assert(out[2] == out[10]);
  assert(out[3] == out[11]);

  assert(out[4] == out[12]);
  assert(out[5] == out[13]);
  assert(out[6] == out[14]);
  assert(out[7] == out[15]);

  printf("passed merge tests !\n");

  // deallocate memory
  free(in);
  free(out);

  return status;
}
