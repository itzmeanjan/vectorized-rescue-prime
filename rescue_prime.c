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

  // kernel execution time in nanoseconds, obtained
  // by enabling profiling in command queue
  cl_ulong ts;
  status = hash_elements(ctx, cq, krnl, in_arr, in_width, out_arr, glb_sz_x,
                         glb_sz_y, loc_sz_x, loc_sz_y, &ts);
  check(status);

  printf("%15s\t\t%5lu x %5lu\t\t%20.2f ms\t\t%15.2f hashes/ sec\n",
         "hash_elements", glb_sz_x, glb_sz_y, (double)ts * 1e-6,
         ((double)(glb_sz_x * glb_sz_y) / (double)ts) * 1e9);

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

  // kernel execution time in nanoseconds, obtained
  // by enabling profiling in command queue
  //
  // make sure
  // https://github.com/itzmeanjan/vectorized-rescue-prime/blob/54df2cd08de2e3d56c7a6e0202981c489ff0ee63/main.c#L35-L44
  // stays as it's
  cl_ulong ts;
  status = merge(ctx, cq, krnl, in_arr, out_arr, global_size_x, global_size_y,
                 local_size_x, local_size_y, &ts);
  check(status);

  printf("%15s\t\t%5lu x %5lu\t\t%20.2f ms\t\t%15.2f merges/ sec\n", "merge",
         global_size_x, global_size_y, (double)ts * 1e-6,
         ((double)(global_size_x * global_size_y) / (double)ts) * 1e9);

  free(in_arr);
  free(out_arr);

  return CL_SUCCESS;
}

cl_int bench_build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                                cl_kernel merge_krnl, cl_kernel tip_krnl,
                                size_t global_size, size_t local_size,
                                const size_t dev_mem_base_addr_align) {
  cl_int status;

  const size_t io_width = 4;
  const size_t in_size = global_size * io_width * sizeof(cl_ulong);
  const size_t out_size = global_size * io_width * sizeof(cl_ulong);

  cl_ulong *in_arr = malloc(in_size);
  cl_ulong *out_arr = malloc(out_size);

  random_field_elements(in_arr, in_size / sizeof(cl_ulong));

  cl_ulong ts;
  status =
      build_merkle_nodes(ctx, cq, merge_krnl, tip_krnl, in_arr, out_arr,
                         global_size, local_size, &ts, dev_mem_base_addr_align);
  check(status);

  printf("%15s\t\t%10lu leaves\t\t%20.2f ms\n", "merklize", global_size,
         (double)ts * 1e-6);

  free(in_arr);
  free(out_arr);

  return CL_SUCCESS;
}

cl_int build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                          cl_kernel merge_krnl, cl_kernel tip_krnl,
                          cl_ulong *in, cl_ulong *out, const size_t leave_count,
                          const size_t wg_size, cl_ulong *ts,
                          const size_t dev_mem_base_addr_align) {
  // leave count of merkle tree should be power of 2
  assert((leave_count & (leave_count - 1ul)) == 0);
  // intermediate nodes of tree, living just above leaves,
  // those can be computed in parallel
  //
  // to be specific
  // https://github.com/novifinancial/winterfell/blob/377e916c47fab3d9fa173b2f6123c7b713ffce03/crypto/src/merkle/mod.rs#L326-L329
  // section
  assert((leave_count >> 1) >= wg_size);
  // input/ output both are 4-field element wide
  // rescue prime hash digests, stored in consequtive
  // memory locations
  const size_t io_width = 4ul;
  const size_t in_size = io_width * leave_count * sizeof(cl_ulong);
  const size_t out_size = io_width * leave_count * sizeof(cl_ulong);

  cl_int status;

  // following three buffers are unchanged during execution of this function
  // these're required to be copied one time
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
  // whole output buffer, allocated on device memory, this buffer will be
  // sub-divided multiple times, in following section
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_size, NULL, &status);
  check(status);

  cl_buffer_region sub_buf_reg;
  sub_buf_reg.origin = (leave_count >> 1) * io_width * sizeof(cl_ulong);
  sub_buf_reg.size = (leave_count >> 1) * io_width * sizeof(cl_ulong);

  cl_mem out_sub_buf_ =
      clCreateSubBuffer(out_buf, CL_MEM_WRITE_ONLY,
                        CL_BUFFER_CREATE_TYPE_REGION, &sub_buf_reg, &status);
  check(status);

  status = clSetKernelArg(merge_krnl, 0, sizeof(cl_mem), &in_buf);
  check(status);
  status = clSetKernelArg(merge_krnl, 1, sizeof(cl_mem), &mds_buf);
  check(status);
  status = clSetKernelArg(merge_krnl, 2, sizeof(cl_mem), &ark1_buf);
  check(status);
  status = clSetKernelArg(merge_krnl, 3, sizeof(cl_mem), &ark2_buf);
  check(status);
  status = clSetKernelArg(merge_krnl, 4, sizeof(cl_mem), &out_sub_buf_);
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

  size_t global_size_0[] = {1, leave_count >> 1};
  size_t local_size_0[] = {1, wg_size};
  cl_event evts_0[] = {evt_0, evt_1, evt_2, evt_3};

  cl_event evt_4;
  status = clEnqueueNDRangeKernel(cq, merge_krnl, 2, NULL, global_size_0,
                                  local_size_0, 4, evts_0, &evt_4);
  check(status);

  // intermediate nodes in tip of tree, to be computed sequentially
  const size_t subtree_count = (dev_mem_base_addr_align >> 5);

  if (!((leave_count >> 1) >= (subtree_count << 1))) {
    cl_buffer_region sub_buf_reg_0;
    sub_buf_reg_0.origin = 0;
    sub_buf_reg_0.size = leave_count * io_width * sizeof(cl_ulong);

    cl_mem in_out_sub_buf = clCreateSubBuffer(out_buf, CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION,
                                              &sub_buf_reg_0, &status);
    check(status);

    cl_mem subtree_count_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
    check(status);

    const size_t subtree_count_ = leave_count >> 1;
    cl_event evt_5;
    status =
        clEnqueueWriteBuffer(cq, subtree_count_buf, CL_FALSE, 0, sizeof(size_t),
                             &subtree_count_, 0, NULL, &evt_5);
    check(status);

    status = clSetKernelArg(tip_krnl, 0, sizeof(cl_mem), &in_out_sub_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 1, sizeof(cl_mem), &subtree_count_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 2, sizeof(cl_mem), &mds_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 3, sizeof(cl_mem), &ark1_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 4, sizeof(cl_mem), &ark2_buf);
    check(status);

    size_t global_size_2[] = {1};
    size_t local_size_2[] = {1};
    cl_event evts_1[] = {evt_4, evt_5};
    cl_event evt_6;
    status = clEnqueueNDRangeKernel(cq, tip_krnl, 1, NULL, global_size_2,
                                    local_size_2, 2, evts_1, &evt_6);
    check(status);

    cl_event evt_7;
    status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out, 1,
                                 &evt_6, &evt_7);
    check(status);

    status = clWaitForEvents(1, &evt_7);
    check(status);

    if (ts != NULL) {
      cl_ulong start, end;
      *ts = 0; // zerod before accumulation, just to be safe

      status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_START,
                                       sizeof(cl_ulong), &start, NULL);
      status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &end, NULL);
      *ts += (end - start);

      status = clGetEventProfilingInfo(evt_6, CL_PROFILING_COMMAND_START,
                                       sizeof(cl_ulong), &start, NULL);
      status = clGetEventProfilingInfo(evt_6, CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &end, NULL);
      *ts += (end - start);
    }

    // deallocate all opencl related resources which were acquired
    // during execution of this function
    clReleaseEvent(evt_0);
    clReleaseEvent(evt_1);
    clReleaseEvent(evt_2);
    clReleaseEvent(evt_3);
    clReleaseEvent(evt_4);
    clReleaseEvent(evt_5);
    clReleaseEvent(evt_6);
    clReleaseEvent(evt_7);

    clReleaseMemObject(subtree_count_buf);
    clReleaseMemObject(in_out_sub_buf);
    clReleaseMemObject(out_sub_buf_);
    clReleaseMemObject(in_buf);
    clReleaseMemObject(out_buf);
    clReleaseMemObject(mds_buf);
    clReleaseMemObject(ark1_buf);
    clReleaseMemObject(ark2_buf);

    return status;
  }

  // data parallel intermediate node compute stages, where n-th one
  // depends on completion of (n-1)-th
  const size_t rounds =
      (size_t)log2((double)((leave_count >> 1) / subtree_count));

  cl_event *evts_1 = malloc(sizeof(cl_event) * rounds);
  cl_mem *rd_sub_bufs = malloc(sizeof(cl_mem) * rounds);
  cl_mem *wr_sub_bufs = malloc(sizeof(cl_mem) * rounds);

  for (size_t i = leave_count >> 1, idx = 0;
       (i >> 1) * io_width * sizeof(cl_ulong) >= dev_mem_base_addr_align;
       i >>= 1, idx++) {
    cl_buffer_region sub_buf_reg_0;
    sub_buf_reg_0.origin = i * io_width * sizeof(cl_ulong);
    sub_buf_reg_0.size = i * io_width * sizeof(cl_ulong);

    cl_mem in_sub_buf = clCreateSubBuffer(out_buf, CL_MEM_READ_ONLY,
                                          CL_BUFFER_CREATE_TYPE_REGION,
                                          &sub_buf_reg_0, &status);
    check(status);
    rd_sub_bufs[idx] = in_sub_buf;

    cl_buffer_region sub_buf_reg_1;
    sub_buf_reg_1.origin = (i >> 1) * io_width * sizeof(cl_ulong);
    sub_buf_reg_1.size = (i >> 1) * io_width * sizeof(cl_ulong);

    cl_mem out_sub_buf = clCreateSubBuffer(out_buf, CL_MEM_WRITE_ONLY,
                                           CL_BUFFER_CREATE_TYPE_REGION,
                                           &sub_buf_reg_1, &status);
    check(status);
    wr_sub_bufs[idx] = out_sub_buf;

    status = clSetKernelArg(merge_krnl, 0, sizeof(cl_mem), &in_sub_buf);
    check(status);
    status = clSetKernelArg(merge_krnl, 1, sizeof(cl_mem), &mds_buf);
    check(status);
    status = clSetKernelArg(merge_krnl, 2, sizeof(cl_mem), &ark1_buf);
    check(status);
    status = clSetKernelArg(merge_krnl, 3, sizeof(cl_mem), &ark2_buf);
    check(status);
    status = clSetKernelArg(merge_krnl, 4, sizeof(cl_mem), &out_sub_buf);
    check(status);

    size_t global_size_1[] = {1, i >> 1};
    // make sure work-group size is compatible with work-size in this iteration
    size_t local_size_1[] = {1, (i >> 1) >= wg_size ? wg_size : (i >> 1)};

    cl_event evt_;
    status = clEnqueueNDRangeKernel(
        cq, merge_krnl, 2, NULL, global_size_1, local_size_1, 1,
        idx == 0 ? &evt_4 : evts_1 + (idx - 1), &evt_);
    check(status);

    evts_1[idx] = evt_;
  }

  cl_ulong ts_ = 0;
  if (subtree_count > 1) {
    cl_buffer_region sub_buf_reg_0;
    sub_buf_reg_0.origin = 0;
    sub_buf_reg_0.size = (subtree_count << 1) * io_width * sizeof(cl_ulong);

    cl_mem in_out_sub_buf = clCreateSubBuffer(out_buf, CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION,
                                              &sub_buf_reg_0, &status);
    check(status);

    cl_mem subtree_count_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
    check(status);

    cl_event evt_5;
    status =
        clEnqueueWriteBuffer(cq, subtree_count_buf, CL_FALSE, 0, sizeof(size_t),
                             &subtree_count, 0, NULL, &evt_5);
    check(status);

    status = clSetKernelArg(tip_krnl, 0, sizeof(cl_mem), &in_out_sub_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 1, sizeof(cl_mem), &subtree_count_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 2, sizeof(cl_mem), &mds_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 3, sizeof(cl_mem), &ark1_buf);
    check(status);
    status = clSetKernelArg(tip_krnl, 4, sizeof(cl_mem), &ark2_buf);
    check(status);

    size_t global_size_2[] = {1};
    size_t local_size_2[] = {1};
    cl_event evts_2[] = {*(evts_1 + (rounds - 1)), evt_5};
    cl_event evt_6;
    status = clEnqueueNDRangeKernel(cq, tip_krnl, 1, NULL, global_size_2,
                                    local_size_2, 2, evts_2, &evt_6);
    check(status);

    cl_event evt_7;
    status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out, 1,
                                 &evt_6, &evt_7);
    check(status);

    status = clWaitForEvents(1, &evt_7);
    check(status);

    cl_ulong start, end;
    status = clGetEventProfilingInfo(evt_6, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start, NULL);
    status = clGetEventProfilingInfo(evt_6, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &end, NULL);
    ts_ += (end - start);

    status = clReleaseEvent(evt_5);
    check(status);
    status = clReleaseEvent(evt_6);
    check(status);
    status = clReleaseEvent(evt_7);
    check(status);

    status = clReleaseMemObject(subtree_count_buf);
    status = clReleaseMemObject(in_out_sub_buf);
  } else {
    cl_event evt_5;
    status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, out, 1,
                                 evts_1 + (rounds - 1), &evt_5);
    check(status);

    status = clWaitForEvents(1, &evt_5);
    check(status);

    status = clReleaseEvent(evt_5);
    check(status);
  }

  // compute total execution time of all three kernels
  // which are dispatched for computing intermediate nodes of merkle tree
  if (ts != NULL) {
    cl_ulong start, end;
    *ts = 0; // zerod before accumulation, just to be safe

    status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start, NULL);
    status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &end, NULL);
    *ts += (end - start);

    for (size_t i = 0; i < rounds; i++) {
      status =
          clGetEventProfilingInfo(*(evts_1 + i), CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &start, NULL);
      status = clGetEventProfilingInfo(*(evts_1 + i), CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &end, NULL);
      *ts += (end - start);
    }

    *ts += ts_;
  }

  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_3);
  clReleaseEvent(evt_4);

  for (size_t i = 0; i < rounds; i++) {
    clReleaseEvent(*(evts_1 + i));
    clReleaseMemObject(*(rd_sub_bufs + i));
    clReleaseMemObject(*(wr_sub_bufs + i));
  }

  clReleaseMemObject(out_sub_buf_);
  clReleaseMemObject(in_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(mds_buf);
  clReleaseMemObject(ark1_buf);
  clReleaseMemObject(ark2_buf);

  free(evts_1);
  free(rd_sub_bufs);
  free(wr_sub_bufs);

  return CL_SUCCESS;
}

cl_int test_build_merkle_nodes(cl_context ctx, cl_command_queue cq,
                               cl_kernel merge_krnl, cl_kernel tip_kernel,
                               const size_t dev_mem_base_addr_align) {
  cl_int status;

  // leave count of merkle tree
  const size_t N = 16;
  // because each rescue prime digest consists of 4 field elements
  const size_t io_width = 4;
  // in terms of bytes
  const size_t io_size = N * io_width * sizeof(cl_long);

  // to be randomly generated and interpreted such that N-many rescue prime
  // hash digests are concatenated one after another
  cl_ulong *in = malloc(io_size);
  // output to be computed by function `build_merkle_tree`, this is what I'm
  // testing
  cl_ulong *out_0 = malloc(io_size);
  // this output it going to be computed manually by invoking `merge` function
  // step by step on a pair of merkle tree nodes
  cl_ulong *out_1 = malloc(io_size);

  // randomly generated N * 4-many prime field elements
  // to be interpreted as N-many rescue prime hash digests
  random_field_elements(in, io_size / sizeof(cl_ulong));

  // compute merkle tree intermediate nodes, to be asserted in next few steps
  status = build_merkle_nodes(ctx, cq, merge_krnl, tip_kernel, in, out_0, N, 1,
                              NULL, dev_mem_base_addr_align);
  check(status);

  // Assume A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  // to be a set of merkle tree leaves, each of width 4-prime field elements
  //
  // so A must have 16 * 4 = 64 prime field elements
  //
  // B = [0; 16], result array for storing all intermediate nodes of tree
  // such that
  // - B[0] == may be some random value, but of no interest
  // - B[1] == root
  // - B[2], B[3] == children of B[1]
  // - B[4], B[5] == children of B[2]
  // - B[6], B[7] == children of B[3]
  // ...
  //
  // B[15] = merge(A[14], A[15])
  // B[14] = merge(A[12], A[13])
  // ...
  // B[8] = merge(A[0], A[1])
  //
  // For N-many leaves N/2 -many intermediate nodes are computed in this step
  for (size_t j = 1; j <= (N >> 1); j++) {
    status = merge(ctx, cq, merge_krnl, in + (N - 2 * j) * io_width,
                   out_1 + (N - j) * io_width, 1, 1, 1, 1, NULL);

    for (size_t i = 0; i < io_width; i++) {
      assert(*(out_1 + (N - j) * io_width + i) ==
             *(out_0 + (N - j) * io_width + i));
    }
  }

  // As soon as level above leaves are computed, I can move to next level of
  // tree
  //
  // In this step I'll compute N/4 -many intermediate nodes, living just above
  // previous level of nodes
  //
  // B[7] = merge(B[14], B[15])
  // B[6] = merge(B[12], B[13])
  // B[5] = merge(B[10], B[11])
  // B[4] = merge(B[8], B[9])
  for (size_t j = 1; j <= (N >> 2); j++) {
    status = merge(ctx, cq, merge_krnl, out_1 + (N - 2 * j) * io_width,
                   out_1 + ((N >> 1) - j) * io_width, 1, 1, 1, 1, NULL);

    for (size_t i = 0; i < io_width; i++) {
      assert(*(out_1 + ((N >> 1) - j) * io_width + i) ==
             *(out_0 + ((N >> 1) - j) * io_width + i));
    }
  }

  // In next level I'll compute N/8 -many intermediates
  //
  // B[3] = merge(B[6], B[7])
  // B[2] = merge(B[4], B[5])
  for (size_t j = 1; j <= (N >> 3); j++) {
    status = merge(ctx, cq, merge_krnl, out_1 + ((N >> 1) - 2 * j) * io_width,
                   out_1 + ((N >> 2) - j) * io_width, 1, 1, 1, 1, NULL);

    for (size_t i = 0; i < io_width; i++) {
      assert(*(out_1 + ((N >> 2) - j) * io_width + i) ==
             *(out_0 + ((N >> 2) - j) * io_width + i));
    }
  }

  // And this is the root of merkle tree !
  //
  // Only one node to be computed
  //
  // B[1] = merge(B[2], B[3])
  for (size_t j = 1; j <= (N >> 4); j++) {
    status = merge(ctx, cq, merge_krnl, out_1 + ((N >> 2) - 2 * j) * io_width,
                   out_1 + ((N >> 3) - j) * io_width, 1, 1, 1, 1, NULL);

    for (size_t i = 0; i < io_width; i++) {
      assert(*(out_1 + ((N >> 3) - j) * io_width + i) ==
             *(out_0 + ((N >> 3) - j) * io_width + i));
    }
  }

  // deallocate resources
  free(in);
  free(out_0);
  free(out_1);

  printf("passed build_merkle_nodes tests !\n");

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

// This function is used for computing (global_size_x x global_size_y)-many
// rescue prime hashes, each on equal sized input of width N, and produces same
// number of outputs, each of width 4
//
// Note when I mentioned about width of N or 4, I mean input/ output
// will have those many 64-bit prime field elements
//
// local_size_{x, y} denotes work-group size vertically/ horizontally
//
// I'm going to use this function for testing `merge` kernel
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/bf40c7e41431487633288b7f64ebd804245fd8eb/kernel.cl#L378
//
// Now I've also started using this generic function for benchmarking
// `hash_elements` kernel
cl_int hash_elements(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
                     cl_ulong *input, const size_t input_width,
                     cl_ulong *output, size_t global_size_x,
                     size_t global_size_y, size_t local_size_x,
                     size_t local_size_y, cl_ulong *ts) {
  cl_int status;

  const size_t output_width = 4ul;
  const size_t in_size =
      global_size_x * global_size_y * input_width * sizeof(cl_ulong);
  const size_t out_size =
      global_size_x * global_size_y * output_width * sizeof(cl_ulong);

  // input is supplied to kernel by this buffer
  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &status);

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
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &status);

  // input being copied to device memory
  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, in_size, input, 0,
                                NULL, &evt_0);
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
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, output, 1,
                               &evt_5, &evt_6);

  status = clWaitForEvents(1, &evt_6);

  // figure out kernel execution time only when asked to
  if (ts != NULL) {
    cl_ulong start, end;
    status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start, NULL);
    check(status);
    status = clGetEventProfilingInfo(evt_5, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &end, NULL);
    check(status);

    *ts = (end - start);
  }

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
// while in input each was of width 8, because two input rescue prime digests
// are being merged
cl_int merge(cl_context ctx, cl_command_queue cq, cl_kernel krnl,
             cl_ulong *input, cl_ulong *output, size_t global_size_x,
             size_t global_size_y, size_t local_size_x, size_t local_size_y,
             cl_ulong *ts) {
  cl_int status;

  const size_t in_width = 8ul;
  const size_t out_width = 4ul;
  const size_t in_size =
      global_size_x * global_size_y * in_width * sizeof(cl_ulong);
  const size_t out_size =
      global_size_x * global_size_y * out_width * sizeof(cl_ulong);

  // input is supplied to kernel by this buffer
  cl_mem in_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &status);

  // Following three buffers will be storing rescue prime hash constants
  cl_mem mds_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(MDS), NULL, &status);
  cl_mem ark1_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK1), NULL, &status);
  cl_mem ark2_buf =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(ARK2), NULL, &status);

  // output to be placed here, after kernel completes hash computation
  cl_mem out_buf =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &status);

  // input being copied to device memory
  cl_event evt_0;
  status = clEnqueueWriteBuffer(cq, in_buf, CL_FALSE, 0, in_size, input, 0,
                                NULL, &evt_0);

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
  status = clEnqueueReadBuffer(cq, out_buf, CL_FALSE, 0, out_size, output, 1,
                               &evt_4, &evt_5);

  status = clWaitForEvents(1, &evt_5);

  // figure out how long kernel execution originally took
  // only when function caller wants this function to figure out
  if (ts != NULL) {
    cl_ulong start, end;
    status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start, NULL);
    check(status);
    status = clGetEventProfilingInfo(evt_4, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &end, NULL);
    check(status);

    *ts = (end - start);
  }

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
  status = hash_elements(ctx, cq, hash_krnl, in, 8, out, 1, 2, 1, 2, NULL);
  // Then merge 8 consequtive elements together, such that they are
  // interpreted to be two rescue prime hash digests concatenated
  // one after another
  //
  // Do this thing twice i.e. using two work-items
  status = merge(ctx, cq, merge_krnl, in, out + 8, 1, 2, 1, 2, NULL);

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
