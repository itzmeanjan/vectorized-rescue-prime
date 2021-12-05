#include <rescue_prime.h>
#include <rescue_prime_constants.h>
#include <stdio.h>

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

  printf("output hash :\n");
  for (size_t i = 0; i < out_width; i++) {
    printf("%lu\t", out_arr[i]);
  }
  printf("\n");

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
