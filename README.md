# vectorized-rescue-prime
Vectorized, Accelerated Rescue Prime Hash Function Implementation, using OpenCL

## Motivation

I've already implemented ZkSTARK-friendly Rescue Prime Hash function for 64-bit prime field `2^64 - 2^32 + 1`, using SYCL/ DPC++, targeting accelerators, running in Data Parallel environments. But in that implementation Rescue Hash state is represented in terms of array of 12 field elements. So I can't make use of vector intrinsics for operating on state, while applying Rescue permutation rounds. Anyway Rescue Prime hash function itself is not parallel, so it's only useful if we've lots of same width input to operate on, we can offload computation onto accelerators. Which is why I decided to take advantage of available vector intrinsics for representing and operating on Rescue Hash state. But note, Rescue Hash function has a fixed state width of 12 and in OpenCL, I can't have a vector of length 12. So instead I'm using `ulong16`, which is a 16-element wide vector with each element being unsigned 64-bit integer, for representing hash state. This comes with an extra cost, as you've guessed. When I've M x N -many independent Rescue Prime Hashes to be computed, each of those M x N -many OpenCL work-items need to spend 4 * 64 bits = 256-bits for padding Rescue Hash state, so that I can use `ulong16`, but only first 12 elements are useful. I'm required to waste M x N x 32 -bytes of memory, when computing M x N -many independent hashes. But computation itself should be fast, because when using vector intrinsics for operating on Hash state, OpenCL device compiler should generate machine code which can make use of SIMD instructions very well. 

> [64-bit Prime field](https://github.com/itzmeanjan/ff-gpu/blob/2c78ddf2cf4ff2d1b678e811761d0f06a4c42f73/include/ff_p.hpp#L4-L7) of interest.

> You may want to read about [Rescue Prime Hash function](https://eprint.iacr.org/2020/1143.pdf).

> My implementation collects quite a lot of motivation from [here](https://github.com/novifinancial/winterfell/tree/4eeb4670387f3682fa0841e09cdcbe1d43302bf3/crypto#rescue-hash-function-implementation)

> My [other implementation](https://github.com/itzmeanjan/ff-gpu/blob/9c57cb13e4b2d96a084da96d558fe3d4707bfcb7/rescue_prime.cpp) of Rescue Prime Hash, in SYCL/ DPC++, non-vectorized.

> [Benchmark results](https://github.com/itzmeanjan/ff-gpu/blob/a0a4ae7e945a4d27f615e1e00a8625566d56159a/benchmarks/rescue_prime.md) of SYCL/ DPC++ Rescue Prime Hash function.

## Prerequisite 

- You must have OpenCL development headers and ICD installed. Take a look [here](https://github.com/kenba/cl3/blob/78f04cb2d55fd313816daeb9d0bb33ea1820cb91/docs/opencl_installation.md)
- You may also install `clinfo`, just to get an overview of your OpenCL installation.

```bash
sudo apt-get install clinfo
```

- GCC is required for compiling host code.

```bash
$ gcc --version
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
```

- You'll also need to have `make`, `clang-format`.

## Usage

- Just run 👇 to compile host code; device code compilation is done at runtime.

```bash
make
```

- Run executable using

```bash
./run
```

- It should run some standard test cases against kernels and finally a benchmark suite is run on Rescue Prime `hash_elements` and `merge` kernel function.

> You probably want to take a look at vectorized `hash_elements`/ `merge` [OpenCL kernels](https://github.com/itzmeanjan/vectorized-rescue-prime/blob/f2316e3b8425e0484e69817e3e45ac0c3d60187b/kernel.cl#L307-L428), if you want to use it in your project.

> You can find relevant examples [here](https://github.com/itzmeanjan/vectorized-rescue-prime/blob/6d2e242ce1af02f4c3d24a182b6068b42f6e1bfb/rescue_prime.c#L630-L828)

## Benchmark

For setting up benchmark in data parallel environment, I use one 2D computational grid of size M x N, and launch `hash_elements`, `merge` kernels with input of size M x N x 8. So each work-item will read 8 randomly generated 64-bit prime field elements and total of M x N-many independent hashes to be computed/ merging of hash digests to be performed. Output to be written into global memory with OpenCL vector store intrinsics. For writing output I provide with one buffer of size M x N x 4, so that each work-item can produce output of width 4-prime field elements i.e. 256-bits. In following benchmark results I only show time to compute hash/ merge by enabling profiling in command queue, host-device and device-host data transfer costs are not included, though they are performed just to ensure compiler doesn't end up optimizing too much so that benchmark suite doesn't run kernels as I expect it to be run.

### Intel CPU with OpenCL

```bash
make && ./run
```

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

kernel build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Options used by backend compiler: -cl-std=CL2.0 -w
Device build done
Kernel <test_apply_sbox> was not vectorized
Kernel <test_reduce_sum_vec2> was successfully vectorized (8)
Kernel <test_apply_mds> was not vectorized
Kernel <test_apply_inv_sbox> was not vectorized
Kernel <test_apply_rescue_permutation> was not vectorized
Kernel <hash_elements> was not vectorized
Kernel <merge> was not vectorized
Kernel <build_merkle_tree_tip_seq> was not vectorized
Done.

passed apply_sbox tests !
passed apply_inv_sbox tests !
passed apply_rescue_permutation tests !
passed apply_mds tests !
passed reduce_sum_vec2 tests !
passed merge tests !
passed build_merkle_nodes tests !

Rescue Prime Hash Benchmark

  hash_elements		  128 x   128		              345.25 ms		       47455.50 hashes/ sec
  hash_elements		  256 x   256		             1382.75 ms		       47395.51 hashes/ sec
  hash_elements		  512 x   512		             5473.14 ms		       47896.49 hashes/ sec
  hash_elements		 1024 x  1024		            22283.28 ms		       47056.63 hashes/ sec

Rescue Prime Merge Benchmark

          merge		  128 x   128		              342.02 ms		       47903.70 merges/ sec
          merge		  256 x   256		             1369.43 ms		       47856.36 merges/ sec
          merge		  512 x   512		             5509.15 ms		       47583.40 merges/ sec
          merge		 1024 x  1024		            21831.19 ms		       48031.09 merges/ sec

Rescue Prime Merkle Tree Benchmark

       merklize		   1048576 leaves		            21846.56 ms
       merklize		   2097152 leaves		            43669.84 ms
       merklize		   4194304 leaves		            87320.22 ms
       merklize		   8388608 leaves		            175640.44 ms
```

### Nvidia Tesla V100 GPU with OpenCL

```bash
make && ./run
```

```bash
running on Tesla V100-SXM2-16GB

kernel build log:



passed apply_sbox tests !
passed apply_inv_sbox tests !
passed apply_rescue_permutation tests !
passed apply_mds tests !
passed reduce_sum_vec2 tests !
passed merge tests !
passed build_merkle_nodes tests !

Rescue Prime Hash Benchmark

  hash_elements		  128 x   128		                2.05 ms		     7992007.99 hashes/ sec
  hash_elements		  256 x   256		                7.62 ms		     8599838.75 hashes/ sec
  hash_elements		  512 x   512		               28.02 ms		     9357067.14 hashes/ sec
  hash_elements		 1024 x  1024		              102.22 ms		    10257848.66 hashes/ sec

Rescue Prime Merge Benchmark

          merge		  128 x   128		                1.93 ms		     8474576.27 merges/ sec
          merge		  256 x   256		                7.15 ms		     9169054.44 merges/ sec
          merge		  512 x   512		               26.40 ms		     9930948.87 merges/ sec
          merge		 1024 x  1024		              105.11 ms		     9975548.22 merges/ sec

Rescue Prime Merkle Tree Benchmark

       merklize		   1048576 leaves		              130.50 ms
       merklize		   2097152 leaves		              235.47 ms
       merklize		   4194304 leaves		              445.62 ms
       merklize		   8388608 leaves		              864.30 ms
```
