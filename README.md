# vectorized-rescue-prime
Vectorized, Accelerated Rescue Prime Hash Function Implementation, using OpenCL

## Motivation

I've already implemented ZkSTARK-friendly Rescue Prime Hash function for 64-bit prime field `2^64 - 2^32 + 1`, using SYCL/ DPC++, targeting accelerators, running in Data Parallel environments. But in that implementation Rescue Hash state is represented in terms of array of 12 field elements. So I can't make use of vector intrinsics for operating on state, while applying Rescue permutation rounds. Anyway Rescue Prime hash function itself is not parallel, so it's only useful if we've lots of same width input to operate on, we can offload computation onto accelerators. Which is why I decided to take advantage of available vector intrinsics for representing and operating on Rescue Hash state. But note, Rescue Hash function has a fixed state width of 12 and in OpenCL, I can't have a vector of length 12. So instead I'm using `ulong16`, which is a 16-element wide vector with each element being unsigned 64-bit integer, for representing hash state. This comes with an extra cost, as you've guessed. When I've M x N -many independent Rescue Prime Hashes to be computed, each of those M x N -many OpenCL work-items need to spend 4 * 64 bits = 256-bits for padding Rescue Hash state, so that I can use `ulong16`, but only first 12 elements are useful. I'm required to waste M x N x 32 -bytes of memory, when computing M x N -many independent hashes. But computation itself should be fast, because when using vector instrinsics for operating on Hash state, OpenCL device compiler should generate machine code which can make use of SIMD instructions very well. 

> [64-bit Prime field](https://github.com/itzmeanjan/ff-gpu/blob/2c78ddf2cf4ff2d1b678e811761d0f06a4c42f73/include/ff_p.hpp#L4-L7) of interest.

> You may want to read about [Rescue Prime Hash function](https://eprint.iacr.org/2020/1143.pdf).

> My [other implementation](https://github.com/itzmeanjan/ff-gpu/blob/9c57cb13e4b2d96a084da96d558fe3d4707bfcb7/rescue_prime.cpp) of Rescue Prime Hash, in SYCL/ DPC++, non-vectorized.

> [Benchmark results](https://github.com/itzmeanjan/ff-gpu/blob/a0a4ae7e945a4d27f615e1e00a8625566d56159a/benchmarks/rescue_prime.md) of SYCL/ DPC++ Rescue Prime Hash function.

## Prerequisite 

- You must have OpenCL development headers, library installed. 
- You may also install `clinfo`, just get an overview of your OpenCL installation.

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

- Just run 👇 to compile host code, device code compilation is done at runtime.

```bash
make
```

- Run executable using 

```bash
./run
```

- It should run some standard test cases against kernels and finally a benchmark suite is run on Rescue Prime Hash function kernel.

> You probably want to take a look at vectorized `hash_elements` [OpenCL kernel](https://github.com/itzmeanjan/vectorized-rescue-prime/blob/fa5ec366d5955f08f3e5734b33bde842cfd570c6/kernel.cl#L320-L376), if you want to use it in your project.

## Benchmark

For setting up benchmark in data parallel environment, I use one 2D computational grid of size M x N, and launch `hash_elements` kernel with input of size M x N x 8. So each work-item will read 8 randomly generated 64-bit prime field elements and total of M x N-many independent hashes to be computed. Output to be written into global memory with OpenCL vector store instrinsics. For writing output I provide with one buffer of size M x N x 4, so that each work-item can produce output of width 4-prime field elements i.e. 256-bits. In following benchmark results I only show time to computes hashes, host-device and device-host data transfer costs are not included.

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
Done.

passed apply_sbox tests !
passed apply_inv_sbox tests !
passed apply_rescue_permutation tests !
passed apply_mds tests !
passed reduce_sum_vec2 tests !

Rescue Prime Hash Benchmark

  128 x   128		    341.97 ms		       47911.06 hashes/ sec
  256 x   256		   1366.71 ms		       47951.69 hashes/ sec
  512 x   512		   5463.32 ms		       47982.57 hashes/ sec
 1024 x  1024		  22157.56 ms		       47323.61 hashes/ sec
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

Rescue Prime Hash Benchmark

  128 x   128		      2.06 ms		     7941832.28 hashes/ sec
  256 x   256		      7.76 ms		     8448627.05 hashes/ sec
  512 x   512		     28.01 ms		     9358943.23 hashes/ sec
 1024 x  1024		    111.23 ms		     9426927.50 hashes/ sec
 2048 x  2048		    425.38 ms		     9860204.01 hashes/ sec
 4096 x  4096		   1628.74 ms		    10300764.70 hashes/ sec
 8192 x  8192		   6472.39 ms		    10368476.33 hashes/ sec
```
