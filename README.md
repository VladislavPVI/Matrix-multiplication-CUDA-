# Matrix multiplication CUDA
Matrix-matrix (N*N) multiplication using CPU and GPU

*Microsoft visual studio 19 +  CUDA Toolkit 11*

Build and Run
-------------

1. Install Microsoft Visual Studio.
2. Install CUDA Toolkit (Nvidea GPU with CUDA-support required).
3. Make new CUDA-project.
4. Enjoy.

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G860 |
| RAM  | 6 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB |
| OS   | Windows 10 64-bit  |

## Results

Average results after 100 times of runs. Matrix elements type is integer.

|    Size     |          CPU        |         GPU       | Acceleration |
|-------------|---------------------|-------------------|--------------|
| 64 х 64   | 1 ms               | 0.2 ms            |    5      |
| 128 х 128   | 9 ms               | 1.3 ms            |    6.9      |
| 256 х 256   | 74 ms               | 9.1 ms            |    8.1      |
| 512 х 512   | 704 ms              | 88.8 ms             |    7.9      |
| 1024 х 1024 | 22159 ms   | 548.9 ms            |    40.3      |
| 2048 х 2048 | 176250 ms | 4378 ms |    40.2      |

