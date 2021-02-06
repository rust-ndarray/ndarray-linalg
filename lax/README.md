Linear Algebra eXtension (LAX)
===============================

ndarray-free safe Rust wrapper for LAPACK FFI for implementing ndarray-linalg crate.
This crate responsibles for

- Linking to LAPACK shared/static libraries
- Dispatching to LAPACK routines based on scalar types by using `Lapack` trait

Features
---------

| Feature          | Link type      | Requirements        | Description |
|:-----------------|:---------------|:--------------------|:----------------------------------------|
| openblas-static  | static         | gcc, gfortran, make | Build OpenBLAS, and link it statically  |
| openblas-system  | dynamic/static | -                   | Seek OpenBLAS in system (e.g. `/usr/lib/libopenblas.so`), and link it. Shared library (use dynamic linking) is prior to static library (`libopenblas.a`) |
| netlib-static    | static         | gfortran, make      | Same as openblas-static except for using reference LAPACK |
| netlib-system    | dynamic/static | -                   | Same as openblas-system except for using reference LAPACK |
| intel-mkl-static | static         | (pkg-config)        | Seek static library of Intel MKL from system, or download if not found, and link it statically |
| intel-mkl-system | dynamic        | (pkg-config)        | Seek shared library of Intel MKL from system, and link it dynamically |

- You must use **just one** feature of them.
- `pkg-config` is used for searching Intel MKL packages in system, and it is optional. See [intel-mkl-src/README.md](https://github.com/rust-math/intel-mkl-src/blob/master/README.md#how-to-find-system-mkl-libraries) for detail.
