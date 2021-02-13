ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)

Linear algebra package for Rust with [ndarray](https://github.com/rust-ndarray/ndarray) based on external LAPACK implementations.

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/ndarray-linalg/examples) directory.

**Note**: To run examples, you must specify which backend will be used (as described below).
For example, you can execute the [solve](ndarray-linalg/examples/solve.rs) example with the OpenBLAS backend like this:

```sh
cargo run --example solve --features=openblas
```

and run all tests of ndarray-linalg with OpenBLAS

```sh
cargo test --features=openblas
```

Backend Features
-----------------

There are three LAPACK source crates:

- [openblas-src](https://github.com/blas-lapack-rs/openblas-src)
- [netlib-src](https://github.com/blas-lapack-rs/netlib-src)
- [intel-mkl-src](https://github.com/rust-math/rust-intel-mkl)

`ndarray_linalg` must link **just one** of them for LAPACK FFI.

```toml
[dependencies]
ndarray = "0.14"
ndarray-linalg = { version = "0.13", features = ["openblas-static"] }
```

Supported features are following:

| Feature          | Link type      | Requirements        | Description                                                                                    |
|:-----------------|:---------------|:--------------------|:-----------------------------------------------------------------------------------------------|
| openblas-static  | static         | gcc, gfortran, make | Build OpenBLAS in your project, and link it statically                                         |
| openblas-system  | dynamic/static | libopenblas-dev     | Seek OpenBLAS in system, and link it                                                           |
| netlib-static    | static         | gfortran, make      | Same as openblas-static except for using reference LAPACK                                      |
| netlib-system    | dynamic/static | liblapack-dev       | Same as openblas-system except for using reference LAPACK                                      |
| intel-mkl-static | static         | (pkg-config)        | Seek static library of Intel MKL from system, or download if not found, and link it statically |
| intel-mkl-system | dynamic        | (pkg-config)        | Seek shared library of Intel MKL from system, and link it dynamically                          |

- You must use **just one** feature of them.
- `dynamic/static` means it depends on what is found in the system. When the system has `/usr/lib/libopenblas.so`, it will be linked dynamically, and `/usr/lib/libopenblas.a` will be linked statically. Dynamic linking is prior to static linking.
- Requirements notices:
  - `gcc` and `gfortran` can be another compiler, e.g. `icc` and `ifort`.
  - `libopenblas-dev` is package name in Debian, Ubuntu, and other derived distributions.
    There are several binary packages of OpenBLAS, i.e. `libopenblas-{openmp,pthread,serial}-dev`.
    It can be other names in other distributions, e.g. Fedora, ArchLinux, and so on.
  - `pkg-config` is used for searching Intel MKL packages in system, and it is optional. See [intel-mkl-src/README.md](https://github.com/rust-math/intel-mkl-src/blob/master/README.md#how-to-find-system-mkl-libraries) for detail.

### For library developer

If you creating a library depending on this crate, we encourage you not to link any backend:

```toml
[dependencies]
ndarray = "0.13"
ndarray-linalg = "0.12"
```

The cargo's feature is additive. If your library (saying `lib1`) set a feature `openblas-static`,
the application using `lib1` builds ndarray_linalg with `openblas-static` feature though they want to use `intel-mkl-static` backend.

See [the cargo reference](https://doc.rust-lang.org/cargo/reference/features.html) for detail

Tested Environments
--------------------

Only x86_64 system is supported currently.

|Backend  | Linux | Windows | macOS |
|:--------|:-----:|:-------:|:-----:|
|OpenBLAS |✔️      |-        |-      |
|Netlib   |✔️      |-        |-      |
|Intel MKL|✔️      |✔️        |✔️      |

Generate document with KaTeX
------------------------------

You need to set `RUSTDOCFLAGS` explicitly:

```shell
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps
```

This **only** works for `--no-deps` build because `katex-header.html` does not exists for dependent crates.
If you wish to set `RUSTDOCFLAGS` automatically in this crate, you can put [.cargo/config](https://doc.rust-lang.org/cargo/reference/config.html):

```toml
[build]
rustdocflags = ["--html-in-header", "katex-header.html"]
```

But, be sure that this works only for `--no-deps`. `cargo doc` will fail with this `.cargo/config`.
