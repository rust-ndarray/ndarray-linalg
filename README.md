ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![wercker status](https://app.wercker.com/status/f04aeba682ea6e79577e15bd946344a5/s/master "wercker status")](https://app.wercker.com/project/byKey/f04aeba682ea6e79577e15bd946344a5)

Linear algebra package for Rust with [rust-ndarray](https://github.com/bluss/rust-ndarray).

LAPACKE Backend
----------------

Currently three LAPACKE implementations are supported and tested:

- [OpenBLAS](https://github.com/cmr/openblas-src)
- [Netlib](https://github.com/cmr/netlib-src)
- [Intel MKL](https://github.com/termoshtt/rust-intel-mkl) (non-free license, see the linked page)

There are two ways to link LAPACKE backend:

### backend features (recommended)
There are three features corresponding to the backend implementations (`openblas` / `netlib` / `intel-mkl`):

```toml
[depencdencies]
ndarray = "0.11"
ndarray-linalg = { version = "0.9", features = ["openblas"] }
```

### link backend crate manually
For the sake of linking flexibility, you can provide LAPACKE implementation (as an `extern crate`) yourself.
You should link a LAPACKE implementation to a final crate (like binary executable or dylib) only, not to a Rust library.

```toml
[depencdencies]
ndarray = "0.11"
ndarray-linalg = "0.9"
openblas-src = "0.5" # or another backend of your choice

```

You must add `extern crate` to your code in this case:

```rust
extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice
```

### For librarian
If you creating a library depending on this crate, we encourage you not to link any backend for flexibility:

```toml
[depencdencies]
ndarray = "0.11"
ndarray-linalg = { version = "0.9", default-features = false }
```

However, if you hope simplicity instead of the flexibility, you can link your favorite backend in the way described above.

### Tests and Examples

To run tests or examples for `ndarray-linalg`, you must specify the desired
backend. For example, you can run the tests with the OpenBLAS backend like
this:

```sh
cargo test --features=openblas
```

Examples
---------
See [examples](https://github.com/termoshtt/ndarray-linalg/tree/master/examples) directory.

Note that to run an example, you must specify the desired backend. For example,
you can run the the `solve` example with the OpenBLAS backend like this:

```sh
cargo run --example solve --features=openblas
```
