Unreleased
===========

Added
------
- Least square problems https://github.com/rust-ndarray/ndarray-linalg/pull/197
- LOBPCG solver https://github.com/rust-ndarray/ndarray-linalg/pull/184

Changed
-------
- `Clone` trait for `LUFactorized` https://github.com/rust-ndarray/ndarray-linalg/pull/192

Maintenance
-----------
- Fix repository URL https://github.com/rust-ndarray/ndarray-linalg/pull/198
- Use GitHub Actions instead of Azure Pipeline https://github.com/rust-ndarray/ndarray-linalg/pull/193
- Test cargo-fmt on CI https://github.com/rust-ndarray/ndarray-linalg/pull/194

0.12.0 - 14 Oct 2019
====================

Added
-----
- SVD by divide-and-conquer https://github.com/rust-ndarray/ndarray-linalg/pull/164
- Householder reflection https://github.com/rust-ndarray/ndarray-linalg/pull/154
- Arnoldi iteration https://github.com/rust-ndarray/ndarray-linalg/pull/155

Changed
----------
- Replace `operator::Operator*` traits by new `LinearOperator trait` https://github.com/rust-ndarray/ndarray-linalg/pull/159
- ndarray 0.13.0 https://github.com/rust-ndarray/ndarray-linalg/pull/172
- blas-src 0.4.0, lapack-src 0.4.0, openblas-src 0.7.0 https://github.com/rust-ndarray/ndarray-linalg/pull/174
- restore `static` feature flag

0.11.1 - 12 June 2019
---------------------

- Hotfix for document generation https://github.com/rust-ndarray/ndarray-linalg/pull/153

0.11.0 - 12 June 2019
====================

Added
--------
- Dependency to cauchy 0.2 https://github.com/rust-ndarray/ndarray-linalg/pull/139
- `generate::random_{unitary,regular}` for debug use https://github.com/rust-ndarray/ndarray-linalg/pull/140
- `krylov` submodule
  - modified Gram-Schmit https://github.com/rust-ndarray/ndarray-linalg/pull/149 https://github.com/rust-ndarray/ndarray-linalg/pull/150
  - Krylov subspace methods are not implemented yet.

Removed
----------
- `static` feature https://github.com/rust-ndarray/ndarray-linalg/pull/136
  - See README for detail
- `accelerate` feature https://github.com/rust-ndarray/ndarray-linalg/pull/141
- Dependencies to derive-new, procedurals

Changed
---------
- Switch CI service: Circle CI -> Azure Pipeline https://github.com/rust-ndarray/ndarray-linalg/pull/141
- submodule `lapack_traits` is renamed to https://github.com/rust-ndarray/ndarray-linalg/pull/139
- `ndarray_linalg::Scalar` trait is split into two parts https://github.com/rust-ndarray/ndarray-linalg/pull/139
  - [cauchy::Scalar](https://docs.rs/cauchy/0.2.0/cauchy/trait.Scalar.html) is a refined real/complex common trait
  - `lapack::Lapack` is a trait for primitive types which LAPACK supports
- Error type becomes simple https://github.com/rust-ndarray/ndarray-linalg/pull/118 https://github.com/rust-ndarray/ndarray-linalg/pull/127
- Assertions becomes more verbose https://github.com/rust-ndarray/ndarray-linalg/pull/147
- blas-src 0.3, lapack-src 0.3
  - intel-mkl-src becomes 0.4, which supports Windows! https://github.com/rust-ndarray/ndarray-linalg/pull/146

0.10.0 - 2 Sep 2018
=======

Update Dependencies
--------------------

- ndarray 0.12
- rand 0.5
- num-complex 0.2
- openblas-src 0.6
- lapacke 0.2

See also https://github.com/rust-ndarray/ndarray-linalg/pull/110

Added
------
- serde-1 feature gate https://github.com/rust-ndarray/ndarray-linalg/pull/99, https://github.com/rust-ndarray/ndarray-linalg/pull/116
