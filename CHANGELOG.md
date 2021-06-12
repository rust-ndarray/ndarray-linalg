Unreleased
-----------

Updated dependencies
---------------------
- ndarray 0.15 https://github.com/rust-ndarray/ndarray-linalg/pull/273
- cauchy 0.4 (num-complex 0.4, rand 0.8), lapack 0.18 https://github.com/rust-ndarray/ndarray-linalg/pull/276

Fixed
-----
- Fix memory layout of the output of inverse of LUFactorized https://github.com/rust-ndarray/ndarray-linalg/pull/297
- Fix Eig for column-major arrays with real elements https://github.com/rust-ndarray/ndarray-linalg/pull/298
- Fix Solve::solve_h_* for complex inputs with standard layout https://github.com/rust-ndarray/ndarray-linalg/pull/296
- Add checks for matching shapes in Solve, SolveH, and EighInplace https://github.com/rust-ndarray/ndarray-linalg/pull/290

Changed
--------
- Avoid unnecessary calculation of eigenvectors in EigVals::eigvals and relax the DataMut bound https://github.com/rust-ndarray/ndarray-linalg/pull/286
- Add basic documentation for eigh module https://github.com/rust-ndarray/ndarray-linalg/pull/283
- Port usage of uninitialized to build_uninit https://github.com/rust-ndarray/ndarray-linalg/pull/287
- Relax type bounds for LeastSquaresSvd family https://github.com/rust-ndarray/ndarray-linalg/pull/272

0.13.1 - 13 March 2021
=====================

Changed
--------
- Not require continious layout for trace https://github.com/rust-ndarray/ndarray-linalg/pull/263

Maintenance
-----------
- Fix doc.rs for KaTeX integration https://github.com/rust-ndarray/ndarray-linalg/issues/268

0.13.0 - 20 Feb 2021
=====================

https://github.com/rust-ndarray/ndarray-linalg/milestone/5

Updated dependencies
---------------------
- ndarray 0.14 https://github.com/rust-ndarray/ndarray-linalg/pull/258
- cauchy 0.3.0 (num-complex 0.3.1, rand 0.7.3), lapack 0.17.0 https://github.com/rust-ndarray/ndarray-linalg/pull/260

### optional dependencies

- openblas-src 0.10.2 https://github.com/rust-ndarray/ndarray-linalg/pull/253
- intel-mkl-src 0.6.0 https://github.com/rust-ndarray/ndarray-linalg/pull/204

Added
------
- Split out `ndarray_linalg::lapack` as "lax" crate https://github.com/rust-ndarray/ndarray-linalg/pull/207
  - cargo-workspace https://github.com/rust-ndarray/ndarray-linalg/pull/209

Changed
--------
- Dual license, MIT or Apache-2.0 License https://github.com/rust-ndarray/ndarray-linalg/pull/262
- Revise tests for least-square problem https://github.com/rust-ndarray/ndarray-linalg/pull/227
- Support static link to LAPACK backend https://github.com/rust-ndarray/ndarray-linalg/pull/204
- Drop LAPACKE dependence, and rewrite them in Rust (see below) https://github.com/rust-ndarray/ndarray-linalg/pull/206
- Named record like `C { row: i32, lda: i32 }` instead of enum for `MatrixLayout` https://github.com/rust-ndarray/ndarray-linalg/pull/211
- Split LAPACK error into computational failure and invalid values https://github.com/rust-ndarray/ndarray-linalg/pull/210
- Use thiserror crate https://github.com/rust-ndarray/ndarray-linalg/pull/208

### LAPACKE rewrite

- Cholesky https://github.com/rust-ndarray/ndarray-linalg/pull/225
- Eigenvalue for general matrix https://github.com/rust-ndarray/ndarray-linalg/pull/212
- Eigenvalue for symmetric/Hermitian matrix https://github.com/rust-ndarray/ndarray-linalg/pull/217
- least squares problem https://github.com/rust-ndarray/ndarray-linalg/pull/220
- QR decomposition https://github.com/rust-ndarray/ndarray-linalg/pull/224
- LU decomposition https://github.com/rust-ndarray/ndarray-linalg/pull/213
- LDL decomposition https://github.com/rust-ndarray/ndarray-linalg/pull/216
- SVD https://github.com/rust-ndarray/ndarray-linalg/pull/218
- SVD divid-and-conquer https://github.com/rust-ndarray/ndarray-linalg/pull/219
- Tridiagonal https://github.com/rust-ndarray/ndarray-linalg/pull/235

Maintenance
-----------
- Coverage report using codecov https://github.com/rust-ndarray/ndarray-linalg/pull/215
- Fix for clippy, and add CI check https://github.com/rust-ndarray/ndarray-linalg/pull/205

0.12.1 - 28 June 2020
======================

Added
------
- Tridiagonal matrix support https://github.com/rust-ndarray/ndarray-linalg/pull/196
- KaTeX support in rustdoc https://github.com/rust-ndarray/ndarray-linalg/pull/202
- Least square problems https://github.com/rust-ndarray/ndarray-linalg/pull/197
- LOBPCG solver https://github.com/rust-ndarray/ndarray-linalg/pull/184

Changed
-------
- Grouping and Plot in benchmark https://github.com/rust-ndarray/ndarray-linalg/pull/200
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
======================

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
===================

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
