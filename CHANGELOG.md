Unreleased
===========

Added
------
- LOBPCG solver [#184](../../pull/184)

Changed
-------
- `Clone` trait for `LUFactorized` [#192](../../192)

Maintenance
-----------
- Use GitHub Actions instead of Azure Pipeline [#193](../../193)
- Test cargo-fmt on CI [#194](../../194)

0.12.0 - 14 Oct 2019
====================

Added
-----
- SVD by divide-and-conquer [#164](../../pull/164)
- Householder reflection [#154](../../pull/154), Arnoldi iteration [#155](../../pull/155)

Changed
----------
- Replace `operator::Operator*` traits by new `LinearOperator trait` [#159](../../pull/159)
- ndarray 0.13.0 [#172](../../pull/172)
- blas-src 0.4.0, lapack-src 0.4.0, openblas-src 0.7.0 [#174](../../pull/174)
- restore `static` feature flag

0.11.1 - 12 June 2019
---------------------

- Hotfix for document generation [#153](../../pull/153)

0.11.0 - 12 June 2019
====================

Added
--------
- Dependency to cauchy 0.2 [#139](../../pull/139)
- `generate::random_{unitary,regular}` for debug use [#140](../../pull/140) 
- `krylov` submodule
  - modified Gram-Schmit [#149](../../pull/149), [#150](../../pull/150)
  - Krylov subspace methods are not implemented yet.

Removed
----------
- `static` feature [#136](../../pull/136)
  - See README for detail
- `accelerate` feature [#141](../../pull/141)
- Dependencies to derive-new, procedurals

Changed
---------
- Switch CI service: Circle CI -> Azure Pipeline [#141](../../pull/141)
- submodule `lapack_traits` is renamed to `lapack` [#139](../../pull/139)
- `ndarray_linalg::Scalar` trait is split into two parts [#139](../../pull/139)
  - [cauchy::Scalar](https://docs.rs/cauchy/0.2.0/cauchy/trait.Scalar.html) is a refined real/complex common trait
  - `lapack::Lapack` is a trait for primitive types which LAPACK supports
- Error type becomes simple [#118](../../pull/118) [#127](../../pull/127)
- Assertions becomes more verbose [#147](../../pull/147)
- blas-src 0.3, lapack-src 0.3
  - intel-mkl-src becomes 0.4, which supports Windows! [#146](../../pull/146)

0.10.0 - 2 Sep 2018
=======

Update Dependencies
--------------------

- ndarray 0.12
- rand 0.5
- num-complex 0.2
- openblas-src 0.6
- lapacke 0.2

See also [#110](../../pull/110)

Added
------
- serde-1 feature gate [#99](../../pull/99), [#116](../../pull/116)
