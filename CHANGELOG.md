Unreleased
===========

Added
--------
- Dependency to cauchy 0.2 [#139](../../pull/139)
- `generate::random_{unitary,regular}` for debug use [#140](../../pull/140) 

Removed
----------
- `static` feature gate [#136](../../pull/136) 
- Dependencies to derive-new, procedurals

Changed
---------
- submodule `lapack_traits` is renamed to `lapack` [#139](../../pull/139)
- `ndarray_linalg::Scalar` trait is split into two parts [#139](../../pull/139)
  - [cauchy::Scalar](https://docs.rs/cauchy/0.2.0/cauchy/trait.Scalar.html) is a refined real/complex common trait
  - `lapack::Lapack` is a trait for primitive types which LAPACK supports
- Error type becomes simple [#118](../../pull/118) [#127](../../pull/127)

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
