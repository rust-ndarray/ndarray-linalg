0.2.0 - 17 July 2021
=====================

Updated dependencies
---------------------
- cauchy 0.4 (num-complex 0.4, rand 0.8), lapack 0.18 https://github.com/rust-ndarray/ndarray-linalg/pull/276

Fixed
-----
- Fix memory layout of the output of inverse of LUFactorized https://github.com/rust-ndarray/ndarray-linalg/pull/297
- Fix Eig for column-major arrays with real elements https://github.com/rust-ndarray/ndarray-linalg/pull/298
- Fix Solve::solve_h_* for complex inputs with standard layout https://github.com/rust-ndarray/ndarray-linalg/pull/296
