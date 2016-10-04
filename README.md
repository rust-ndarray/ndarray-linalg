WIP: ndarray-linalg [![Build Status](https://travis-ci.org/termoshtt/ndarray-linalg.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-linalg)
===============
Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray)

Examples
---------

```rust
let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
let (e, vecs) = a.eigh().unwrap();  // eigenvalue and eigenvectors (for Hermite matrix)
```

See complete example at [src/main.rs](src/main.rs).

Progress
---------
- LAPACK bindings using [stainless-steel/lapack](https://github.com/stainless-steel/lapack) (for small matrix):
  - [ ] lu: LU factorization
  - [ ] qr: QR factorization
  - [ ] svd: singular-value decomposition (SVD)
  - [ ] inv: inverse matrix
  - [x] eigh: eigenvalue analysis for Hermite matrix
  - [ ] eig: eigenvalue analysis for general matrix
- Iterating methods (for large matrix):
  - [ ] SOR
  - [ ] BiCGSTAB
  - [ ] GMRES
