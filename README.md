ndarray-linalg [![Document](https://img.shields.io/badge/document-0.1-blue.svg)](https://termoshtt.github.io/ndarray-linalg/ndarray_linalg/index.html) [![wercker status](https://app.wercker.com/status/a45df26fa97eab7debf53b32fc576b35/s/master "wercker status")](https://app.wercker.com/project/byKey/a45df26fa97eab7debf53b32fc576b35)
===============
Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK via [stainless-steel/lapack](https://github.com/stainless-steel/lapack)

Examples
---------

```rust
let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
let (e, vecs) = a.eigh().unwrap();  // eigenvalue and eigenvectors (for Hermite matrix)
```

See complete example at [src/bin/main.rs](src/bin/main.rs).

Progress: WIP
---------
Only a few algorithms are implemented. See [#6](https://github.com/termoshtt/ndarray-linalg/issues/6).
