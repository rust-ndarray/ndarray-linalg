ndarray-linalg
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-linalg)](https://crates.io/crates/ndarray-linalg)
[![docs.rs](https://docs.rs/ndarray-linalg/badge.svg)](https://docs.rs/ndarray-linalg)
[![wercker status](https://app.wercker.com/status/f04aeba682ea6e79577e15bd946344a5/s/master "wercker status")](https://app.wercker.com/project/byKey/f04aeba682ea6e79577e15bd946344a5)

Linear algebra package for [rust-ndarray](https://github.com/bluss/rust-ndarray) using LAPACK via [stainless-steel/lapack](https://github.com/stainless-steel/lapack)

Examples
---------

```rust
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;

fn main() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs) = a.clone().eigh().unwrap();
    println!("eigenvalues = \n{:?}", e);
    println!("V = \n{:?}", vecs);
    let av = a.dot(&vecs);
    println!("AV = \n{:?}", av);
}
```

See complete example at [src/bin/main.rs](src/bin/main.rs).

Progress
---------
Some algorithms have not been implemented yet. See [#6](https://github.com/termoshtt/ndarray-linalg/issues/6).

Similar Projects
-----------------
- [linxal](https://github.com/masonium/linxal)
