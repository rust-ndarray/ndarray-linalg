
extern crate ndarray;
extern crate ndarray_linalg;
#[cfg(feature = "lapack-src")]
extern crate lapack_src;

use ndarray::*;
use ndarray_linalg::*;

// Solve `Ax=b`
fn solve() -> Result<(), error::LinalgError> {
    let a: Array2<f64> = random((3, 3));
    let b: Array1<f64> = random(3);
    let _x = a.solve(&b)?;
    Ok(())
}

// Solve `Ax=b` for many b with fixed A
fn factorize() -> Result<(), error::LinalgError> {
    let a: Array2<f64> = random((3, 3));
    let f = a.factorize_into()?; // LU factorize A (A is consumed)
    for _ in 0..10 {
        let b: Array1<f64> = random(3);
        let _x = f.solve_into(b)?; // solve Ax=b using factorized L, U
    }
    Ok(())
}

fn main() {
    solve().unwrap();
    factorize().unwrap();
}
