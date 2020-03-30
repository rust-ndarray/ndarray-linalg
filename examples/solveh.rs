extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

// Solve `Ax=b` for Hermite matrix A
fn solve() -> Result<(), error::LinalgError> {
    let a: Array2<c64> = random_hermite(3); // complex Hermite positive definite matrix
    let b: Array1<c64> = random(3);
    println!("b = {:?}", &b);
    let x = a.solveh(&b)?;
    println!("Ax = {:?}", a.dot(&x));
    Ok(())
}

// Solve `Ax=b` for many b with fixed A
fn factorize() -> Result<(), error::LinalgError> {
    let a: Array2<f64> = random_hpd(3);
    let f = a.factorizeh_into()?;
    // once factorized, you can use it several times:
    for _ in 0..10 {
        let b: Array1<f64> = random(3);
        let _x = f.solveh_into(b)?;
    }
    Ok(())
}

fn main() {
    solve().unwrap();
    factorize().unwrap();
}
