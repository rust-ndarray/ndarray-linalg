use ndarray::*;
use ndarray_linalg::*;

// Solve `Ax=b` for tridiagonal matrix
fn solve() -> Result<(), error::LinalgError> {
    let mut a: Array2<f64> = random((3, 3));
    let b: Array1<f64> = random(3);
    a[[0, 2]] = 0.0;
    a[[2, 0]] = 0.0;
    let _x = a.solve_tridiagonal(&b)?;
    Ok(())
}

// Solve `Ax=b` for many b with fixed A
fn factorize() -> Result<(), error::LinalgError> {
    let mut a: Array2<f64> = random((3, 3));
    a[[0, 2]] = 0.0;
    a[[2, 0]] = 0.0;
    let f = a.factorize_tridiagonal()?; // LU factorize A (A is *not* consumed)
    for _ in 0..10 {
        let b: Array1<f64> = random(3);
        let _x = f.solve_tridiagonal_into(b)?; // solve Ax=b using factorized L, U
    }
    Ok(())
}

fn main() {
    solve().unwrap();
    factorize().unwrap();
}