extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let n = 10;
    let v = random_unitary(n);

    // set eigenvalues in decreasing order
    let t = Array1::linspace(n as f64, -(n as f64), n);

    println!("Generate spectrum: {:?}", &t);

    let t = Array2::from_diag(&t);
    let a = v.dot(&t.dot(&v.t()));

    // calculate the truncated eigenproblem decomposition
    for (val, _) in TruncatedEig::new(a, TruncatedOrder::Largest) {
        println!("Found eigenvalue {}", val[0]);
    }
}
