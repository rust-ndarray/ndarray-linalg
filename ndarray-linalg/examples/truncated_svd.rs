extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a = arr2(&[[3., 2., 2.], [2., 3., -2.]]);

    // calculate the truncated singular value decomposition for 2 singular values
    let result = TruncatedSvd::new(a, TruncatedOrder::Largest)
        .decompose(2)
        .unwrap();

    // acquire singular values, left-singular vectors and right-singular vectors
    let (u, sigma, v_t) = result.values_vectors();
    println!("Result of the singular value decomposition A = UΣV^T:");
    println!(" === U ===");
    println!("{:?}", u);
    println!(" === Σ ===");
    println!("{:?}", Array2::from_diag(&sigma));
    println!(" === V^T ===");
    println!("{:?}", v_t);
}
