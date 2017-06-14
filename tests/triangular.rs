
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::prelude::*;

fn test1d<A, Sa, Sb, Tol>(uplo: UPLO, a: ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix1>, tol: Tol)
    where A: Field + Absolute<Output = Tol>,
          Sa: Data<Elem = A>,
          Sb: DataMut<Elem = A> + DataClone,
          Tol: RealField
{
    println!("a = {:?}", &a);
    println!("b = {:?}", &b);
    let ans = b.clone();
    let x = a.solve_triangular(uplo, Diag::NonUnit, b).unwrap();
    let b_ = a.dot(&x);
    assert_close_l2!(&b_, &ans, tol);
}
