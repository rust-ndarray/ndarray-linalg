
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate rand;
extern crate float_cmp;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;
use float_cmp::ApproxEqRatio;

fn approx_eq(val: f64, truth: f64, ratio: f64) {
    if !val.approx_eq_ratio(&truth, ratio) {
        panic!("Not almost equal! val={:?}, truth={:?}, ratio={:?}",
               val,
               truth,
               ratio);
    }
}

fn random_hermite(n: usize) -> Array<f64, Ix2> {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    a.dot(&a.t())
}

#[test]
fn deth() {
    let a = random_hermite(3);
    let (e, _) = a.clone().eigh().unwrap();
    let deth = a.clone().deth().unwrap();
    let det_eig = e.iter().fold(1.0, |x, y| x * y);
    approx_eq(deth, det_eig, 1.0e-7);
}
