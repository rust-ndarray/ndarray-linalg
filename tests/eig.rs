
extern crate rand;
extern crate num_complex;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;
use num_complex::Complex;

type c64 = Complex<f64>;

fn dot(a: Array<c64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    let (n, l) = a.size();
    let (_, m) = b.size();
    let mut c = Array::<c64, _>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            for k in 0..l {
                c[(i, j)] = c[(i, j)] + a[(i, k)] * b[(k, j)];
            }
        }
    }
    c
}

#[test]
fn eig_random() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (w, vr) = a.clone().eig().unwrap();
    println!("w = \n{:?}", &w);
    println!("vr = \n{:?}", &vr);
    let mut lm = Array::zeros((3, 3));
    for i in 0..3 {
        lm[(i, i)] = w[i];
    }
    println!("lm = \n{:?}", &lm);
    let mut lv = Array::<Complex<f64>, _>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                lv[(i, j)] = lv[(i, j)] + lm[(i, k)] * vr[(k, j)];
            }
        }
    }
    println!("lv = \n{:?}", &lv);
    panic!("Manual Kill");
}
