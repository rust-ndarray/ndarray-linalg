
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn qr_square() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    let (q, _) = a.clone().qr().unwrap();
    all_close(q.dot(&q.t()), Array::eye(3));
}

#[test]
fn qr_3x4() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist);
    let (q, _) = a.clone().qr().unwrap();
    all_close(q.dot(&q.t()), Array::eye(3));
}

#[test]
fn qr_4x3() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist);
    let (q, _) = a.clone().qr().unwrap();
    all_close(q.dot(&q.t()), Array::eye(3));
}

#[test]
fn qr__() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    let (q, r) = a.clone().qr().unwrap();
    println!("a = \n{:?}", a);
    println!("q = \n{:?}", q);
    println!("r = \n{:?}", r);
    panic!("Manual kill!!");
}
