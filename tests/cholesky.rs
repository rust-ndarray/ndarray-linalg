
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;

use rand::distributions::*;
use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, (Ix, Ix)>, b: Array<f64, (Ix, Ix)>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn cholesky() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    println!("a = \n{:?}", a);
    let c = a.clone().cholesky().unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    all_close(c.t().dot(&c), a);
}

#[test]
fn cholesky_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
    println!("a = \n{:?}", a);
    let c = a.clone().cholesky().unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    all_close(c.t().dot(&c), a);
}
