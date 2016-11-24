
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close<D: Dimension>(a: Array<f64, D>, b: Array<f64, D>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

#[test]
fn solve_upper() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_upper(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close(a.dot(&x), b);
}

#[test]
fn solve_upper_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_upper(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close(a.dot(&x), b);
}
