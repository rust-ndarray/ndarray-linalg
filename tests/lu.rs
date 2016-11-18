
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
fn permutate_t() {
    let a = Array::<f64, _>::range(1., 10., 1.).into_shape((3, 3)).unwrap().reversed_axes();
    println!("a= \n{:?}", &a);
    let pa = a.permutate_column(&vec![2, 2, 3]);
    println!("permutated = \n{:?}", &pa);
    panic!("Manual KILL!!");
}

#[test]
fn permutate_3x4_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = vec![1, 3, 3];
    println!("permutation = \n{:?}", &p);
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    panic!("Manual KILL!!");
}

#[test]
fn permutate_4x3_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = vec![4, 2, 3, 4];
    println!("permutation = \n{:?}", &p);
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    panic!("Manual KILL!!");
}

#[test]
fn lu_square_upper() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    panic!("Manual KILL!!");
}

#[test]
fn lu_square_upper_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    panic!("Manual KILL!!");
}
