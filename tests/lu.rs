
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
fn permutate() {
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    println!("a= \n{:?}", &a);
    let p = vec![2, 2, 3]; // replace 1-2
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa, arr2(&[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]))
}

#[test]
fn permutate_t() {
    let a = arr2(&[[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = vec![2, 2, 3]; // replace 1-2
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa, arr2(&[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]))
}

#[test]
fn permutate_3x4_t() {
    let a = arr2(&[[1., 5., 9.], [2., 6., 10.], [3., 7., 11.], [4., 8., 12.]]).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = vec![1, 3, 3]; // replace 2-3
    println!("permutation = \n{:?}", &p);
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa,
              arr2(&[[1., 2., 3., 4.], [9., 10., 11., 12.], [5., 6., 7., 8.]]));
}

#[test]
fn permutate_4x3_t() {
    let a = arr2(&[[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]]).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = vec![4, 2, 3, 4]; // replace 1-4
    println!("permutation = \n{:?}", &p);
    let pa = a.permutate_column(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa,
              arr2(&[[10., 11., 12.], [4., 5., 6.], [7., 8., 9.], [1., 2., 3.]]))
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
    all_close(l.dot(&u).permutate_column(&p), a);
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
    all_close(l.dot(&u).permutate_column(&p), a);
}

#[test]
fn lu_square_lower() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close(l.dot(&u).permutate_column(&p), a);
}

#[test]
fn lu_square_lower_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close(l.dot(&u).permutate_column(&p), a);
}

#[test]
fn lu_square() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close(l.dot(&u).permutate_column(&p), a);
}

#[test]
fn lu_square_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    all_close(l.dot(&u).permutate_column(&p), a);
}
