
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use rand::distributions::*;
use ndarray_rand::RandomExt;

fn all_close(a: Array<f64, Ix2>, b: Array<f64, Ix2>) {
    if !a.all_close(&b, 1.0e-7) {
        panic!("\nTwo matrices are not equal:\na = \n{:?}\nb = \n{:?}\n",
               a,
               b);
    }
}

macro_rules! test_permutate {
    ($testname:ident, $permutate:expr, $input:expr, $answer:expr) => {
#[test]
fn $testname() {
    let a = arr2($input);
    println!("a= \n{:?}", &a);
    let p = $permutate; // replace 1-2
    let pa = a.permutated(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa, arr2($answer))
}
}} // end test_permutate

macro_rules! test_permutate_t {
    ($testname:ident, $permutate:expr, $input:expr, $answer:expr) => {
#[test]
fn $testname() {
    let a = arr2($input).reversed_axes();
    println!("a= \n{:?}", &a);
    let p = $permutate; // replace 1-2
    let pa = a.permutated(&p);
    println!("permutated = \n{:?}", &pa);
    all_close(pa, arr2($answer))
}
}} // end test_permutate_t

test_permutate!(permutate,
                vec![2, 2, 3],
                &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                &[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]);
test_permutate_t!(permutate_t,
                  vec![2, 2, 3],
                  &[[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]],
                  &[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]);
test_permutate!(permutate_3x4,
                vec![1, 3, 3],
                &[[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]],
                &[[1., 4., 7., 10.], [3., 6., 9., 12.], [2., 5., 8., 11.]]);
test_permutate_t!(permutate_3x4_t,
                  vec![1, 3, 3],
                  &[[1., 5., 9.], [2., 6., 10.], [3., 7., 11.], [4., 8., 12.]],
                  &[[1., 2., 3., 4.], [9., 10., 11., 12.], [5., 6., 7., 8.]]);
test_permutate!(permutate_4x3,
                vec![4, 2, 3, 4],
                &[[1., 5., 9.], [2., 6., 10.], [3., 7., 11.], [4., 8., 12.]],
                &[[4., 8., 12.], [2., 6., 10.], [3., 7., 11.], [1., 5., 9.]]);
test_permutate_t!(permutate_4x3_t,
                  vec![4, 2, 3, 4],
                  &[[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]],
                  &[[10., 11., 12.], [4., 5., 6.], [7., 8., 9.], [1., 2., 3.]]);

fn test_lu(a: Array<f64, Ix2>) {
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close(l.dot(&u).permutated(&p), a);
}

macro_rules! test_lu_upper {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random(($n, $m), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
}} // end test_lu_upper
test_lu_upper!(lu_square_upper, lu_square_upper_t, 3, 3);
test_lu_upper!(lu_3x4_upper, lu_3x4_upper_t, 3, 4);
test_lu_upper!(lu_4x3_upper, lu_4x3_upper_t, 4, 3);

macro_rules! test_lu_lower {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random(($n, $m), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
}} // end test_lu_lower
test_lu_lower!(lu_square_lower, lu_square_lower_t, 3, 3);
test_lu_lower!(lu_3x4_lower, lu_3x4_lower_t, 3, 4);
test_lu_lower!(lu_4x3_lower, lu_4x3_lower_t, 4, 3);

macro_rules! test_lu {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random(($n, $m), r_dist);
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    test_lu(a);
}
}} // end test_lu
test_lu!(lu_square, lu_square_t, 3, 3);
test_lu!(lu_3x4, lu_3x4_t, 3, 4);
test_lu!(lu_4x3, lu_4x3_t, 4, 3);
