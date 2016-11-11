
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
fn qr_square_upper() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.clone(), Array::eye(3));
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(r, a);
}

#[test]
fn qr_square_upper_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.clone(), Array::eye(3));
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(r, a);
}

#[test]
fn qr_square() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_square_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_3x4_upper() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 4), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.clone(), Array::eye(3));
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_3x4_upper_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((4, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.clone(), Array::eye(3));
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_3x4() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_3x4_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.dot(&q.t()), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_4x3_upper() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((4, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.t().dot(&q), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_4x3_upper_t() {
    let r_dist = Range::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 4), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.t().dot(&q), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_4x3() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.t().dot(&q), Array::eye(3));
    all_close(q.dot(&r), a);
}

#[test]
fn qr_4x3_t() {
    let r_dist = Range::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close(q.t().dot(&q), Array::eye(3));
    all_close(q.dot(&r), a);
}
