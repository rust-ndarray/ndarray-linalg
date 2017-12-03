
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
#[cfg(feature = "lapack-src")]
extern crate lapack_src;

use ndarray::*;
use ndarray_linalg::*;

#[test]
fn eigen_vector_manual() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Upper).unwrap();
    assert_close_l2!(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7);
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn diagonalize() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Upper).unwrap();
    let s = vecs.t().dot(&a).dot(&vecs);
    for i in 0..3 {
        assert_rclose!(e[i], s[(i, i)], 1e-7);
    }
}

#[test]
fn ssqrt() {
    let a: Array2<f64> = random_hpd(3);
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Upper).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}

#[test]
fn ssqrt_t() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Upper).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}
