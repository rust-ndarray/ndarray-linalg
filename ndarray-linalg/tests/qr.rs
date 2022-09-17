use ndarray::*;
use ndarray_linalg::*;
use std::cmp::min;

fn test(a: &Array2<f64>, n: usize, m: usize) {
    let ans = a.clone();
    println!("a = \n{:?}", a);
    let (q, r): (Array2<_>, Array2<_>) = a.qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    assert_close_l2!(&q.t().dot(&q), &Array::eye(min(n, m)), 1e-7);
    assert_close_l2!(&q.dot(&r), &ans, 1e-7);
    assert_close_l2!(&r.clone().into_triangular(UPLO::Upper), &r, 1e-7);
}

fn test_square(a: &Array2<f64>, n: usize, m: usize) {
    let ans = a.clone();
    println!("a = \n{:?}", a);
    let (q, r): (Array2<_>, Array2<_>) = a.qr_square().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    assert_close_l2!(&q.t().dot(&q), &Array::eye(min(n, m)), 1e-7);
    assert_close_l2!(&q.dot(&r), &ans, 1e-7);
    assert_close_l2!(&r.clone().into_triangular(UPLO::Upper), &r, 1e-7);
}

#[test]
fn qr_sq() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 3), &mut rng);
    test_square(&a, 3, 3);
}

#[test]
fn qr_sq_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 3).f(), &mut rng);
    test_square(&a, 3, 3);
}

#[test]
fn qr_3x3() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 3), &mut rng);
    test(&a, 3, 3);
}

#[test]
fn qr_3x3_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 3).f(), &mut rng);
    test(&a, 3, 3);
}

#[test]
fn qr_3x4() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 4), &mut rng);
    test(&a, 3, 4);
}

#[test]
fn qr_3x4_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((3, 4).f(), &mut rng);
    test(&a, 3, 4);
}

#[test]
fn qr_4x3() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((4, 3), &mut rng);
    test(&a, 4, 3);
}

#[test]
fn qr_4x3_t() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a = random_using((4, 3).f(), &mut rng);
    test(&a, 4, 3);
}
