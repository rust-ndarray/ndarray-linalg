use ndarray::*;
use ndarray_linalg::*;

#[should_panic]
#[test]
fn solveh_shape_mismatch() {
    let a: Array2<f64> = random_hpd(3);
    let b: Array1<f64> = random(2);
    let _ = a.solveh_into(b);
}

#[should_panic]
#[test]
fn factorizeh_solveh_shape_mismatch() {
    let a: Array2<f64> = random_hpd(3);
    let b: Array1<f64> = random(2);
    let f = a.factorizeh_into().unwrap();
    let _ = f.solveh_into(b);
}

#[test]
fn solveh_random() {
    let a: Array2<f64> = random_hpd(3);
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);

    let b = a.dot(&x);
    let f = a.factorizeh_into().unwrap();
    let y = f.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[should_panic]
#[test]
fn solveh_t_shape_mismatch() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let b: Array1<f64> = random(2);
    let _ = a.solveh_into(b);
}

#[should_panic]
#[test]
fn factorizeh_solveh_t_shape_mismatch() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let b: Array1<f64> = random(2);
    let f = a.factorizeh_into().unwrap();
    let _ = f.solveh_into(b);
}

#[test]
fn solveh_random_t() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);

    let b = a.dot(&x);
    let f = a.factorizeh_into().unwrap();
    let y = f.solveh_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}
