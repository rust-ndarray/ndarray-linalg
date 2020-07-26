use ndarray::*;
use ndarray_linalg::*;

#[test]
fn inv_random() {
    let a: Array2<f64> = random((3, 3));
    let ai: Array2<_> = (&a).inv().unwrap();
    let id = Array::eye(3);
    assert_close_l2!(&ai.dot(&a), &id, 1e-7);
}

#[test]
fn inv_random_t() {
    let a: Array2<f64> = random((3, 3).f());
    let ai: Array2<_> = (&a).inv().unwrap();
    let id = Array::eye(3);
    assert_close_l2!(&ai.dot(&a), &id, 1e-7);
}

#[test]
#[should_panic]
fn inv_error() {
    // do not have inverse
    let a = Array::<f64, _>::zeros(9).into_shape((3, 3)).unwrap();
    let a_inv = a.inv().unwrap();
    println!("{:?}", a_inv);
}

#[test]
fn inv_2x2() {
    // Related to issue #123 where this problem led to a wrongly computed inverse when using the
    // `openblas` backend.
    let a: Array2<f64> = array!([1.0, 2.0], [3.0, 4.0]);
    let a_inv = a.inv().unwrap();
    assert_close_l2!(&a_inv, &array!([-2.0, 1.0], [1.5, -0.5]), 1e-7);
}
