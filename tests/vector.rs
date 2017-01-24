include!("header.rs");

#[test]
fn vector_norm() {
    let a = Array::range(1., 10., 1.);
    a.norm().assert_close(285.0.sqrt(), 1e-7);
}

#[test]
fn vector_norm_l1() {
    let a = arr1(&[1.0, -1.0]);
    a.norm_l1().assert_close(2.0, 1e-7);
    let b = arr2(&[[0.0, -1.0], [1.0, 0.0]]);
    b.norm_l1().assert_close(2.0, 1e-7);
}

#[test]
fn vector_norm_max() {
    let a = arr1(&[1.0, 1.0, -3.0]);
    a.norm_max().assert_close(3.0, 1e-7);
    let b = arr2(&[[1.0, 3.0], [1.0, -4.0]]);
    b.norm_max().assert_close(4.0, 1e-7);
}
