include!("header.rs");

macro_rules! impl_test {
    ($funcname:ident, $random:path, $n:expr, $m:expr, $t:expr) => {
#[test]
fn $funcname() {
    use std::cmp::min;
    use ndarray::prelude::*;
    use ndarray_linalg::prelude::*;
    let a = $random($n, $m, $t);
    let answer = a.clone();
    println!("a = \n{}", &a);
    let (u, s, vt) = a.svd().unwrap();
    println!("u = \n{}", &u);
    println!("s = \n{}", &s);
    println!("v = \n{}", &vt);
    let mut sm = Array::zeros(($n, $m));
    for i in 0..min($n, $m) {
        sm[(i, i)] = s[i];
    }
    all_close_l2(&u.dot(&sm).dot(&vt), &answer, 1e-7).unwrap();
}
}} // impl_test

macro_rules! impl_test_svd {
    ($modname:ident, $random:path) => {
mod $modname {
    impl_test!(svd_square, $random, 3, 3, false);
    impl_test!(svd_square_t, $random, 3, 3, true);
    impl_test!(svd_4x3, $random, 4, 3, false);
    impl_test!(svd_4x3_t, $random, 4, 3, true);
    impl_test!(svd_3x4, $random, 3, 4, false);
    impl_test!(svd_3x4_t, $random, 3, 4, true);
}
}} // impl_test_svd

impl_test_svd!(owned, super::random_owned);
impl_test_svd!(shared, super::random_shared);
