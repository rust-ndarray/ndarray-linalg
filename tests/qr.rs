include!("header.rs");

macro_rules! impl_test {
    ($funcname:ident, $random:path, $n:expr, $m:expr, $t:expr) => {
#[test]
fn $funcname() {
    use std::cmp::min;
    use ndarray::prelude::*;
    use ndarray_linalg::prelude::*;
    let a = $random($n, $m, $t);
    let ans = a.clone();
    println!("a = \n{:?}", &a);
    let (q, r) = a.qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    all_close_l2(&q.t().dot(&q), &Array::eye(min($n, $m)), 1e-7).unwrap();
    all_close_l2(&q.dot(&r), &ans, 1e-7).unwrap();
    all_close_l2(&drop_lower(r.clone()), &r, 1e-7).unwrap();
}
}} // impl_test

macro_rules! impl_test_qr {
    ($modname:ident, $random:path) => {
mod $modname {
    impl_test!(qr_square, $random, 3, 3, false);
    impl_test!(qr_square_t, $random, 3, 3, true);
    impl_test!(qr_3x4, $random, 3, 4, false);
    impl_test!(qr_3x4_t, $random, 3, 4, true);
    impl_test!(qr_4x3, $random, 4, 3, false);
    impl_test!(qr_4x3_t, $random, 4, 3, true);
}
}} // impl_test_qr

impl_test_qr!(owned, super::random_owned);
impl_test_qr!(shared, super::random_shared);
