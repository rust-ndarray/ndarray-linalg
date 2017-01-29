include!("header.rs");

macro_rules! impl_test {
    ($funcname:ident, $random:path, $n:expr, $m:expr, $t:expr) => {
#[test]
fn $funcname() {
    use ndarray_linalg::prelude::*;
    let a = $random($n, $m, $t);
    let ans = a.clone();
    let (p, l, u) = a.lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close_l2(&l.dot(&u).permutated(&p), &ans, 1e-7).unwrap();
}
}} // impl_test

macro_rules! impl_test_lu {
    ($modname:ident, $random:path) => {
mod $modname {
    impl_test!(lu_square, $random, 3, 3, false);
    impl_test!(lu_square_t, $random, 3, 3, true);
    impl_test!(lu_3x4, $random, 3, 4, false);
    impl_test!(lu_3x4_t, $random, 3, 4, true);
    impl_test!(lu_4x3, $random, 4, 3, false);
    impl_test!(lu_4x3_t, $random, 4, 3, true);
}
}} // impl_test_lu

impl_test_lu!(owned, super::random_owned);
impl_test_lu!(shared, super::random_shared);
