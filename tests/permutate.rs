include!("header.rs");

macro_rules! impl_test {
    ($testname:ident, $permutate:expr, $input:expr, $answer:expr) => {
#[test]
fn $testname() {
    use ndarray_linalg::prelude::*;
    let a = $input;
    println!("a= \n{:?}", &a);
    let p = $permutate; // replace 1-2
    let pa = a.permutated(&p);
    println!("permutated = \n{:?}", &pa);
    assert_close_l2!(&pa, &$answer, 1e-7);
}
}} // impl_test

macro_rules! impl_test_permuate {
    ($modname:ident, $array:path) => {
mod $modname {
    use ndarray;
    impl_test!(permutate,
               vec![2, 2, 3],
               $array(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
               $array(&[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]));
    impl_test!(permutate_t,
               vec![2, 2, 3],
               $array(&[[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]).reversed_axes(),
               $array(&[[4., 5., 6.], [1., 2., 3.], [7., 8., 9.]]));
    impl_test!(permutate_3x4,
               vec![1, 3, 3],
               $array(&[[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]]),
               $array(&[[1., 4., 7., 10.], [3., 6., 9., 12.], [2., 5., 8., 11.]]));
    impl_test!(permutate_3x4_t,
               vec![1, 3, 3],
               $array(&[[1., 5., 9.], [2., 6., 10.], [3., 7., 11.], [4., 8., 12.]]).reversed_axes(),
               $array(&[[1., 2., 3., 4.], [9., 10., 11., 12.], [5., 6., 7., 8.]]));
    impl_test!(permutate_4x3,
               vec![4, 2, 3, 4],
               $array(&[[1., 5., 9.], [2., 6., 10.], [3., 7., 11.], [4., 8., 12.]]),
               $array(&[[4., 8., 12.], [2., 6., 10.], [3., 7., 11.], [1., 5., 9.]]));
    impl_test!(permutate_4x3_t,
               vec![4, 2, 3, 4],
               $array(&[[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]]).reversed_axes(),
               $array(&[[10., 11., 12.], [4., 5., 6.], [7., 8., 9.], [1., 2., 3.]]));
}
}} // impl_test_permuate

impl_test_permuate!(owned, ndarray::arr2);
impl_test_permuate!(shared, ndarray::rcarr2);
