include!("header.rs");

macro_rules! impl_test{
    ($modname:ident, $clone:ident) => {
mod $modname {
    use super::random_hermite;
    use ndarray_linalg::prelude::*;
    #[test]
    fn ssqrt() {
        let a = random_hermite(3);
        let ar = a.$clone().ssqrt().unwrap();
        assert_close_l2!(&ar.clone().t(), &ar, 1e-7; "not symmetric");
        assert_close_l2!(&ar.dot(&ar), &a, 1e-7; "not sqrt");
    }
    #[test]
    fn ssqrt_t() {
        let a = random_hermite(3).reversed_axes();
        let ar = a.$clone().ssqrt().unwrap();
        assert_close_l2!(&ar.clone().t(), &ar, 1e-7; "not symmetric");
        assert_close_l2!(&ar.dot(&ar), &a, 1e-7; "not sqrt");
    }
}
}} // impl_test

impl_test!(owned, clone);
impl_test!(shared, to_shared);
