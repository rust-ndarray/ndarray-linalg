include!("header.rs");

macro_rules! impl_test {
    ($modname:ident, $clone:ident) => {
mod $modname {
    use super::random_hermite;
    use ndarray_linalg::prelude::*;
    #[test]
    fn cholesky() {
        let a = random_hermite(3);
        println!("a = \n{:?}", a);
        let c = a.$clone().cholesky().unwrap();
        println!("c = \n{:?}", c);
        println!("cc = \n{:?}", c.t().dot(&c));
        assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
    }
    #[test]
    fn cholesky_t() {
        let a = random_hermite(3);
        println!("a = \n{:?}", a);
        let c = a.$clone().cholesky().unwrap();
        println!("c = \n{:?}", c);
        println!("cc = \n{:?}", c.t().dot(&c));
        assert_close_l2!(&c.t().dot(&c), &a, 1e-7);
    }
}
}} // impl_test

impl_test!(owned, clone);
impl_test!(shared, to_shared);
