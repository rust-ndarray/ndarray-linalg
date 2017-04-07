include!("header.rs");

macro_rules! impl_test{
    ($modname:ident, $clone:ident) => {
mod $modname {
    use super::random_square;
    use ndarray::prelude::*;
    use ndarray_linalg::prelude::*;
    #[test]
    fn inv_random() {
        let a = random_square(3);
        let ai = a.$clone().inv().unwrap();
        let id = Array::eye(3);
        all_close_l2(&ai.dot(&a), &id, 1e-7).unwrap();
    }

    #[test]
    fn inv_random_t() {
        let a = random_square(3).reversed_axes();
        let ai = a.$clone().inv().unwrap();
        let id = Array::eye(3);
        all_close_l2(&ai.dot(&a), &id, 1e-7).unwrap();
    }

    #[test]
    #[should_panic]
    fn inv_error() {
        // do not have inverse
        let a = Array::<f64, _>::zeros(9).into_shape((3, 3)).unwrap();
        let a_inv = a.$clone().inv().unwrap();
        println!("{:?}", a_inv);
    }
}
}} // impl_test

impl_test!(owned, clone);
impl_test!(shared, to_shared);
