include!("header.rs");

macro_rules! impl_test_det {
    ($modname:ident, $clone:ident) => {
mod $modname {
    use super::random_hermite;
    use ndarray_linalg::prelude::*;
    use ndarray_numtest::prelude::*;
    #[test]
    fn deth() {
        let a = random_hermite(3);
        let (e, _) = a.$clone().eigh().unwrap();
        let deth = a.$clone().deth().unwrap();
        let det_eig = e.iter().fold(1.0, |x, y| x * y);
        deth.assert_close(det_eig, 1.0e-7);
    }
}
}} // impl_test_det

impl_test_det!(owned, clone);
impl_test_det!(shared, to_shared);
