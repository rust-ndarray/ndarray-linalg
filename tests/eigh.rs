include!("header.rs");

macro_rules! impl_test {
    ($modname:ident, $clone:ident) => {
mod $modname {
    use ndarray::prelude::*;
    use ndarray_linalg::prelude::*;
    #[test]
    fn eigen_vector_manual() {
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let (e, vecs) = a.$clone().eigh().unwrap();
        all_close_l2(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7).unwrap();
        for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
            let av = a.dot(&v);
            let ev = v.mapv(|x| e[i] * x);
            all_close_l2(&av, &ev, 1.0e-7).unwrap();
        }
    }
    #[test]
    fn diagonalize() {
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let (e, vecs) = a.$clone().eigh().unwrap();
        let s = vecs.t().dot(&a).dot(&vecs);
        for i in 0..3 {
            e[i].assert_close(s[(i, i)], 1e-7);
        }
    }
}
}} // impl_test

impl_test!(owned, clone);
impl_test!(shared, to_shared);
