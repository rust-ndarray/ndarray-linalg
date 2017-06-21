
macro_rules! impl_test {
    ($funcname:ident, $a:expr, $op1:expr, $opi:expr, $opf:expr) => {
#[test]
fn $funcname() {
    let a = $a;
    println!("ONE = {:?}", a.opnorm_one());
    println!("INF = {:?}", a.opnorm_inf());
    println!("FRO = {:?}", a.opnorm_fro());
    assert_rclose!(a.opnorm_fro().unwrap(), $opf, 1e-7; "Frobenius norm");
    assert_rclose!(a.opnorm_one().unwrap(), $op1, 1e-7; "One norm");
    assert_rclose!(a.opnorm_inf().unwrap(), $opi, 1e-7; "Infinity norm");
}
}} // impl_test

macro_rules! impl_test_opnorm {
    ($modname:ident, $array:ty, $range:path) => {
mod $modname {
    use ndarray::*;
    use ndarray_linalg::*;
    use num_traits::Float;
    fn gen(i: usize, j: usize, rev: bool) -> $array {
        let n = (i * j + 1) as f64;
        if rev {
            $range(1., n, 1.).into_shape((j, i)).unwrap().reversed_axes()
        } else {
            $range(1., n, 1.).into_shape((i, j)).unwrap()
        }
    }
    impl_test!(opnorm_square, gen(3, 3, false), 18.0, 24.0, 285.0.sqrt());
    impl_test!(opnorm_square_t, gen(3, 3, true), 24.0, 18.0, 285.0.sqrt());
    impl_test!(opnorm_3x4, gen(3, 4, false), 24.0, 42.0, 650.0.sqrt());
    impl_test!(opnorm_4x3_t, gen(4, 3, true), 42.0, 24.0, 650.0.sqrt());
    impl_test!(opnorm_3x4_t, gen(3, 4, true), 33.0, 30.0, 650.0.sqrt());
    impl_test!(opnorm_4x3, gen(4, 3, false), 30.0, 33.0, 650.0.sqrt());
}
}} // impl_test_opnorm

impl_test_opnorm!(owned, Array<f64, Ix2>, Array::range);
impl_test_opnorm!(shared, RcArray<f64, Ix2>, RcArray::range);
