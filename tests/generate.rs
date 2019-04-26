use ndarray::*;
use ndarray_linalg::*;

#[test]
fn random_regular_transpose() {
    let a: Array2<f32> = random_regular(3);
    assert!(a.is_standard_layout());
    let a: Array2<f32> = random_regular_t(3);
    assert!(!a.is_standard_layout());
}
