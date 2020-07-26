use ndarray_linalg::assert_rclose;

#[test]
fn assert() {
    assert_rclose!(1.0, 1.0, 1e-7);
}
