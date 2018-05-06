extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;

#[test]
fn assert() {
    assert_rclose!(1.0, 1.0, 1e-7);
}
