
extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
#[cfg(feature = "lapack-src")]
extern crate lapack_src;

#[test]
fn assert() {
    assert_rclose!(1.0, 1.0, 1e-7);
}
