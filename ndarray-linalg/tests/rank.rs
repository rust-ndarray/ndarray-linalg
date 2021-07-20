use ndarray::*;
use ndarray_linalg::*;

#[test]
fn rank_test_zero_3x3() {
    #[rustfmt::skip]
    let a: Array2<f64> = arr2(&[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ],
    );
    assert_eq!(0, a.rank().unwrap());
}

#[test]
fn rank_test_partial_3x3() {
    #[rustfmt::skip]
    let a: Array2<f64> = arr2(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ],
    );
    assert_eq!(2, a.rank().unwrap());
}

#[test]
fn rank_test_full_3x3() {
    #[rustfmt::skip]
    let a: Array2<f64> = arr2(&[
            [1., 0., 2.],
            [2., 1., 0.],
            [3., 2., 1.],
        ],
    );
    assert_eq!(3, a.rank().unwrap());
}
