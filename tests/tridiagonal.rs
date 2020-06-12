use ndarray::*;
use ndarray_linalg::*;

#[test]
fn to_tridiagonal() {
    let a: Array2<f64> = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let t = a.to_tridiagonal().unwrap();
    assert_close_l2!(&t.dl, &arr1(&[4.0, 8.0]), 1e-7);
    assert_close_l2!(&t.d, &arr1(&[1.0, 5.0, 9.0]), 1e-7);
    assert_close_l2!(&t.du, &arr1(&[2.0, 6.0]), 1e-7);
}

#[test]
fn solve_tridiagonal_f64() {
    // https://www.nag-j.co.jp/lapack/dgttrs.htm
    let a: Array2<f64> = arr2(&[
        [3.0, 2.1, 0.0, 0.0, 0.0],
        [3.4, 2.3, -1.0, 0.0, 0.0],
        [0.0, 3.6, -5.0, 1.9, 0.0],
        [0.0, 0.0, 7.0, -0.9, 8.0],
        [0.0, 0.0, 0.0, -6.0, 7.1],
    ]);
    let b: Array2<f64> = arr2(&[
        [2.7, 6.6],
        [-0.5, 10.8],
        [2.6, -3.2],
        [0.6, -11.2],
        [2.7, 19.1],
    ]);
    let x: Array2<f64> = arr2(&[
        [-4.0, 5.0],
        [7.0, -4.0],
        [3.0, -3.0],
        [-4.0, -2.0],
        [-3.0, 1.0],
    ]);
    let y = a.solve_tridiagonal_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}
//`*gttrf`, `*gtcon` and `*gttrs`
#[test]
fn solve_tridiagonal_c64() {
    // https://www.nag-j.co.jp/lapack/zgttrs.htm
    let a: Array2<c64> = arr2(&[
        [
            c64::new(-1.3, 1.3),
            c64::new(2.0, -1.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(1.0, -2.0),
            c64::new(-1.3, 1.3),
            c64::new(2.0, 1.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(1.0, 1.0),
            c64::new(-1.3, 3.3),
            c64::new(-1.0, 1.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(2.0, -3.0),
            c64::new(-0.3, 4.3),
            c64::new(1.0, -1.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 1.0),
            c64::new(-3.3, 1.3),
        ],
    ]);
    let b: Array2<c64> = arr2(&[
        [c64::new(2.4, -5.0), c64::new(2.7, 6.9)],
        [c64::new(3.4, 18.2), c64::new(-6.9, -5.3)],
        [c64::new(-14.7, 9.7), c64::new(-6.0, -0.6)],
        [c64::new(31.9, -7.7), c64::new(-3.9, 9.3)],
        [c64::new(-1.0, 1.6), c64::new(-3.0, 12.2)],
    ]);
    let x: Array2<c64> = arr2(&[
        [c64::new(1.0, 1.0), c64::new(2.0, -1.0)],
        [c64::new(3.0, -1.0), c64::new(1.0, 2.0)],
        [c64::new(4.0, 5.0), c64::new(-1.0, 1.0)],
        [c64::new(-1.0, -2.0), c64::new(2.0, 1.0)],
        [c64::new(1.0, -1.0), c64::new(2.0, -2.0)],
    ]);
    let y = a.solve_tridiagonal_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[test]
fn solve_tridiagonal_random() {
    let mut a: Array2<f64> = random((3, 3));
    a[[0, 2]] = 0.0;
    a[[2, 0]] = 0.0;
    let x: Array1<f64> = random(3);
    let b1 = a.dot(&x);
    let b2 = b1.clone();
    let y1 = a.solve_tridiagonal_into(b1).unwrap();
    let y2 = a.solve_into(b2).unwrap();
    assert_close_l2!(&x, &y1, 1e-7);
    assert_close_l2!(&y1, &y2, 1e-7);
}

#[test]
fn solve_tridiagonal_random_t() {
    let mut a: Array2<f64> = random((3, 3));
    a[[0, 2]] = 0.0;
    a[[2, 0]] = 0.0;
    let x: Array1<f64> = random(3);
    let at = a.t();
    let b1 = at.dot(&x);
    let b2 = b1.clone();
    let y1 = a.solve_t_tridiagonal_into(b1).unwrap();
    let y2 = a.solve_t_into(b2).unwrap();
    assert_close_l2!(&x, &y1, 1e-7);
    assert_close_l2!(&y1, &y2, 1e-7);
}

#[test]
fn det_tridiagonal_f64() {
    let a: Array2<f64> = arr2(&[[10.0, -9.0, 0.0], [7.0, -12.0, 11.0], [0.0, 10.0, 3.0]]);
    assert_aclose!(a.det_tridiagonal().unwrap(), -1271.0, 1e-7);
    assert_aclose!(a.det_tridiagonal().unwrap(), a.det().unwrap(), 1e-7);
}

#[test]
fn det_tridiagonal_random() {
    let mut a: Array2<f64> = random((3, 3));
    a[[0, 2]] = 0.0;
    a[[2, 0]] = 0.0;
    assert_aclose!(a.det_tridiagonal().unwrap(), a.det().unwrap(), 1e-7);
}

#[test]
fn rcond_tridiagonal_f64() {
    // https://www.nag-j.co.jp/lapack/dgtcon.htm
    let a: Array2<f64> = arr2(&[
        [3.0, 2.1, 0.0, 0.0, 0.0],
        [3.4, 2.3, -1.0, 0.0, 0.0],
        [0.0, 3.6, -5.0, 1.9, 0.0],
        [0.0, 0.0, 7.0, -0.9, 8.0],
        [0.0, 0.0, 0.0, -6.0, 7.1],
    ]);
    assert_aclose!(1.0 / a.rcond_tridiagonal().unwrap(), 9.27e1, 0.1);
    assert_aclose!(a.rcond_tridiagonal().unwrap(), a.rcond().unwrap(), 1e-3);
}

#[test]
fn rcond_tridiagonal_c64() {
    // https://www.nag-j.co.jp/lapack/dgtcon.htm
    let a: Array2<c64> = arr2(&[
        [
            c64::new(-1.3, 1.3),
            c64::new(2.0, -1.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(1.0, -2.0),
            c64::new(-1.3, 1.3),
            c64::new(2.0, 1.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(1.0, 1.0),
            c64::new(-1.3, 3.3),
            c64::new(-1.0, 1.0),
            c64::new(0.0, 0.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(2.0, -3.0),
            c64::new(-0.3, 4.3),
            c64::new(1.0, -1.0),
        ],
        [
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 1.0),
            c64::new(-3.3, 1.3),
        ],
    ]);
    assert_aclose!(1.0 / a.rcond_tridiagonal().unwrap(), 1.84e2, 1.0);
    assert_aclose!(a.rcond_tridiagonal().unwrap(), a.rcond().unwrap(), 1e-3);
}

#[test]
fn rcond_tridiagonal_identity() {
    macro_rules! rcond_identity {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let a = Array2::<$elem>::eye($rows);
            assert_aclose!(a.rcond_tridiagonal().unwrap(), 1., $atol);
        };
    }
    for rows in 2..6 {
        // cannot make 1x1 tridiagonal matrices.
        rcond_identity!(f64, rows, 1e-9);
        rcond_identity!(f32, rows, 1e-3);
        rcond_identity!(c64, rows, 1e-9);
        rcond_identity!(c32, rows, 1e-3);
    }
}
