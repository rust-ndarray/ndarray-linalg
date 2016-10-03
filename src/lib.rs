
extern crate num_traits;
extern crate ndarray;
extern crate lapack;

use ndarray::prelude::*;
use lapack::fortran::*;
use num_traits::Zero;

pub trait Matrix: Sized {
    type Vector;
    // fn svd(self) -> (Self, Self::Vector, Self);
}

pub trait SquareMatrix: Matrix {
    // fn qr(self) -> (Self, Self);
    // fn lu(self) -> (Self, Self);
    /// eigenvalue decomposition for Hermite matrix
    fn eigh(self) -> (Self::Vector, Self);
}

impl<A> Matrix for Array<A, (Ix, Ix)> {
    type Vector = Array<A, Ix>;
}

pub trait Eigh: Sized {
    fn syev(jobz: u8,
            uplo: u8,
            n: i32,
            a: &mut [Self],
            lda: i32,
            w: &mut [Self],
            work: &mut [Self],
            lwork: i32,
            info: &mut i32);
}

impl Eigh for f64 {
    fn syev(jobz: u8,
            uplo: u8,
            n: i32,
            a: &mut [f64],
            lda: i32,
            w: &mut [f64],
            work: &mut [f64],
            lwork: i32,
            info: &mut i32) {
        dsyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
}

impl Eigh for f32 {
    fn syev(jobz: u8,
            uplo: u8,
            n: i32,
            a: &mut [f32],
            lda: i32,
            w: &mut [f32],
            work: &mut [f32],
            lwork: i32,
            info: &mut i32) {
        ssyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
}

impl<A: Eigh + Clone + Zero> SquareMatrix for Array<A, (Ix, Ix)> {
    fn eigh(self) -> (Self::Vector, Self) {
        let rows = self.rows();
        let cols = self.cols();
        assert_eq!(rows, cols);

        let mut a = self.into_raw_vec();
        let n = rows as i32;
        let mut w: Vec<A> = vec![A::zero(); n as usize];
        let mut work: Vec<A> = vec![A::zero(); 4 * n as usize];
        let mut info = 0;
        Eigh::syev(b'V',
                   b'U',
                   n,
                   &mut a,
                   n,
                   &mut w,
                   &mut work,
                   4 * n,
                   &mut info);

        let ea = Array::from_vec(w);
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap();
        (ea, va)
    }
}
