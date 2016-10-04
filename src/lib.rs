
extern crate ndarray;
extern crate lapack;

use ndarray::prelude::*;
use lapack::fortran::*;

pub trait Matrix: Sized {
    type Vector;
    // fn svd(self) -> (Self, Self::Vector, Self);
}

pub trait SquareMatrix: Matrix {
    // fn qr(self) -> (Self, Self);
    // fn lu(self) -> (Self, Self);
    // fn eig(self) -> (Self::Vector, Self);
    /// eigenvalue decomposition for Hermite matrix
    fn eigh(self) -> Option<(Self::Vector, Self)>;
}

impl<A> Matrix for Array<A, (Ix, Ix)> {
    type Vector = Array<A, Ix>;
}

pub trait Eigh: Sized {
    fn syev(i32, &mut [Self]) -> Option<Vec<Self>>;
}

impl Eigh for f64 {
    fn syev(n: i32, a: &mut [f64]) -> Option<Vec<f64>> {
        let mut w = vec![0.0; n as usize];
        let mut work = vec![0.0; 4 * n as usize];
        let mut info = 0;
        dsyev(b'V', b'U', n, a, n, &mut w, &mut work, 4 * n, &mut info);
        if info == 0 { Some(w) } else { None }
    }
}

impl Eigh for f32 {
    fn syev(n: i32, a: &mut [f32]) -> Option<Vec<f32>> {
        let mut w = vec![0.0; n as usize];
        let mut work = vec![0.0; 4 * n as usize];
        let mut info = 0;
        ssyev(b'V', b'U', n, a, n, &mut w, &mut work, 4 * n, &mut info);
        if info == 0 { Some(w) } else { None }
    }
}

fn fill_lower<A>(mut a: &mut Array<A, (Ix, Ix)>) {
    //
}

impl<A> SquareMatrix for Array<A, (Ix, Ix)>
    where A: Eigh
{
    fn eigh(self) -> Option<(Self::Vector, Self)> {
        let rows = self.rows();
        let cols = self.cols();
        if rows != cols {
            return None;
        }

        let mut a = self.into_raw_vec();
        let w = match Eigh::syev(rows as i32, &mut a) {
            Some(w) => w,
            None => return None,
        };

        let ea = Array::from_vec(w);
        let mut va = Array::from_vec(a).into_shape((rows, cols)).unwrap();
        fill_lower(&mut va);
        Some((ea, va))
    }
}
