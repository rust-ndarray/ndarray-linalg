
extern crate ndarray;
pub mod lapack_binding;

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use lapack_binding::Eigh;

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

impl<A> SquareMatrix for Array<A, (Ix, Ix)>
    where A: Eigh + LinalgScalar
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
        let va = Array::from_vec(a).into_shape((rows, cols)).unwrap().reversed_axes();
        Some((ea, va))
    }
}
