//! Cholesky decomposition

use ndarray::*;
use num_traits::Zero;

use super::error::*;
use super::layout::*;
use super::triangular::IntoTriangular;

use lapack_traits::LapackScalar;
pub use lapack_traits::UPLO;

pub trait Cholesky<K> {
    fn cholesky(self, UPLO) -> Result<K>;
}

impl<A, S> Cholesky<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
where
    A: LapackScalar + Zero,
    S: DataMut<Elem = A>,
{
    fn cholesky(mut self, uplo: UPLO) -> Result<ArrayBase<S, Ix2>> {
        A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok(self.into_triangular(uplo))
    }
}

impl<'a, A, S> Cholesky<&'a mut ArrayBase<S, Ix2>> for &'a mut ArrayBase<S, Ix2>
where
    A: LapackScalar + Zero,
    S: DataMut<Elem = A>,
{
    fn cholesky(mut self, uplo: UPLO) -> Result<&'a mut ArrayBase<S, Ix2>> {
        A::cholesky(self.square_layout()?, uplo, self.as_allocated_mut()?)?;
        Ok(self.into_triangular(uplo))
    }
}

impl<'a, A, Si, So> Cholesky<ArrayBase<So, Ix2>> for &'a ArrayBase<Si, Ix2>
where
    A: LapackScalar + Copy + Zero,
    Si: Data<Elem = A>,
    So: DataMut<Elem = A> + DataOwned,
{
    fn cholesky(self, uplo: UPLO) -> Result<ArrayBase<So, Ix2>> {
        let mut a = replicate(self);
        A::cholesky(a.square_layout()?, uplo, a.as_allocated_mut()?)?;
        Ok(a.into_triangular(uplo))
    }
}
