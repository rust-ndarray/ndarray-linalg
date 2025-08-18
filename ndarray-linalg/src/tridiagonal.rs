//! Vectors as a Tridiagonal matrix
//! &
//! Methods for tridiagonal matrices

use super::convert::*;
use super::error::*;
use super::layout::*;
use cauchy::Scalar;
use lax::*;
use ndarray::*;
use num_traits::One;

pub use lax::{LUFactorizedTridiagonal, Tridiagonal};

/// An interface for making a Tridiagonal struct.
pub trait ExtractTridiagonal<A: Scalar> {
    /// Extract tridiagonal elements and layout of the raw matrix.
    ///
    /// If the raw matrix has some non-tridiagonal elements,
    /// they will be ignored.
    ///
    /// The shape of raw matrix should be equal to or larger than (2, 2).
    fn extract_tridiagonal(&self) -> Result<Tridiagonal<A>>;
}

impl<A, S> ExtractTridiagonal<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn extract_tridiagonal(&self) -> Result<Tridiagonal<A>> {
        let l = self.square_layout()?;
        let (n, _) = l.size();
        if n < 2 {
            return Err(LinalgError::NotStandardShape {
                obj: "Tridiagonal",
                rows: 1,
                cols: 1,
            });
        }

        let dl = self.slice(s![1..n, 0..n - 1]).diag().to_vec();
        let d = self.diag().to_vec();
        let du = self.slice(s![0..n - 1, 1..n]).diag().to_vec();
        Ok(Tridiagonal { l, dl, d, du })
    }
}

pub trait SolveTridiagonal<A: Scalar, D: Dimension> {
    /// Solves a system of linear equations `A * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_tridiagonal<S: Data<Elem = A>>(&self, b: &ArrayBase<S, D>) -> Result<Array<A, D>>;
    /// Solves a system of linear equations `A * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_tridiagonal_into<S: DataMut<Elem = A>>(
        &self,
        b: ArrayBase<S, D>,
    ) -> Result<ArrayBase<S, D>>;
    /// Solves a system of linear equations `A^T * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_t_tridiagonal<S: Data<Elem = A>>(&self, b: &ArrayBase<S, D>) -> Result<Array<A, D>>;
    /// Solves a system of linear equations `A^T * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_t_tridiagonal_into<S: DataMut<Elem = A>>(
        &self,
        b: ArrayBase<S, D>,
    ) -> Result<ArrayBase<S, D>>;
    /// Solves a system of linear equations `A^H * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_h_tridiagonal<S: Data<Elem = A>>(&self, b: &ArrayBase<S, D>) -> Result<Array<A, D>>;
    /// Solves a system of linear equations `A^H * x = b` with tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result.
    fn solve_h_tridiagonal_into<S: DataMut<Elem = A>>(
        &self,
        b: ArrayBase<S, D>,
    ) -> Result<ArrayBase<S, D>>;
}

pub trait SolveTridiagonalInplace<A: Scalar, D: Dimension> {
    /// Solves a system of linear equations `A * x = b` tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result. The value of `x` is also assigned to the
    /// argument.
    fn solve_tridiagonal_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, D>,
    ) -> Result<&'a mut ArrayBase<S, D>>;
    /// Solves a system of linear equations `A^T * x = b` tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result. The value of `x` is also assigned to the
    /// argument.
    fn solve_t_tridiagonal_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, D>,
    ) -> Result<&'a mut ArrayBase<S, D>>;
    /// Solves a system of linear equations `A^H * x = b` tridiagonal
    /// matrix `A`, where `A` is `self`, `b` is the argument, and
    /// `x` is the successful result. The value of `x` is also assigned to the
    /// argument.
    fn solve_h_tridiagonal_inplace<'a, S: DataMut<Elem = A>>(
        &self,
        b: &'a mut ArrayBase<S, D>,
    ) -> Result<&'a mut ArrayBase<S, D>>;
}

macro_rules! impl_traits { ($dim: ident, $layout: ident) => {
    impl<A> SolveTridiagonal<A, $dim> for LUFactorizedTridiagonal<A>
    where
        A: Scalar + Lapack,
    {
        fn solve_tridiagonal<S: Data<Elem = A>>(
            &self,
            b: &ArrayBase<S, $dim>
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_tridiagonal_into<S: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<S, $dim>,
        ) -> Result<ArrayBase<S, $dim>> {
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal<S: Data<Elem = A>>(
            &self,
            b: &ArrayBase<S, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal_into<S: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<S, $dim>,
        ) -> Result<ArrayBase<S, $dim>> {
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal<S: Data<Elem = A>>(
            &self,
            b: &ArrayBase<S, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal_into<S: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<S, $dim>,
        ) -> Result<ArrayBase<S, $dim>> {
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
    }

    impl<A> SolveTridiagonal<A, $dim> for Tridiagonal<A>
    where
        A: Scalar + Lapack,
    {
        fn solve_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
    }

    impl<A, S> SolveTridiagonal<A, $dim> for ArrayBase<S, Ix2>
    where
        A: Scalar + Lapack,
        S: Data<Elem = A>,
    {
        fn solve_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_t_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_t_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal<Sb: Data<Elem = A>>(
            &self,
            b: &ArrayBase<Sb, $dim>,
        ) -> Result<Array<A, $dim>> {
            let mut b = replicate(b);
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
        fn solve_h_tridiagonal_into<Sb: DataMut<Elem = A>>(
            &self,
            mut b: ArrayBase<Sb, $dim>,
        ) -> Result<ArrayBase<Sb, $dim>> {
            self.solve_h_tridiagonal_inplace(&mut b)?;
            Ok(b)
        }
    }

    impl<A> SolveTridiagonalInplace<A, $dim> for LUFactorizedTridiagonal<A>
    where
        A: Scalar + Lapack,
    {
        fn solve_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            A::solve_tridiagonal(
                self,
                $layout!(rhs),
                Transpose::No,
                rhs.as_slice_mut().unwrap(),
            )?;
            Ok(rhs)
        }
        fn solve_t_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            A::solve_tridiagonal(
                self,
                $layout!(rhs),
                Transpose::Transpose,
                rhs.as_slice_mut().unwrap(),
            )?;
            Ok(rhs)
        }
        fn solve_h_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            A::solve_tridiagonal(
                self,
                $layout!(rhs),
                Transpose::Hermite,
                rhs.as_slice_mut().unwrap(),
            )?;
            Ok(rhs)
        }
    }

    impl<A> SolveTridiagonalInplace<A, $dim> for Tridiagonal<A>
    where
        A: Scalar + Lapack,
    {
        fn solve_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_tridiagonal_inplace(rhs)
        }
        fn solve_t_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_t_tridiagonal_inplace(rhs)
        }
        fn solve_h_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_h_tridiagonal_inplace(rhs)
        }
    }

    impl<A, S> SolveTridiagonalInplace<A, $dim> for ArrayBase<S, Ix2>
    where
        A: Scalar + Lapack,
        S: Data<Elem = A>,
    {
        fn solve_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_tridiagonal_inplace(rhs)
        }
        fn solve_t_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_t_tridiagonal_inplace(rhs)
        }
        fn solve_h_tridiagonal_inplace<'a, Sb>(
            &self,
            rhs: &'a mut ArrayBase<Sb, $dim>,
        ) -> Result<&'a mut ArrayBase<Sb, $dim>>
        where
            Sb: DataMut<Elem = A>,
        {
            let f = self.factorize_tridiagonal()?;
            f.solve_h_tridiagonal_inplace(rhs)
        }
    }
}}

macro_rules! layoutIx1 { ($rhs: ident) => {
    MatrixLayout::C { row: $rhs.dim() as i32, lda: 1 }
}}
impl_traits!(Ix1, layoutIx1);

macro_rules! layoutIx2 { ($rhs: ident) => { $rhs.layout()? }}
impl_traits!(Ix2, layoutIx2);

/// An interface for computing LU factorizations of tridiagonal matrix refs.
pub trait FactorizeTridiagonal<A: Scalar> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize_tridiagonal(&self) -> Result<LUFactorizedTridiagonal<A>>;
}

/// An interface for computing LU factorizations of tridiagonal matrices.
pub trait FactorizeTridiagonalInto<A: Scalar> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize_tridiagonal_into(self) -> Result<LUFactorizedTridiagonal<A>>;
}

impl<A> FactorizeTridiagonalInto<A> for Tridiagonal<A>
where
    A: Scalar + Lapack,
{
    fn factorize_tridiagonal_into(self) -> Result<LUFactorizedTridiagonal<A>> {
        Ok(A::lu_tridiagonal(self)?)
    }
}

impl<A> FactorizeTridiagonal<A> for Tridiagonal<A>
where
    A: Scalar + Lapack,
{
    fn factorize_tridiagonal(&self) -> Result<LUFactorizedTridiagonal<A>> {
        let a = self.clone();
        Ok(A::lu_tridiagonal(a)?)
    }
}

impl<A, S> FactorizeTridiagonal<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn factorize_tridiagonal(&self) -> Result<LUFactorizedTridiagonal<A>> {
        let a = self.extract_tridiagonal()?;
        Ok(A::lu_tridiagonal(a)?)
    }
}

/// Calculates the recurrent relation,
/// f_k = a_k * f_{k-1} - c_{k-1} * b_{k-1} * f_{k-2}
/// where {a_1, a_2, ..., a_n} are diagonal elements,
/// {b_1, b_2, ..., b_{n-1}} are super-diagonal elements, and
/// {c_1, c_2, ..., c_{n-1}} are sub-diagonal elements of matrix.
///
/// f[n] is used to calculate the determinant.
/// (https://en.wikipedia.org/wiki/Tridiagonal_matrix#Determinant)
///
/// In the future, the vector `f` can be used to calculate the inverce matrix.
/// (https://en.wikipedia.org/wiki/Tridiagonal_matrix#Inversion)
fn rec_rel<A: Scalar>(tridiag: &Tridiagonal<A>) -> Vec<A> {
    let n = tridiag.d.len();
    let mut f = Vec::with_capacity(n + 1);
    f.push(One::one());
    f.push(tridiag.d[0]);
    for i in 1..n {
        f.push(tridiag.d[i] * f[i] - tridiag.dl[i - 1] * tridiag.du[i - 1] * f[i - 1]);
    }
    f
}

/// An interface for calculating determinants of tridiagonal matrix refs.
pub trait DeterminantTridiagonal<A: Scalar> {
    /// Computes the determinant of the matrix.
    /// Unlike `.det()` of Determinant trait, this method
    /// doesn't returns the natural logarithm of the determinant
    /// but the determinant itself.
    fn det_tridiagonal(&self) -> Result<A>;
}

impl<A> DeterminantTridiagonal<A> for Tridiagonal<A>
where
    A: Scalar,
{
    fn det_tridiagonal(&self) -> Result<A> {
        let n = self.d.len();
        Ok(rec_rel(self)[n])
    }
}

impl<A, S> DeterminantTridiagonal<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn det_tridiagonal(&self) -> Result<A> {
        let tridiag = self.extract_tridiagonal()?;
        let n = tridiag.d.len();
        Ok(rec_rel(&tridiag)[n])
    }
}

/// An interface for *estimating* the reciprocal condition number of tridiagonal matrix refs.
pub trait ReciprocalConditionNumTridiagonal<A: Scalar> {
    /// *Estimates* the reciprocal of the condition number of the tridiagonal matrix in
    /// 1-norm.
    ///
    /// This method uses the LAPACK `*gtcon` routines, which *estimate*
    /// `self.inv_tridiagonal().opnorm_one()` and then compute `rcond = 1. /
    /// (self.opnorm_one() * self.inv_tridiagonal().opnorm_one())`.
    ///
    /// * If `rcond` is near `0.`, the matrix is badly conditioned.
    /// * If `rcond` is near `1.`, the matrix is well conditioned.
    fn rcond_tridiagonal(&self) -> Result<A::Real>;
}

/// An interface for *estimating* the reciprocal condition number of tridiagonal matrices.
pub trait ReciprocalConditionNumTridiagonalInto<A: Scalar> {
    /// *Estimates* the reciprocal of the condition number of the tridiagonal matrix in
    /// 1-norm.
    ///
    /// This method uses the LAPACK `*gtcon` routines, which *estimate*
    /// `self.inv_tridiagonal().opnorm_one()` and then compute `rcond = 1. /
    /// (self.opnorm_one() * self.inv_tridiagonal().opnorm_one())`.
    ///
    /// * If `rcond` is near `0.`, the matrix is badly conditioned.
    /// * If `rcond` is near `1.`, the matrix is well conditioned.
    fn rcond_tridiagonal_into(self) -> Result<A::Real>;
}

impl<A> ReciprocalConditionNumTridiagonal<A> for LUFactorizedTridiagonal<A>
where
    A: Scalar + Lapack,
{
    fn rcond_tridiagonal(&self) -> Result<A::Real> {
        Ok(A::rcond_tridiagonal(self)?)
    }
}

impl<A> ReciprocalConditionNumTridiagonalInto<A> for LUFactorizedTridiagonal<A>
where
    A: Scalar + Lapack,
{
    fn rcond_tridiagonal_into(self) -> Result<A::Real> {
        self.rcond_tridiagonal()
    }
}

impl<A, S> ReciprocalConditionNumTridiagonal<A> for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    fn rcond_tridiagonal(&self) -> Result<A::Real> {
        self.factorize_tridiagonal()?.rcond_tridiagonal_into()
    }
}
