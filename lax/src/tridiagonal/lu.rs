use crate::*;
use cauchy::*;
use num_traits::Zero;

/// Represents the LU factorization of a tridiagonal matrix `A` as `A = P*L*U`.
#[derive(Clone, PartialEq)]
pub struct LUFactorizedTridiagonal<A: Scalar> {
    /// A tridiagonal matrix which consists of
    /// - l : layout of raw matrix
    /// - dl: (n-1) multipliers that define the matrix L.
    /// - d : (n) diagonal elements of the upper triangular matrix U.
    /// - du: (n-1) elements of the first super-diagonal of U.
    pub a: Tridiagonal<A>,
    /// (n-2) elements of the second super-diagonal of U.
    pub du2: Vec<A>,
    /// The pivot indices that define the permutation matrix `P`.
    pub ipiv: Pivot,

    pub a_opnorm_one: A::Real,
}

impl<A: Scalar> Tridiagonal<A> {
    fn opnorm_one(&self) -> A::Real {
        let mut col_sum: Vec<A::Real> = self.d.iter().map(|val| val.abs()).collect();
        for i in 0..col_sum.len() {
            if i < self.dl.len() {
                col_sum[i] += self.dl[i].abs();
            }
            if i > 0 {
                col_sum[i] += self.du[i - 1].abs();
            }
        }
        let mut max = A::Real::zero();
        for &val in &col_sum {
            if max < val {
                max = val;
            }
        }
        max
    }
}

pub struct LuTridiagonalWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub du2: Vec<MaybeUninit<T>>,
    pub ipiv: Vec<MaybeUninit<i32>>,
}

pub trait LuTridiagonalWorkImpl {
    type Elem: Scalar;
    fn new(layout: MatrixLayout) -> Self;
    fn eval(self, a: Tridiagonal<Self::Elem>) -> Result<LUFactorizedTridiagonal<Self::Elem>>;
}

macro_rules! impl_lu_tridiagonal_work {
    ($s:ty, $trf:path) => {
        impl LuTridiagonalWorkImpl for LuTridiagonalWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Self {
                let (n, _) = layout.size();
                let du2 = vec_uninit((n - 2) as usize);
                let ipiv = vec_uninit(n as usize);
                LuTridiagonalWork { layout, du2, ipiv }
            }

            fn eval(
                mut self,
                mut a: Tridiagonal<Self::Elem>,
            ) -> Result<LUFactorizedTridiagonal<Self::Elem>> {
                let (n, _) = self.layout.size();
                // We have to calc one-norm before LU factorization
                let a_opnorm_one = a.opnorm_one();
                let mut info = 0;
                unsafe {
                    $trf(
                        &n,
                        AsPtr::as_mut_ptr(&mut a.dl),
                        AsPtr::as_mut_ptr(&mut a.d),
                        AsPtr::as_mut_ptr(&mut a.du),
                        AsPtr::as_mut_ptr(&mut self.du2),
                        AsPtr::as_mut_ptr(&mut self.ipiv),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(LUFactorizedTridiagonal {
                    a,
                    du2: unsafe { self.du2.assume_init() },
                    ipiv: unsafe { self.ipiv.assume_init() },
                    a_opnorm_one,
                })
            }
        }
    };
}

impl_lu_tridiagonal_work!(c64, lapack_sys::zgttrf_);
impl_lu_tridiagonal_work!(c32, lapack_sys::cgttrf_);
impl_lu_tridiagonal_work!(f64, lapack_sys::dgttrf_);
impl_lu_tridiagonal_work!(f32, lapack_sys::sgttrf_);
