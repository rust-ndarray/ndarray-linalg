use crate::*;
use cauchy::*;
use num_traits::Zero;

pub struct RcondTridiagonalWork<T: Scalar> {
    pub work: Vec<MaybeUninit<T>>,
    pub iwork: Option<Vec<MaybeUninit<i32>>>,
}

pub trait RcondTridiagonalWorkImpl {
    type Elem: Scalar;
    fn new(layout: MatrixLayout) -> Self;
    fn calc(
        &mut self,
        lu: &LUFactorizedTridiagonal<Self::Elem>,
    ) -> Result<<Self::Elem as Scalar>::Real>;
}

macro_rules! impl_rcond_tridiagonal_work_c {
    ($c:ty, $gtcon:path) => {
        impl RcondTridiagonalWorkImpl for RcondTridiagonalWork<$c> {
            type Elem = $c;

            fn new(layout: MatrixLayout) -> Self {
                let (n, _) = layout.size();
                let work = vec_uninit(2 * n as usize);
                RcondTridiagonalWork { work, iwork: None }
            }

            fn calc(
                &mut self,
                lu: &LUFactorizedTridiagonal<Self::Elem>,
            ) -> Result<<Self::Elem as Scalar>::Real> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
                let mut rcond = <Self::Elem as Scalar>::Real::zero();
                let mut info = 0;
                unsafe {
                    $gtcon(
                        NormType::One.as_ptr().cast(),
                        &n,
                        AsPtr::as_ptr(&lu.a.dl),
                        AsPtr::as_ptr(&lu.a.d),
                        AsPtr::as_ptr(&lu.a.du),
                        AsPtr::as_ptr(&lu.du2),
                        ipiv.as_ptr().cast(),
                        &lu.a_opnorm_one,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut self.work),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(rcond)
            }
        }
    };
}

impl_rcond_tridiagonal_work_c!(c64, lapack_sys::zgtcon_);
impl_rcond_tridiagonal_work_c!(c32, lapack_sys::cgtcon_);

macro_rules! impl_rcond_tridiagonal_work_r {
    ($c:ty, $gtcon:path) => {
        impl RcondTridiagonalWorkImpl for RcondTridiagonalWork<$c> {
            type Elem = $c;

            fn new(layout: MatrixLayout) -> Self {
                let (n, _) = layout.size();
                let work = vec_uninit(2 * n as usize);
                let iwork = vec_uninit(n as usize);
                RcondTridiagonalWork {
                    work,
                    iwork: Some(iwork),
                }
            }

            fn calc(
                &mut self,
                lu: &LUFactorizedTridiagonal<Self::Elem>,
            ) -> Result<<Self::Elem as Scalar>::Real> {
                let (n, _) = lu.a.l.size();
                let mut rcond = <Self::Elem as Scalar>::Real::zero();
                let mut info = 0;
                unsafe {
                    $gtcon(
                        NormType::One.as_ptr().cast(),
                        &n,
                        AsPtr::as_ptr(&lu.a.dl),
                        AsPtr::as_ptr(&lu.a.d),
                        AsPtr::as_ptr(&lu.a.du),
                        AsPtr::as_ptr(&lu.du2),
                        AsPtr::as_ptr(&lu.ipiv),
                        &lu.a_opnorm_one,
                        &mut rcond,
                        AsPtr::as_mut_ptr(&mut self.work),
                        AsPtr::as_mut_ptr(self.iwork.as_mut().unwrap()),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(rcond)
            }
        }
    };
}

impl_rcond_tridiagonal_work_r!(f64, lapack_sys::dgtcon_);
impl_rcond_tridiagonal_work_r!(f32, lapack_sys::sgtcon_);
