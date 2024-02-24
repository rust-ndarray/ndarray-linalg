//! Operator norm

use crate::*;
use cauchy::*;

pub struct OperatorNormWork<T: Scalar> {
    pub ty: NormType,
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T::Real>>,
}

pub trait OperatorNormWorkImpl {
    type Elem: Scalar;
    fn new(t: NormType, l: MatrixLayout) -> Self;
    fn calc(&mut self, a: &[Self::Elem]) -> <Self::Elem as Scalar>::Real;
}

macro_rules! impl_operator_norm {
    ($s:ty, $lange:path) => {
        impl OperatorNormWorkImpl for OperatorNormWork<$s> {
            type Elem = $s;

            fn new(ty: NormType, layout: MatrixLayout) -> Self {
                let m = layout.lda();
                let work = match (ty, layout) {
                    (NormType::Infinity, MatrixLayout::F { .. })
                    | (NormType::One, MatrixLayout::C { .. }) => vec_uninit(m as usize),
                    _ => Vec::new(),
                };
                OperatorNormWork { ty, layout, work }
            }

            fn calc(&mut self, a: &[Self::Elem]) -> <Self::Elem as Scalar>::Real {
                let m = self.layout.lda();
                let n = self.layout.len();
                let t = match self.layout {
                    MatrixLayout::F { .. } => self.ty,
                    MatrixLayout::C { .. } => self.ty.transpose(),
                };
                unsafe {
                    $lange(
                        t.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut self.work),
                    )
                }
            }
        }
    };
}
impl_operator_norm!(c64, lapack_sys::zlange_);
impl_operator_norm!(c32, lapack_sys::clange_);
impl_operator_norm!(f64, lapack_sys::dlange_);
impl_operator_norm!(f32, lapack_sys::slange_);
