use crate::{error::*, layout::*, *};
use cauchy::*;

pub trait LapackStrict: Scalar {
    /// Allocate working memory for eigenvalue problem $A x = \lambda x$
    fn eigh_work(calc_eigenvec: bool, layout: MatrixLayout, uplo: UPLO) -> Result<EighWork<Self>>;

    /// Solve eigenvalue problem $A x = \lambda x$ using allocated working memory
    fn eigh_calc<'work>(
        work: &'work mut EighWork<Self>,
        a: &mut [Self],
    ) -> Result<&'work [Self::Real]>;

    /// Allocate working memory for generalized eigenvalue problem $Ax = \lambda Bx$
    fn eigh_generalized_work(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
    ) -> Result<EighGeneralizedWork<Self>>;

    /// Solve generalized eigenvalue problem $Ax = \lambda Bx$ using allocated working memory
    fn eigh_generalized_calc<'work>(
        work: &'work mut EighGeneralizedWork<Self>,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<&'work [Self::Real]>;
}

macro_rules! impl_lapack_strict_component {
    ($impl_trait:path; fn $name:ident $(<$lt:lifetime>)* ( $( $arg_name:ident : $arg_type:ty, )*) -> $result:ty ;) => {
        fn $name $(<$lt>)* ($($arg_name:$arg_type,)*) -> $result {
            <Self as $impl_trait>::$name($($arg_name),*)
        }
    };
}

macro_rules! impl_lapack_strict {
    ($scalar:ty) => {
        impl LapackStrict for $scalar {
            impl_lapack_strict_component!(
                Eigh;
                fn eigh_work(
                    calc_eigenvec: bool,
                    layout: MatrixLayout,
                    uplo: UPLO,
                ) -> Result<EighWork<Self>>;
            );
            impl_lapack_strict_component!(
                Eigh;
                fn eigh_calc<'work>(
                    work: &'work mut EighWork<Self>,
                    a: &mut [Self],
                ) -> Result<&'work [Self::Real]>;
            );

            impl_lapack_strict_component! (
                EighGeneralized;
                fn eigh_generalized_work(
                    calc_eigenvec: bool,
                    layout: MatrixLayout,
                    uplo: UPLO,
                ) -> Result<EighGeneralizedWork<Self>>;
            );

            impl_lapack_strict_component! (
                EighGeneralized;
                fn eigh_generalized_calc<'work>(
                    work: &'work mut EighGeneralizedWork<Self>,
                    a: &mut [Self],
                    b: &mut [Self],
                ) -> Result<&'work [Self::Real]>;
            );
        }
    };
}

impl_lapack_strict!(f32);
impl_lapack_strict!(f64);
impl_lapack_strict!(c32);
impl_lapack_strict!(c64);
