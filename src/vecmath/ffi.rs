use intel_mkl_sys::*;

trait VecMath: Sized {
    /* Arthmetic */
    fn add(a: &[Self], b: &[Self], out: &mut [Self]);
    fn sub(a: &[Self], b: &[Self], out: &mut [Self]);
    fn mul(a: &[Self], b: &[Self], out: &mut [Self]);
    fn abs(in_: &[Self], out: &mut [Self]);

    /* Power and Root */
    fn div(a: &[Self], b: &[Self], out: &mut [Self]);
    fn sqrt(in_: &[Self], out: &mut [Self]);
    fn pow(a: &[Self], b: &[Self], out: &mut [Self]);
    fn powx(a: &[Self], b: Self, out: &mut [Self]);

    /* Exponential and Logarithmic */
    fn exp(in_: &[Self], out: &mut [Self]);
    fn ln(in_: &[Self], out: &mut [Self]);
    fn log10(in_: &[Self], out: &mut [Self]);

    /* Trigonometric */
    fn cos(in_: &[Self], out: &mut [Self]);
    fn sin(in_: &[Self], out: &mut [Self]);
    fn tan(in_: &[Self], out: &mut [Self]);
    fn acos(in_: &[Self], out: &mut [Self]);
    fn asin(in_: &[Self], out: &mut [Self]);
    fn atan(in_: &[Self], out: &mut [Self]);

    /* Hyperbolic */
    fn cosh(in_: &[Self], out: &mut [Self]);
    fn sinh(in_: &[Self], out: &mut [Self]);
    fn tanh(in_: &[Self], out: &mut [Self]);
    fn acosh(in_: &[Self], out: &mut [Self]);
    fn asinh(in_: &[Self], out: &mut [Self]);
    fn atanh(in_: &[Self], out: &mut [Self]);
}

trait VecMathReal: Sized {
    /* Arthmetic */
    fn sqr(in_: &[Self], out: &mut [Self]);
    fn linear_frac(in_: &[Self], out: &mut [Self]);
    fn fmod(in_: &[Self], out: &mut [Self]);
    fn remainder(in_: &[Self], out: &mut [Self]);

    /* Power and Root */
    fn inv(in_: &[Self], out: &mut [Self]);
    fn inv_sqrt(in_: &[Self], out: &mut [Self]);
    fn cbrt(in_: &[Self], out: &mut [Self]);
    fn inv_cbrt(in_: &[Self], out: &mut [Self]);
    fn pow2o3(in_: &[Self], out: &mut [Self]);
    fn pow3o2(in_: &[Self], out: &mut [Self]);
    fn powr(in_: &[Self], out: &mut [Self]);
    fn hypot(in_: &[Self], out: &mut [Self]);

    /* Exponential and Logarithmic */
    fn exp2(in_: &[Self], out: &mut [Self]);
    fn exp10(in_: &[Self], out: &mut [Self]);
    fn expm1(in_: &[Self], out: &mut [Self]);
    fn log2(in_: &[Self], out: &mut [Self]);
    fn log1p(in_: &[Self], out: &mut [Self]);
    fn logb(in_: &[Self], out: &mut [Self]);

    /* Trigonometric */
    fn sin_cos(in_: &[Self], out: &mut [Self]);
    fn atan2(in_: &[Self], out: &mut [Self]);

    /* Special */
    fn erf(in_: &[Self], out: &mut [Self]);
    fn erfc(in_: &[Self], out: &mut [Self]);
    fn cdf_normal(in_: &[Self], out: &mut [Self]);
    fn erf_inv(in_: &[Self], out: &mut [Self]);
    fn erfc_inv(in_: &[Self], out: &mut [Self]);
    fn cdf_normal_inv(in_: &[Self], out: &mut [Self]);
    fn ln_gamma(in_: &[Self], out: &mut [Self]);
    fn gamma(in_: &[Self], out: &mut [Self]);
    fn exp_integral(in_: &[Self], out: &mut [Self]);

    /* Rounding */
    fn floor(in_: &[Self], out: &mut [Self]);
    fn ceil(in_: &[Self], out: &mut [Self]);
    fn trunc(in_: &[Self], out: &mut [Self]);
    fn round(in_: &[Self], out: &mut [Self]);
    fn near_by_int(in_: &[Self], out: &mut [Self]);
    fn rint(in_: &[Self], out: &mut [Self]);
    fn modf(in_: &[Self], out: &mut [Self]);
    fn frac(in_: &[Self], out: &mut [Self]);

    /* Miscellaneous */
    fn copy_sign(in_: &[Self], out: &mut [Self]);
    fn next_after(in_: &[Self], out: &mut [Self]);
    fn fdim(in_: &[Self], out: &mut [Self]);
    fn fmax(in_: &[Self], out: &mut [Self]);
    fn fmin(in_: &[Self], out: &mut [Self]);
    fn maxmag(in_: &[Self], out: &mut [Self]);
    fn minmag(in_: &[Self], out: &mut [Self]);
}

trait VecMathComplex: Sized {
    /* Arthmetic */
    fn mulbyconj(in_: &[Self], out: &mut [Self]);
    fn conj(in_: &[Self], out: &mut [Self]);
    fn arg(in_: &[Self], out: &mut [Self]);

    /* Trigonometric */
    fn cis(in_: &[Self], out: &mut [Self]);
}

macro_rules! impl_unary {
    ($scalar:ty, $name:ident, $impl_name:ident) => {
        fn $name(in_: &[$scalar], out: &mut [$scalar]) {
            assert_eq!(in_.len(), out.len());
            let n = in_.len() as i32;
            unsafe { $impl_name(n, in_.as_ptr(), out.as_mut_ptr()) }
        }
    };
}

macro_rules! impl_binary {
    ($scalar:ty, $name:ident, $impl_name:ident) => {
        fn $name(a: &[$scalar], b: &[$scalar], out: &mut [$scalar]) {
            assert_eq!(a.len(), out.len());
            assert_eq!(b.len(), out.len());
            let n = out.len() as i32;
            unsafe { $impl_name(n, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) }
        }
    };
}

macro_rules! impl_binary_scalar {
    ($scalar:ty, $name:ident, $impl_name:ident) => {
        fn $name(a: &[$scalar], b: $scalar, out: &mut [$scalar]) {
            assert_eq!(a.len(), out.len());
            let n = out.len() as i32;
            unsafe { $impl_name(n, a.as_ptr(), b, out.as_mut_ptr()) }
        }
    };
}

impl VecMath for f32 {
    impl_binary!(f32, add, vsAdd);
    impl_binary!(f32, sub, vsSub);
    impl_binary!(f32, mul, vsMul);
    impl_unary!(f32, abs, vsAbs);

    impl_binary!(f32, div, vsDiv);
    impl_unary!(f32, sqrt, vsSqrt);
    impl_binary!(f32, pow, vsPow);
    impl_binary_scalar!(f32, powx, vsPowx);

    impl_unary!(f32, exp, vsExp);
    impl_unary!(f32, ln, vsLn);
    impl_unary!(f32, log10, vsLog10);

    impl_unary!(f32, cos, vsCos);
    impl_unary!(f32, sin, vsSin);
    impl_unary!(f32, tan, vsTan);
    impl_unary!(f32, acos, vsAcos);
    impl_unary!(f32, asin, vsAsin);
    impl_unary!(f32, atan, vsAtan);

    impl_unary!(f32, cosh, vsCosh);
    impl_unary!(f32, sinh, vsSinh);
    impl_unary!(f32, tanh, vsTanh);
    impl_unary!(f32, acosh, vsAcosh);
    impl_unary!(f32, asinh, vsAsinh);
    impl_unary!(f32, atanh, vsAtanh);
}

impl VecMath for f64 {
    impl_binary!(f64, add, vdAdd);
    impl_binary!(f64, sub, vdSub);
    impl_binary!(f64, mul, vdMul);
    impl_unary!(f64, abs, vdAbs);

    impl_binary!(f64, div, vdDiv);
    impl_unary!(f64, sqrt, vdSqrt);
    impl_binary!(f64, pow, vdPow);
    impl_binary_scalar!(f64, powx, vdPowx);

    impl_unary!(f64, exp, vdExp);
    impl_unary!(f64, ln, vdLn);
    impl_unary!(f64, log10, vdLog10);

    impl_unary!(f64, cos, vdCos);
    impl_unary!(f64, sin, vdSin);
    impl_unary!(f64, tan, vdTan);
    impl_unary!(f64, acos, vdAcos);
    impl_unary!(f64, asin, vdAsin);
    impl_unary!(f64, atan, vdAtan);

    impl_unary!(f64, cosh, vdCosh);
    impl_unary!(f64, sinh, vdSinh);
    impl_unary!(f64, tanh, vdTanh);
    impl_unary!(f64, acosh, vdAcosh);
    impl_unary!(f64, asinh, vdAsinh);
    impl_unary!(f64, atanh, vdAtanh);
}
