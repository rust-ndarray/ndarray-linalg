
extern crate lapack;

use self::lapack::fortran::*;

pub trait LapackBinding: Sized {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32);
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32);
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self;
    fn _gesvd(jobu: u8,
              jobvt: u8,
              m: i32,
              n: i32,
              a: &mut [Self],
              lda: i32,
              s: &mut [Self],
              u: &mut [Self],
              ldu: i32,
              vt: &mut [Self],
              ldvt: i32,
              work: &mut [Self],
              lwork: i32,
              info: &mut i32);
}

impl LapackBinding for f64 {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32) {
        dsyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32) {
        dgetrf(m, n, a, lda, ipiv, info);
    }
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        dgetri(n, a, lda, ipiv, work, lwork, info);
    }
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self {
        dlange(norm, m, n, a, lda, work)
    }
    fn _gesvd(jobu: u8,
              jobvt: u8,
              m: i32,
              n: i32,
              a: &mut [Self],
              lda: i32,
              s: &mut [Self],
              u: &mut [Self],
              ldu: i32,
              vt: &mut [Self],
              ldvt: i32,
              work: &mut [Self],
              lwork: i32,
              info: &mut i32) {
        dgesvd(jobu,
               jobvt,
               m,
               n,
               a,
               lda,
               s,
               u,
               ldu,
               vt,
               ldvt,
               work,
               lwork,
               info);
    }
}

impl LapackBinding for f32 {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32) {
        ssyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32) {
        sgetrf(m, n, a, lda, ipiv, info);
    }
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        sgetri(n, a, lda, ipiv, work, lwork, info);
    }
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self {
        slange(norm, m, n, a, lda, work)
    }
    fn _gesvd(jobu: u8,
              jobvt: u8,
              m: i32,
              n: i32,
              a: &mut [Self],
              lda: i32,
              s: &mut [Self],
              u: &mut [Self],
              ldu: i32,
              vt: &mut [Self],
              ldvt: i32,
              work: &mut [Self],
              lwork: i32,
              info: &mut i32) {
        sgesvd(jobu,
               jobvt,
               m,
               n,
               a,
               lda,
               s,
               u,
               ldu,
               vt,
               ldvt,
               work,
               lwork,
               info);
    }
}
