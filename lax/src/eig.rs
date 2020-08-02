//! Eigenvalue decomposition for general matrices

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Wraps `*geev` for general matrices
pub trait Eig_: Scalar {
    /// Calculate Right eigenvalue
    fn eig(
        calc_v: bool,
        l: MatrixLayout,
        a: &mut [Self],
    ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)>;
}

macro_rules! impl_eig_complex {
    ($scalar:ty, $ev:path) => {
        impl Eig_ for $scalar {
            fn eig(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
            ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                // Because LAPACK assumes F-continious array, C-continious array should be taken Hermitian conjugate.
                // However, we utilize a fact that left eigenvector of A^H corresponds to the right eigenvector of A
                let (jobvl, jobvr) = if calc_v {
                    match l {
                        MatrixLayout::C { .. } => (b'V', b'N'),
                        MatrixLayout::F { .. } => (b'N', b'V'),
                    }
                } else {
                    (b'N', b'N')
                };
                let mut eigs = unsafe { vec_uninit(n as usize) };
                let mut rwork = unsafe { vec_uninit(2 * n as usize) };

                let mut vl = if jobvl == b'V' {
                    Some(unsafe { vec_uninit((n * n) as usize) })
                } else {
                    None
                };
                let mut vr = if jobvr == b'V' {
                    Some(unsafe { vec_uninit((n * n) as usize) })
                } else {
                    None
                };

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $ev(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut vl.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut vr.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut work_size,
                        -1,
                        &mut rwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actal ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $ev(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut vl.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut vr.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut work,
                        lwork as i32,
                        &mut rwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // Hermite conjugate
                if jobvl == b'V' {
                    for c in vl.as_mut().unwrap().iter_mut() {
                        c.im = -c.im
                    }
                }

                Ok((eigs, vr.or(vl).unwrap_or(Vec::new())))
            }
        }
    };
}

impl_eig_complex!(c64, lapack::zgeev);
impl_eig_complex!(c32, lapack::cgeev);

macro_rules! impl_eig_real {
    ($scalar:ty, $ev:path) => {
        impl Eig_ for $scalar {
            fn eig(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
            ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                // Because LAPACK assumes F-continious array, C-continious array should be taken Hermitian conjugate.
                // However, we utilize a fact that left eigenvector of A^H corresponds to the right eigenvector of A
                let (jobvl, jobvr) = if calc_v {
                    match l {
                        MatrixLayout::C { .. } => (b'V', b'N'),
                        MatrixLayout::F { .. } => (b'N', b'V'),
                    }
                } else {
                    (b'N', b'N')
                };
                let mut eig_re = unsafe { vec_uninit(n as usize) };
                let mut eig_im = unsafe { vec_uninit(n as usize) };

                let mut vl = if jobvl == b'V' {
                    Some(unsafe { vec_uninit((n * n) as usize) })
                } else {
                    None
                };
                let mut vr = if jobvr == b'V' {
                    Some(unsafe { vec_uninit((n * n) as usize) })
                } else {
                    None
                };

                // calc work size
                let mut info = 0;
                let mut work_size = [0.0];
                unsafe {
                    $ev(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut eig_re,
                        &mut eig_im,
                        vl.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        vr.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut work_size,
                        -1,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $ev(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut eig_re,
                        &mut eig_im,
                        vl.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        vr.as_mut().map(|v| v.as_mut_slice()).unwrap_or(&mut []),
                        n,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // reconstruct eigenvalues
                let eigs: Vec<Self::Complex> = eig_re
                    .iter()
                    .zip(eig_im.iter())
                    .map(|(&re, &im)| Self::complex(re, im))
                    .collect();

                if !calc_v {
                    return Ok((eigs, Vec::new()));
                }

                // Reconstruct eigenvectors into complex-array
                // --------------------------------------------
                //
                // From LAPACK API https://software.intel.com/en-us/node/469230
                //
                // - If the j-th eigenvalue is real,
                //   - v(j) = VR(:,j), the j-th column of VR.
                //
                // - If the j-th and (j+1)-st eigenvalues form a complex conjugate pair,
                //   - v(j)   = VR(:,j) + i*VR(:,j+1)
                //   - v(j+1) = VR(:,j) - i*VR(:,j+1).
                //
                // ```
                //  j ->         <----pair---->  <----pair---->
                // [ ... (real), (imag), (imag), (imag), (imag), ... ] : eigs
                //       ^       ^       ^       ^       ^
                //       false   false   true    false   true          : is_conjugate_pair
                // ```
                let n = n as usize;
                let v = vr.or(vl).unwrap();
                let mut eigvecs = unsafe { vec_uninit(n * n) };
                let mut is_conjugate_pair = false; // flag for check `j` is complex conjugate
                for j in 0..n {
                    if eig_im[j] == 0.0 {
                        // j-th eigenvalue is real
                        for i in 0..n {
                            eigvecs[i + j * n] = Self::complex(v[i + j * n], 0.0);
                        }
                    } else {
                        // j-th eigenvalue is complex
                        // complex conjugated pair can be `j-1` or `j+1`
                        if is_conjugate_pair {
                            let j_pair = j - 1;
                            assert!(j_pair < n);
                            for i in 0..n {
                                eigvecs[i + j * n] = Self::complex(v[i + j_pair * n], v[i + j * n]);
                            }
                        } else {
                            let j_pair = j + 1;
                            assert!(j_pair < n);
                            for i in 0..n {
                                eigvecs[i + j * n] =
                                    Self::complex(v[i + j * n], -v[i + j_pair * n]);
                            }
                        }
                        is_conjugate_pair = !is_conjugate_pair;
                    }
                }

                Ok((eigs, eigvecs))
            }
        }
    };
}

impl_eig_real!(f64, lapack::dgeev);
impl_eig_real!(f32, lapack::sgeev);
