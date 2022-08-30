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
    fn eigg(
        calc_v: bool,
        l: MatrixLayout,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)>;
}

macro_rules! impl_eig_complex {
    ($scalar:ty, $ev1:path, $ev2:path) => {
        impl Eig_ for $scalar {
            fn eig(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
            ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                // LAPACK assumes a column-major input. A row-major input can
                // be interpreted as the transpose of a column-major input. So,
                // for row-major inputs, we we want to solve the following,
                // given the column-major input `A`:
                //
                //   A^T V = V Λ ⟺ V^T A = Λ V^T ⟺ conj(V)^H A = Λ conj(V)^H
                //
                // So, in this case, the right eigenvectors are the conjugates
                // of the left eigenvectors computed with `A`, and the
                // eigenvalues are the eigenvalues computed with `A`.
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
                    $ev1(
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
                    $ev1(
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

            fn eigg(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
                mut b: &mut [Self],
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
                let mut alpha = unsafe { vec_uninit(n as usize) };
                let mut beta = unsafe { vec_uninit(n as usize) };
                let mut rwork = unsafe { vec_uninit(8 * n as usize) };

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
                    $ev2(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut alpha,
                        &mut beta,
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
                    $ev2(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut alpha,
                        &mut beta,
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
                // reconstruct eigenvalues
                let eigs: Vec<Self::Complex> = alpha
                    .iter()
                    .zip(beta.iter())
                    .map(|(&a, &b)| a / b)
                    .collect();

                Ok((eigs, vr.or(vl).unwrap_or(Vec::new())))
            }
        }
    };
}

impl_eig_complex!(c64, lapack::zgeev, lapack::zggev);
impl_eig_complex!(c32, lapack::cgeev, lapack::cggev);

macro_rules! impl_eig_real {
    ($scalar:ty, $ev1:path, $ev2:path) => {
        impl Eig_ for $scalar {
            fn eig(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
            ) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                // LAPACK assumes a column-major input. A row-major input can
                // be interpreted as the transpose of a column-major input. So,
                // for row-major inputs, we we want to solve the following,
                // given the column-major input `A`:
                //
                //   A^T V = V Λ ⟺ V^T A = Λ V^T ⟺ conj(V)^H A = Λ conj(V)^H
                //
                // So, in this case, the right eigenvectors are the conjugates
                // of the left eigenvectors computed with `A`, and the
                // eigenvalues are the eigenvalues computed with `A`.
                //
                // We could conjugate the eigenvalues instead of the
                // eigenvectors, but we have to reconstruct the eigenvectors
                // into new matrices anyway, and by not modifying the
                // eigenvalues, we preserve the nice ordering specified by
                // `sgeev`/`dgeev`.
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
                    $ev1(
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
                    $ev1(
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
                // In the C-layout case, we need the conjugates of the left
                // eigenvectors, so the signs should be reversed.

                let n = n as usize;
                let v = vr.or(vl).unwrap();
                let mut eigvecs = unsafe { vec_uninit(n * n) };
                let mut col = 0;
                while col < n {
                    if eig_im[col] == 0. {
                        // The corresponding eigenvalue is real.
                        for row in 0..n {
                            let re = v[row + col * n];
                            eigvecs[row + col * n] = Self::complex(re, 0.);
                        }
                        col += 1;
                    } else {
                        // This is a complex conjugate pair.
                        assert!(col + 1 < n);
                        for row in 0..n {
                            let re = v[row + col * n];
                            let mut im = v[row + (col + 1) * n];
                            if jobvl == b'V' {
                                im = -im;
                            }
                            eigvecs[row + col * n] = Self::complex(re, im);
                            eigvecs[row + (col + 1) * n] = Self::complex(re, -im);
                        }
                        col += 2;
                    }
                }

                Ok((eigs, eigvecs))
            }

            fn eigg(
                calc_v: bool,
                l: MatrixLayout,
                mut a: &mut [Self],
                mut b: &mut [Self],
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
                let mut alphar = unsafe { vec_uninit(n as usize) };
                let mut alphai = unsafe { vec_uninit(n as usize) };
                let mut beta = unsafe { vec_uninit(n as usize) };

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
                    $ev2(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut alphar,
                        &mut alphai,
                        &mut beta,
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
                    $ev2(
                        jobvl,
                        jobvr,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut alphar,
                        &mut alphai,
                        &mut beta,
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
                let alpha: Vec<Self::Complex> = alphar
                    .iter()
                    .zip(alphai.iter())
                    .map(|(&re, &im)| Self::complex(re, im))
                    .collect();

                let eigs: Vec<Self::Complex> = alpha
                    .iter()
                    .zip(beta.iter())
                    .map(|(&a, &b)| a / b)
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
                    if alphai[j] == 0.0 {
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

impl_eig_real!(f64, lapack::dgeev, lapack::dggev);
impl_eig_real!(f32, lapack::sgeev, lapack::sggev);
