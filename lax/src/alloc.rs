use cauchy::*;
use std::mem::MaybeUninit;

/// Helper for getting pointer of slice
pub(crate) trait AsPtr: Sized {
    type Elem;
    fn as_ptr(vec: &[Self]) -> *const Self::Elem;
    fn as_mut_ptr(vec: &mut [Self]) -> *mut Self::Elem;
}

macro_rules! impl_as_ptr {
    ($target:ty, $elem:ty) => {
        impl AsPtr for $target {
            type Elem = $elem;
            fn as_ptr(vec: &[Self]) -> *const Self::Elem {
                vec.as_ptr() as *const _
            }
            fn as_mut_ptr(vec: &mut [Self]) -> *mut Self::Elem {
                vec.as_mut_ptr() as *mut _
            }
        }
    };
}
impl_as_ptr!(i32, i32);
impl_as_ptr!(f32, f32);
impl_as_ptr!(f64, f64);
impl_as_ptr!(c32, lapack_sys::__BindgenComplex<f32>);
impl_as_ptr!(c64, lapack_sys::__BindgenComplex<f64>);
impl_as_ptr!(MaybeUninit<i32>, i32);
impl_as_ptr!(MaybeUninit<f32>, f32);
impl_as_ptr!(MaybeUninit<f64>, f64);
impl_as_ptr!(MaybeUninit<c32>, lapack_sys::__BindgenComplex<f32>);
impl_as_ptr!(MaybeUninit<c64>, lapack_sys::__BindgenComplex<f64>);

pub(crate) trait VecAssumeInit {
    type Target;
    unsafe fn assume_init(self) -> Self::Target;
}

impl<T> VecAssumeInit for Vec<MaybeUninit<T>> {
    type Target = Vec<T>;
    unsafe fn assume_init(self) -> Self::Target {
        // FIXME use Vec::into_raw_parts instead after stablized
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_raw_parts
        let mut me = std::mem::ManuallyDrop::new(self);
        Vec::from_raw_parts(me.as_mut_ptr() as *mut T, me.len(), me.capacity())
    }
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
pub(crate) unsafe fn vec_uninit<T: Sized>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}
