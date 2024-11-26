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
                vec.as_ptr().cast() as *const _
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
    type Elem;
    unsafe fn assume_init(self) -> Vec<Self::Elem>;

    /// An replacement of unstable API
    /// https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#method.slice_assume_init_ref
    unsafe fn slice_assume_init_ref(&self) -> &[Self::Elem];

    /// An replacement of unstable API
    /// https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#method.slice_assume_init_mut
    unsafe fn slice_assume_init_mut(&mut self) -> &mut [Self::Elem];
}

impl<T> VecAssumeInit for Vec<MaybeUninit<T>> {
    type Elem = T;
    unsafe fn assume_init(self) -> Vec<T> {
        // FIXME use Vec::into_raw_parts instead after stablized
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_raw_parts
        let mut me = std::mem::ManuallyDrop::new(self);
        Vec::from_raw_parts(me.as_mut_ptr() as *mut T, me.len(), me.capacity())
    }

    unsafe fn slice_assume_init_ref(&self) -> &[T] {
        std::slice::from_raw_parts(self.as_ptr().cast() as *const T, self.len())
    }

    unsafe fn slice_assume_init_mut(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut T, self.len())
    }
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
pub(crate) fn vec_uninit<T: Sized>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut v = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
    }
    v
}
