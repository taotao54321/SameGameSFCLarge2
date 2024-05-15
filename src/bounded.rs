//! 数値型の値域を制限した newtype を扱う。

/// 最大値を `$max` に制限した符号なし整数型を実装する (最小値は 0)。
///
/// `$ty` は外部でたとえば以下のように定義されていなければならない:
///
/// ```
/// #[repr(transparent)]
/// #[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// pub struct Bounded(u32);
/// ```
///
/// この場合、`Option<Bounded>` は `Bounded` よりサイズが大きくなる。
/// これを避けたい場合、最小値を非零として `impl_bounded_nonzero_uint!` を使うこと。
macro_rules! impl_bounded_uint {
    ($ty:ty, $ty_inner:ty, $max:expr) => {
        // $ty は内部値の型と同じサイズでなければならない。
        const _: () =
            ::std::assert!(::std::mem::size_of::<$ty>() == ::std::mem::size_of::<$ty_inner>());

        impl $ty {
            pub const NUM: usize = $max as usize + 1;

            pub const MIN_VALUE: $ty_inner = 0;
            pub const MAX_VALUE: $ty_inner = $max;

            pub const MIN: Self = unsafe { Self::from_inner_unchecked(Self::MIN_VALUE) };
            pub const MAX: Self = unsafe { Self::from_inner_unchecked(Self::MAX_VALUE) };

            /// 内部値から値を作る。
            pub const fn from_inner(inner: $ty_inner) -> ::std::option::Option<Self> {
                if Self::inner_is_valid(inner) {
                    Some(unsafe { Self::from_inner_unchecked(inner) })
                } else {
                    None
                }
            }

            /// 内部値から値を作る。
            ///
            /// # Safety
            ///
            /// `inner` は有効値でなければならない。
            pub const unsafe fn from_inner_unchecked(inner: $ty_inner) -> Self {
                $crate::hint::assert_unchecked!(Self::inner_is_valid(inner));

                Self(inner)
            }

            const fn inner_is_valid(inner: $ty_inner) -> bool {
                ::std::matches!(inner, Self::MIN_VALUE..=Self::MAX_VALUE)
            }

            /// 0-based のインデックスから値を作る。
            pub const fn from_index(idx: usize) -> ::std::option::Option<Self> {
                if Self::index_is_valid(idx) {
                    Some(unsafe { Self::from_index_unchecked(idx) })
                } else {
                    None
                }
            }

            /// 0-based のインデックスから値を作る。
            ///
            /// # Safety
            ///
            /// `idx` は有効なインデックスでなければならない。
            pub const unsafe fn from_index_unchecked(idx: usize) -> Self {
                $crate::hint::assert_unchecked!(Self::index_is_valid(idx));

                Self::from_inner_unchecked(idx as $ty_inner)
            }

            const fn index_is_valid(idx: usize) -> bool {
                idx < Self::NUM
            }

            /// 内部値を返す。
            pub const fn to_inner(self) -> $ty_inner {
                self.0
            }

            /// 0-based のインデックスに変換する。
            pub const fn to_index(self) -> usize {
                self.to_inner() as usize
            }

            /// 全ての値を昇順で列挙する。
            pub fn all() -> impl ::std::iter::DoubleEndedIterator<Item = Self>
                   + ::std::iter::ExactSizeIterator
                   + ::std::iter::FusedIterator
                   + ::std::clone::Clone {
                (Self::MIN_VALUE..=Self::MAX_VALUE)
                    .map(|inner| unsafe { Self::from_inner_unchecked(inner) })
            }
        }
    };
}
pub(crate) use impl_bounded_uint;

/// 値域を `$min..=$max` に制限した非零符号なし整数型を実装する。
///
/// `$ty` は外部でたとえば以下のように定義されていなければならない:
///
/// ```
/// use std::num::NonZeroU32;
///
/// #[repr(transparent)]
/// #[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// pub struct Bounded(NonZeroU32);
/// ```
macro_rules! impl_bounded_nonzero_uint {
    ($ty:ty, $ty_inner:ty, $min:expr, $max:expr) => {
        // $ty は Option に入れたときサイズが変わってはならない。
        const _: () = ::std::assert!(
            ::std::mem::size_of::<::std::option::Option<$ty>>() == ::std::mem::size_of::<$ty>()
        );
        // $ty は内部値の型と同じサイズでなければならない。
        const _: () =
            ::std::assert!(::std::mem::size_of::<$ty>() == ::std::mem::size_of::<$ty_inner>());
        // 0 < $min <= $max でなければならない。
        const _: () = ::std::assert!(0 < $min);
        const _: () = ::std::assert!($min <= $max);

        impl $ty {
            pub const NUM: usize = ($max - $min + 1) as usize;

            pub const MIN_VALUE: $ty_inner = $min;
            pub const MAX_VALUE: $ty_inner = $max;

            pub const MIN: Self = unsafe { Self::from_inner_unchecked(Self::MIN_VALUE) };
            pub const MAX: Self = unsafe { Self::from_inner_unchecked(Self::MAX_VALUE) };

            /// 内部値から値を作る。
            pub const fn from_inner(inner: $ty_inner) -> ::std::option::Option<Self> {
                if Self::inner_is_valid(inner) {
                    Some(unsafe { Self::from_inner_unchecked(inner) })
                } else {
                    None
                }
            }

            /// 内部値から値を作る。
            ///
            /// # Safety
            ///
            /// `inner` は有効値でなければならない。
            pub const unsafe fn from_inner_unchecked(inner: $ty_inner) -> Self {
                $crate::hint::assert_unchecked!(Self::inner_is_valid(inner));

                Self($crate::nonzero::NonZero::<$ty_inner>::new_unchecked(inner))
            }

            const fn inner_is_valid(inner: $ty_inner) -> bool {
                ::std::matches!(inner, Self::MIN_VALUE..=Self::MAX_VALUE)
            }

            /// 0-based のインデックスから値を作る。
            pub const fn from_index(idx: usize) -> ::std::option::Option<Self> {
                if Self::index_is_valid(idx) {
                    Some(unsafe { Self::from_index_unchecked(idx) })
                } else {
                    None
                }
            }

            /// 0-based のインデックスから値を作る。
            ///
            /// # Safety
            ///
            /// `idx` は有効なインデックスでなければならない。
            pub const unsafe fn from_index_unchecked(idx: usize) -> Self {
                $crate::hint::assert_unchecked!(Self::index_is_valid(idx));

                Self::from_inner_unchecked((idx + $min as usize) as $ty_inner)
            }

            const fn index_is_valid(idx: usize) -> bool {
                idx < Self::NUM
            }

            /// 内部値を返す。
            pub const fn to_inner(self) -> $ty_inner {
                self.0.get()
            }

            /// 0-based のインデックスに変換する。
            pub const fn to_index(self) -> usize {
                (self.to_inner() - $min) as usize
            }

            /// 全ての値を昇順で列挙する。
            pub fn all() -> impl ::std::iter::DoubleEndedIterator<Item = Self>
                   + ::std::iter::ExactSizeIterator
                   + ::std::iter::FusedIterator
                   + ::std::clone::Clone {
                (Self::MIN_VALUE..=Self::MAX_VALUE)
                    .map(|inner| unsafe { Self::from_inner_unchecked(inner) })
            }
        }
    };
}
pub(crate) use impl_bounded_nonzero_uint;
