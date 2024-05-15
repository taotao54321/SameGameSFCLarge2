/// 指定した型でインデックスアクセスできる配列を定義する。
///
/// `$ty_idx` は `bounded` モジュールを用いて定義された型であることを仮定している。
macro_rules! array_newtype {
    ($name:ident, $ty_idx:ty) => {
        #[repr(transparent)]
        #[derive(::std::clone::Clone, ::std::fmt::Debug, ::std::cmp::Eq, ::std::cmp::PartialEq)]
        pub struct $name<T>([T; <$ty_idx>::NUM]);

        impl<T: ::std::default::Default> ::std::default::Default for $name<T> {
            fn default() -> Self {
                Self::from_fn(|_| T::default())
            }
        }

        impl<T: ::std::clone::Clone> $name<T> {
            pub fn from_elem(elem: T) -> Self {
                Self::from_fn(|_| elem.clone())
            }
        }

        impl<T> $name<T> {
            pub const fn new(inner: [T; <$ty_idx>::NUM]) -> Self {
                Self(inner)
            }

            pub fn from_fn(mut f: impl ::std::ops::FnMut($ty_idx) -> T) -> Self {
                Self::new(::std::array::from_fn(|i| {
                    f(unsafe { <$ty_idx>::from_index_unchecked(i) })
                }))
            }

            pub const fn as_array(&self) -> &[T; <$ty_idx>::NUM] {
                &self.0
            }

            pub fn enumerate(
                &self,
            ) -> impl ::std::iter::DoubleEndedIterator<Item = ($ty_idx, &T)>
                   + ::std::iter::ExactSizeIterator
                   + ::std::iter::FusedIterator
                   + ::std::clone::Clone {
                <$ty_idx>::all().map(|x| (x, &self[x]))
            }
        }

        impl<T> ::std::ops::Index<$ty_idx> for $name<T> {
            type Output = T;

            fn index(&self, x: $ty_idx) -> &Self::Output {
                unsafe { self.0.get_unchecked(x.to_index()) }
            }
        }

        impl<T> ::std::ops::IndexMut<$ty_idx> for $name<T> {
            fn index_mut(&mut self, x: $ty_idx) -> &mut Self::Output {
                unsafe { self.0.get_unchecked_mut(x.to_index()) }
            }
        }
    };
}
pub(crate) use array_newtype;
