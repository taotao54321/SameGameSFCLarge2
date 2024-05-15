use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

pub(crate) type NonZero<T> = <T as NonZeroTrait>::NonZeroT;

pub(crate) trait NonZeroTrait {
    /// 元の型に対応する非零型。
    type NonZeroT;
}

impl NonZeroTrait for u8 {
    type NonZeroT = NonZeroU8;
}

impl NonZeroTrait for u16 {
    type NonZeroT = NonZeroU16;
}

impl NonZeroTrait for u32 {
    type NonZeroT = NonZeroU32;
}

impl NonZeroTrait for u64 {
    type NonZeroT = NonZeroU64;
}

impl NonZeroTrait for usize {
    type NonZeroT = NonZeroUsize;
}
