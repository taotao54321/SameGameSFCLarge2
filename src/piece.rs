use std::num::NonZeroU8;

use crate::array::array_newtype;
use crate::bounded::impl_bounded_nonzero_uint;

/// 駒種。
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece(NonZeroU8);

impl_bounded_nonzero_uint!(Piece, u8, 1, 5);

array_newtype!(PieceArray, Piece);
