//! zobrist hash 関連。

use crate::asset::asset_include;
use crate::piece::{Piece, PieceArray};
use crate::square::{Square, SquareArray};

type TableBoard = PieceArray<SquareArray<u64>>;

#[derive(Debug)]
pub struct ZobristTable;

impl ZobristTable {
    const BOARD: TableBoard = asset_include!("zobrist_board.in");

    /// `piece` が `sq` にあるときのハッシュ値を返す。
    pub fn board(&self, piece: Piece, sq: Square) -> u64 {
        Self::BOARD[piece][sq]
    }
}

pub const ZOBRIST_TABLE: ZobristTable = ZobristTable;
