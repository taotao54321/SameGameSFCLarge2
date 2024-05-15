//! zobrist hash 用テーブルを生成する (`zobrist.rs` 内で `include!` する)。
//!
//! `once_cell` などを使うと若干オーバーヘッドが生じるので...。

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::Path;

use rand::{rngs::StdRng, Rng, SeedableRng as _};

use samegame_sfc_large_2::*;

fn main() -> anyhow::Result<()> {
    const PATH_BOARD: &str = "zobrist_board.in";

    let mut rng = StdRng::seed_from_u64(2024);

    make_table_board(PATH_BOARD, &mut rng)?;

    Ok(())
}

fn make_table_board(path: impl AsRef<Path>, rng: &mut impl Rng) -> anyhow::Result<()> {
    let mut wtr = create_file(path)?;

    write!(wtr, "PieceArray::new([")?;

    for _ in 0..Piece::NUM {
        write!(wtr, "SquareArray::new([")?;

        for _ in 0..Square::NUM {
            let key: u64 = rng.gen();
            write!(wtr, "0x{key:016x},")?;
        }

        write!(wtr, "]),")?;
    }

    write!(wtr, "])")?;

    Ok(())
}

fn create_file(path: impl AsRef<Path>) -> anyhow::Result<BufWriter<File>> {
    let wtr = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;

    Ok(BufWriter::new(wtr))
}
