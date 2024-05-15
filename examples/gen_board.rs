use clap::Parser;
use itertools::Itertools as _;
use log::{info, warn};

use samegame_sfc_large_2::*;

/// 与えられたパラメータで盤面を生成する。
#[derive(Debug, Parser)]
struct Cli {
    #[arg(value_parser = parse_int::parse::<u16>)]
    rng_state: u16,

    #[arg(value_parser = parse_int::parse::<u8>)]
    nmi_counter: u8,

    nmi_timing: usize,

    entropy: GameEntropy,
}

fn main() -> anyhow::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    let cli = Cli::parse();

    let param = RandomBoardParam {
        rng_state: cli.rng_state,
        nmi_counter: cli.nmi_counter,
        nmi_timing: cli.nmi_timing,
        entropy: cli.entropy,
    };
    let (board, legal, rng_after) = param.gen_board();

    if !legal {
        warn!("再生成判定に引っ掛かる");
    }

    let pos = Position::new(board.clone());

    info!("RNG after: 0x{:04X}", rng_after.state());
    info!(
        "piece counts: [{}]",
        Piece::all().map(|piece| pos.piece_count(piece)).join(", ")
    );
    info!("gain upper bound: {}", pos.gain_upper_bound());

    print!("{board}");

    Ok(())
}
