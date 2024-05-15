use anyhow::ensure;
use clap::Parser;
use log::info;

use samegame_sfc_large_2::*;

#[derive(Debug, Parser)]
struct Cli {
    /// 最終スコアがこの値を超えないとわかったノードを枝刈りする。
    #[arg(long, default_value_t = 0)]
    prune_score_max: Score,

    /// ビーム幅。
    #[arg(value_parser = parse_int::parse::<usize>)]
    beam_capacity: usize,

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
    let (board, valid, _rng_after) = param.gen_board();
    ensure!(valid, "盤面再生成判定に引っ掛かる");

    let mut solver = BeamSearch::new(cli.beam_capacity, cli.prune_score_max);
    if let Some((score, solution)) = solver.solve(board) {
        println!("{score}\t{solution}");
    } else {
        info!("NO SOLUTION");
    }

    Ok(())
}
