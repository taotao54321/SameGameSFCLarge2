use std::io::BufRead as _;

use anyhow::anyhow;
use clap::Parser;
use log::info;

use samegame_sfc_large_2::*;

/// 標準入力から盤面生成パラメータ集合を読み、それぞれについてビームサーチを行う。
#[derive(Debug, Parser)]
struct Cli {
    /// 最終スコアがこの値を超えないとわかったノードを枝刈りする。
    #[arg(long, default_value_t = 0)]
    prune_score_max: Score,

    /// ビーム幅。
    #[arg(value_parser = parse_int::parse::<usize>)]
    beam_capacity: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    let cli = Cli::parse();

    let mut solver = BeamSearch::new(cli.beam_capacity, cli.prune_score_max);

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let param: RandomBoardParam = line.parse()?;
        let (board, _) = param
            .gen_legal_board()
            .ok_or_else(|| anyhow!("illegal param: {param}"))?;

        let RandomBoardParam {
            rng_state,
            nmi_counter,
            nmi_timing,
            entropy,
        } = param;

        info!("Search: rng_state=0x{rng_state:04X} nmi_counter=0x{nmi_counter:02X} nmi_timing={nmi_timing} entropy={entropy}");

        if let Some((score, solution)) = solver.solve(board) {
            println!("0x{rng_state:04X}\t0x{nmi_counter:02X}\t{nmi_timing}\t{entropy}\t{score}\t{solution}");
            // 同点の解は全て列挙したいので -1 する。
            solver.chmax_prune_score_max(score.saturating_sub(1));
        }
    }

    Ok(())
}
