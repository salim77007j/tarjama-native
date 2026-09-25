mod asr;
mod audio;
mod cli;
mod glossary;
mod gui;
mod pipeline;
mod selftest;
mod srt;
mod translate;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--selftest" {
        return selftest::run();
    }
    if args.len() >= 2 && args[1] == "--cli" {
        return cli::run(&args[2..]);
    }
    gui::run()
}
