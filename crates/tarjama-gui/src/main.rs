mod engine;
mod ui;

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Compatibility passthrough: forward --selftest / --cli to the engine
    // (safe build, works on every x86_64 CPU) and mirror its exit code.
    if args.len() >= 2 && (args[1] == "--selftest" || args[1] == "--cli") {
        let code = engine::forward_to_engine(&args[1..]);
        std::process::exit(code);
    }
    ui::run()
}
