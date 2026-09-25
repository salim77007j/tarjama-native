/// Glossary: lines of `turkish=arabic`, `#` comments allowed.
pub fn parse(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        if let Some((tr, ar)) = line.split_once('=') {
            let tr = tr.trim();
            let ar = ar.trim();
            if !tr.is_empty() && !ar.is_empty() {
                out.push((tr.to_string(), ar.to_string()));
            }
        }
    }
    out
}

pub fn apply(ar: &mut String, pairs: &[(String, String)]) {
    for (tr, repl) in pairs {
        if ar.contains(tr.as_str()) {
            *ar = ar.replace(tr.as_str(), repl);
        } else {
            let lower = ar.to_lowercase();
            if let Some(pos) = lower.find(&tr.to_lowercase()) {
                // replace preserving the original span
                let end = pos + tr.len().min(ar.len() - pos);
                ar.replace_range(pos..end, repl);
            }
        }
    }
}
