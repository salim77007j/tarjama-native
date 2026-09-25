use crate::Seg;

fn ts_srt(ms: i64) -> String {
    let ms = ms.max(0);
    let (s, rem) = (ms / 1000, ms % 1000);
    let (h, r) = (s / 3600, s % 3600);
    format!("{:02}:{:02}:{:02},{:03}", h, r / 60, r % 60, rem)
}

fn ts_vtt(ms: i64) -> String {
    let ms = ms.max(0);
    let (s, rem) = (ms / 1000, ms % 1000);
    let (h, r) = (s / 3600, s % 3600);
    format!("{:02}:{:02}:{:02}.{:03}", h, r / 60, r % 60, rem)
}

pub fn srt_ar(segs: &[Seg]) -> String {
    let mut out = String::new();
    for (i, s) in segs.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            ts_srt(s.t0),
            ts_srt(s.t1),
            s.ar
        ));
    }
    out
}

pub fn srt_bilingual(segs: &[Seg]) -> String {
    let mut out = String::new();
    for (i, s) in segs.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n{}\n\n",
            i + 1,
            ts_srt(s.t0),
            ts_srt(s.t1),
            s.ar,
            s.tr
        ));
    }
    out
}

pub fn vtt_ar(segs: &[Seg]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (i, s) in segs.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            ts_vtt(s.t0),
            ts_vtt(s.t1),
            s.ar
        ));
    }
    out
}

pub fn txt_ar(segs: &[Seg]) -> String {
    segs.iter()
        .map(|s| s.ar.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

pub fn txt_tr(segs: &[Seg]) -> String {
    segs.iter()
        .map(|s| s.tr.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}
