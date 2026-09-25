//! Native iced UI. The heavy pipeline runs in a separate engine process, so a
//! crash there surfaces as a readable error + automatic fallback instead of
//! killing the whole application.

use crate::engine;
use anyhow::Result;
use iced::widget::{button, column, container, horizontal_rule, progress_bar, radio, row, scrollable, text, text_editor, text_input};
use iced::{clipboard, font, Alignment, Color, Element, Font, Length, Subscription, Task, Theme};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tarjama_core::{srt, JobSpec, Kind, Seg};

const NOTO_SANS: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
const NOTO_SANS_ARABIC: &[u8] = include_bytes!("../assets/fonts/NotoSansArabic-Regular.ttf");
const AR_FONT: Font = Font::with_name("Noto Sans Arabic");

pub type EventQueue = Arc<Mutex<VecDeque<Prog>>>;

/// UI-side messages from the engine runner.
pub enum Prog {
    Engine(EngineEventAlias),
    Exited {
        variant: String,
        code: Option<i32>,
        stderr: String,
        saw_terminal: bool,
    },
}

type EngineEventAlias = tarjama_core::EngineEvent;

pub fn run() -> Result<()> {
    iced::application("Tarjama Studio Native", App::update, App::view)
        .theme(|_| Theme::Dark)
        .font(NOTO_SANS)
        .subscription(App::subscription)
        .window(iced::window::Settings {
            size: iced::Size::new(1060.0, 780.0),
            resizable: true,
            ..Default::default()
        })
        .run_with(|| {
            let mut app = App::new();
            // automation hook for headless testing
            if std::env::var("TARJAMA_AUTOTEST").as_deref() == Ok("1") {
                if let Ok(f) = std::env::var("TARJAMA_AUTOTEST_FILE") {
                    app.file = Some(PathBuf::from(f));
                    if let Ok(m) = std::env::var("TARJAMA_AUTOTEST_MODEL") {
                        if let Ok(k) = Kind::parse(&m) {
                            app.model = k;
                        }
                    }
                    app.autostart = true;
                    app.autotest = true;
                    app.log_line("autotest mode: file preset");
                }
            }
            (
                app,
                Task::batch(vec![font::load(NOTO_SANS_ARABIC).map(|_| Message::Noop)]),
            )
        })
        .map_err(|e| anyhow::anyhow!("iced error: {e:?}"))?;
    Ok(())
}

struct App {
    model: Kind,
    file: Option<PathBuf>,
    context: String,
    glossary: text_editor::Content,
    running: bool,
    cancel_requested: bool,
    child: Option<engine::SharedChild>,
    child_variant: Option<String>,
    retried_safe: bool,
    queue: EventQueue,
    base_stage: String,
    stage: String,
    stage_started: Option<Instant>,
    progress: f32,
    logs: String,
    segs: Vec<Seg>,
    err: Option<String>,
    exported: bool,
    autostart: bool,
    autotest: bool,
    autotest_at: Option<Instant>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
enum Message {
    Noop,
    ModelSel(Kind),
    PickFile,
    FilePicked(Option<PathBuf>),
    Context(String),
    Editor(text_editor::Action),
    Start,
    Cancel,
    Poll,
    ExportAll,
    OpenFolder,
    CopyAll,
    QuitNow,
}

impl App {
    fn new() -> Self {
        Self {
            model: Kind::Small,
            file: None,
            context: String::new(),
            glossary: text_editor::Content::new(),
            running: false,
            cancel_requested: false,
            child: None,
            child_variant: None,
            retried_safe: false,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            base_stage: String::from("Idle - choose a file to begin"),
            stage: String::from("Idle - choose a file to begin"),
            stage_started: None,
            progress: 0.0,
            logs: String::new(),
            segs: Vec::new(),
            err: None,
            exported: false,
            autostart: false,
            autotest: false,
            autotest_at: None,
        }
    }

    fn log_line(&mut self, s: impl Into<String>) {
        let s: String = s.into();
        println!("{s}");
        self.logs.push_str(&s);
        self.logs.push('\n');
    }

    fn set_stage(&mut self, base: String, timed: bool) {
        self.base_stage = base.clone();
        self.stage_started = if timed { Some(Instant::now()) } else { None };
        self.stage = base;
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::QuitNow => iced::exit(),
            Message::Noop => Task::none(),
            Message::ModelSel(k) => {
                if !self.running {
                    self.model = k;
                }
                Task::none()
            }
            Message::PickFile => {
                let fut = async {
                    let dialog = rfd::AsyncFileDialog::new()
                        .add_filter(
                            "Media (video/audio)",
                            &[
                                "mp4", "mkv", "webm", "mov", "avi", "m4v", "ts", "flv", "wmv",
                                "mpg", "mpeg", "3gp", "mp3", "wav", "m4a", "aac", "flac", "ogg",
                                "opus",
                            ],
                        )
                        .pick_file()
                        .await;
                    Message::FilePicked(dialog.map(|h| h.path().to_path_buf()))
                };
                Task::perform(fut, |m| m)
            }
            Message::FilePicked(f) => {
                match f {
                    Some(p) => {
                        self.log_line(format!("Selected: {}", p.display()));
                        self.file = Some(p);
                        self.err = None;
                    }
                    None => self.log_line("File selection cancelled"),
                }
                Task::none()
            }
            Message::Context(s) => {
                self.context = s;
                Task::none()
            }
            Message::Editor(a) => {
                self.glossary.perform(a);
                Task::none()
            }
            Message::Start => {
                if self.running {
                    return Task::none();
                }
                let Some(file) = self.file.clone() else {
                    self.err = Some("Choose a video or audio file first".into());
                    return Task::none();
                };
                match tarjama_core::models_dir() {
                    Ok(_) => {}
                    Err(e) => {
                        self.err = Some(format!("{e:#}"));
                        return Task::none();
                    }
                }
                self.err = None;
                self.segs.clear();
                self.exported = false;
                self.progress = 0.0;
                self.set_stage("Starting engine...".into(), false);
                self.running = true;
                self.cancel_requested = false;
                self.retried_safe = false;
                let variant = engine::preferred_variant().to_string();
                self.launch(&variant, file, None);
                Task::none()
            }
            Message::Cancel => {
                if self.running {
                    self.cancel_requested = true;
                    if let Some(c) = &self.child {
                        engine::kill(c);
                    }
                    self.log_line("Cancel requested - stopping engine...");
                }
                Task::none()
            }
            Message::Poll => {
                let msgs: Vec<Prog> = {
                    let mut g = self.queue.lock().unwrap();
                    g.drain(..).collect()
                };
                for m in msgs {
                    match m {
                        Prog::Engine(tarjama_core::EngineEvent::Log { s }) => self.log_line(s),
                        Prog::Engine(tarjama_core::EngineEvent::Stage { s }) => {
                            self.progress = 0.0;
                            match s.as_str() {
                                "extracting" => {
                                    self.set_stage("Extracting audio...".into(), true)
                                }
                                "loading-model" => self.set_stage(
                                    format!("Loading whisper model ({})...", self.model.as_str()),
                                    true,
                                ),
                                "transcribe" => {
                                    self.set_stage("Transcribing Turkish speech...".into(), true)
                                }
                                "loading-mt" => {
                                    self.set_stage("Loading translation model...".into(), true)
                                }
                                "translate" => {
                                    self.set_stage("Translating to Arabic...".into(), true)
                                }
                                other => self.set_stage(other.to_string(), false),
                            }
                        }
                        Prog::Engine(tarjama_core::EngineEvent::Heartbeat { s }) => {
                            self.stage = format!("{} ({}s)", self.base_stage, s);
                        }
                        Prog::Engine(tarjama_core::EngineEvent::Mt { a, b }) => {
                            self.progress = if b > 0 { a as f32 / b as f32 } else { 0.0 };
                            self.set_stage(format!("Translating to Arabic ({a}/{b})..."), true);
                        }
                        Prog::Engine(tarjama_core::EngineEvent::Done { segs }) => {
                            self.running = false;
                            self.progress = 1.0;
                            self.stage = format!("Done - {} subtitle lines ready", segs.len());
                            self.log_line(format!("DONE: {} lines produced", segs.len()));
                            self.segs = segs;
                            if self.autotest {
                                let _ = self.update(Message::ExportAll);
                                let marker = std::env::temp_dir().join("tarjama_autotest_ok");
                                let _ = std::fs::write(
                                    &marker,
                                    format!(
                                        "{} {} variant={}\n",
                                        self.segs.len(),
                                        self.model.as_str(),
                                        self.child_variant.as_deref().unwrap_or("?")
                                    ),
                                );
                                self.log_line("AUTOTEST COMPLETE");
                                self.autotest_at = Some(Instant::now());
                            }
                        }
                        Prog::Engine(tarjama_core::EngineEvent::Failed { err }) => {
                            self.running = false;
                            self.err = Some(err.clone());
                            self.set_stage("Failed".into(), false);
                            self.log_line(format!("ERROR: {err}"));
                            if self.autotest {
                                let marker = std::env::temp_dir().join("tarjama_autotest_ok");
                                let _ = std::fs::write(&marker, format!("FAILED: {err}\n"));
                            }
                        }
                        Prog::Exited { variant, code, stderr, saw_terminal } => {
                            self.child = None;
                            if !self.running {
                                continue; // already handled (Done/Failed/Cancel)
                            }
                            if self.cancel_requested {
                                self.running = false;
                                self.set_stage("Cancelled".into(), false);
                                self.log_line("Stopped by user");
                                continue;
                            }
                            if saw_terminal || code == Some(0) {
                                // engine reported its own outcome already
                                self.running = false;
                                continue;
                            }
                            // Native crash (no terminal event, non-zero exit).
                            if variant == "fast" && !self.retried_safe {
                                self.retried_safe = true;
                                self.log_line(format!(
                                    "The fast engine ({variant}) crashed (exit {:?}) - switching to the maximum-compatibility engine and retrying automatically...",
                                    code
                                ));
                                self.set_stage("Restarting with compatibility engine...".into(), false);
                                if let Some(f) = self.file.clone() {
                                    self.launch("safe", f, Some(0.5));
                                }
                            } else {
                                self.running = false;
                                let mut msg = format!(
                                    "The engine crashed (exit code {:?}, variant {variant}).",
                                    code
                                );
                                if !stderr.is_empty() {
                                    msg.push_str(&format!(" Details: {stderr}"));
                                }
                                msg.push_str(
                                    " Suggestions: try the Tiny model, use a shorter file, or close other memory-heavy programs.",
                                );
                                self.err = Some(msg);
                                self.set_stage("Failed".into(), false);
                                self.log_line("ERROR: engine crash");
                            }
                        }
                    }
                }
                // live elapsed timer while a timed stage runs
                if self.running {
                    if let Some(t) = self.stage_started {
                        let secs = t.elapsed().as_secs();
                        if secs >= 3 {
                            self.stage = format!("{} ({}s)", self.base_stage, secs);
                        }
                    }
                }
                if let Some(t) = self.autotest_at {
                    if t.elapsed() > std::time::Duration::from_secs(7) {
                        return iced::exit();
                    }
                }
                if self.autostart
                    && self.file.is_some()
                    && !self.running
                    && self.segs.is_empty()
                    && self.err.is_none()
                {
                    self.autostart = false;
                    return self.update(Message::Start);
                }
                Task::none()
            }
            Message::ExportAll => {
                if self.segs.is_empty() {
                    return Task::none();
                }
                let dir = self
                    .file
                    .as_ref()
                    .and_then(|f| f.parent().map(|d| d.to_path_buf()))
                    .unwrap_or_else(|| PathBuf::from("."));
                let stem = self
                    .file
                    .as_ref()
                    .and_then(|f| f.file_stem().map(|s| s.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "subtitles".into());
                let files: Vec<(String, String)> = vec![
                    (format!("{stem}.ar.srt"), srt::srt_ar(&self.segs)),
                    (format!("{stem}.bilingual.srt"), srt::srt_bilingual(&self.segs)),
                    (format!("{stem}.ar.vtt"), srt::vtt_ar(&self.segs)),
                    (format!("{stem}.ar.txt"), srt::txt_ar(&self.segs)),
                    (format!("{stem}.tr.txt"), srt::txt_tr(&self.segs)),
                ];
                let mut ok = 0;
                for (name, body) in &files {
                    let p = dir.join(name);
                    match std::fs::write(&p, body) {
                        Ok(_) => {
                            ok += 1;
                            self.log_line(format!("Exported: {}", p.display()));
                        }
                        Err(e) => self.log_line(format!("Export FAILED {}: {e}", p.display())),
                    }
                }
                self.exported = ok > 0;
                Task::none()
            }
            Message::OpenFolder => {
                if let Some(dir) = self.file.as_ref().and_then(|f| f.parent()) {
                    let d = dir.to_path_buf();
                    std::thread::spawn(move || {
                        let _ = open::that(&d);
                    });
                }
                Task::none()
            }
            Message::CopyAll => {
                let txt = srt::txt_ar(&self.segs);
                Task::batch(vec![clipboard::write(txt)])
                }
        }
    }

    /// Spawn the engine with the current job; `init_progress` seeds the bar.
    fn launch(&mut self, variant: &str, file: PathBuf, init_progress: Option<f32>) {
        if let Some(p) = init_progress {
            self.progress = p;
        }
        self.child_variant = Some(variant.to_string());
        let job = JobSpec {
            input: file,
            model: self.model,
            context: self.context.clone(),
            glossary: self.glossary.text(),
            models_dir: tarjama_core::models_dir().unwrap_or_default(),
            ffmpeg: String::new(),
            out_prefix: None,
            n_threads: 0,
        };
        match engine::spawn_engine(&job, variant, &self.queue) {
            Ok(spawned) => {
                let pid = spawned
                    .child
                    .lock()
                    .ok()
                    .and_then(|g| g.as_ref().map(|c| c.id().to_string()))
                    .unwrap_or_default();
                self.log_line(format!("Engine started ({variant} build, pid {pid})"));
                self.child = Some(spawned.child);
            }
            Err(e) => {
                // preferred engine missing? try the other one before failing
                let other = if variant == "fast" { "safe" } else { "fast" };
                if engine::engine_path(variant).is_none() && engine::engine_path(other).is_some() {
                    self.log_line(format!(
                        "{variant} engine not found - falling back to {other} engine"
                    ));
                    if other == "safe" {
                        self.retried_safe = true;
                    }
                    self.child_variant = Some(other.to_string());
                    match engine::spawn_engine(&job, other, &self.queue) {
                        Ok(spawned) => {
                            self.child = Some(spawned.child);
                            return;
                        }
                        Err(e2) => {
                            self.running = false;
                            self.err = Some(format!("{e2:#}"));
                            return;
                        }
                    }
                }
                self.running = false;
                self.err = Some(format!("{e:#}"));
                self.set_stage("Failed".into(), false);
            }
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(60)).map(|_| Message::Poll)
    }

    fn view(&self) -> Element<'_, Message> {
        let title = text("Tarjama Studio Native").size(24).font(Font::MONOSPACE);
        let subtitle = text("Turkish video/audio -> Arabic subtitles. 100% offline, no internet needed.")
            .size(13)
            .color(Color::from_rgb(0.65, 0.65, 0.7));

        let model_row = row![
            text("Whisper model:").size(14),
            row(Kind::all().iter().map(|k|
                radio(k.label(), *k, Some(self.model), Message::ModelSel).into()
            ))
            .spacing(14),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        let file_label = match &self.file {
            Some(f) => text(f.display().to_string()).size(13),
            None => text("no file chosen")
                .size(13)
                .color(Color::from_rgb(0.5, 0.5, 0.55)),
        };
        let pick_btn = button("1. Choose video / audio file")
            .padding([8, 14])
            .on_press_maybe(if self.running { None } else { Some(Message::PickFile) });

        let context_input = text_input(
            "Optional context hint for recognition (names, jargon...)",
            &self.context,
        )
        .on_input(Message::Context)
        .size(13)
        .padding(8);

        let glossary_label = text("Glossary (optional, one per line:  turkish=arabic)")
            .size(12)
            .color(Color::from_rgb(0.6, 0.6, 0.65));
        let glossary = text_editor(&self.glossary)
            .on_action(Message::Editor)
            .height(Length::Fixed(64.0))
            .padding(6);

        let start_btn = button(text("2. TRANSLATE  ->").size(15).color(Color::WHITE))
            .padding([10, 22])
            .style(|theme, status| {
                let mut s = iced::widget::button::primary(theme, status);
                s.background = Some(iced::Background::Color(if self.running {
                    Color::from_rgb(0.3, 0.3, 0.32)
                } else {
                    Color::from_rgb(0.05, 0.5, 0.3)
                }));
                s
            })
            .on_press_maybe(if self.running { None } else { Some(Message::Start) });
        let cancel_btn = button("Cancel")
            .padding([10, 14])
            .on_press_maybe(if self.running { Some(Message::Cancel) } else { None });

        let stage_text = text(self.stage.clone()).size(13);
        let bar = progress_bar(0.0..=1.0, self.progress).width(Length::Fill);

        let err = self.err.as_ref().map(|e| {
            text(format!("Error: {e}"))
                .size(13)
                .color(Color::from_rgb(0.95, 0.3, 0.3))
        });

        let mut export_row = row![].spacing(10).align_y(Alignment::Center);
        if !self.segs.is_empty() {
            export_row = export_row
                .push(text("3. Export:").size(13))
                .push(
                    button("SRT (Arabic)").padding([6, 10]).on_press_maybe(
                        if self.exported { None } else { Some(Message::ExportAll) },
                    ),
                )
                .push(button("All formats").padding([6, 10]).on_press(Message::ExportAll))
                .push(button("Copy AR text").padding([6, 10]).on_press(Message::CopyAll))
                .push(button("Open folder").padding([6, 10]).on_press(Message::OpenFolder));
        }

        let mut results = column![].spacing(6);
        for (i, s) in self.segs.iter().enumerate() {
            let time = format!(
                "{:02}:{:02} -> {:02}:{:02}",
                s.t0 / 60000,
                (s.t0 / 1000) % 60,
                s.t1 / 60000,
                (s.t1 / 1000) % 60
            );
            results = results.push(
                container(
                    column![
                        row![text(format!("{:>3}", i + 1)).size(11).color(Color::from_rgb(0.5, 0.5, 0.55)),
                             text(time).size(11).color(Color::from_rgb(0.5, 0.5, 0.55))]
                            .spacing(10),
                        text(s.ar.clone()).size(15).font(AR_FONT),
                        text(s.tr.clone()).size(12).color(Color::from_rgb(0.55, 0.55, 0.6)),
                    ]
                    .spacing(2),
                )
                .padding([6, 10])
                .style(|_| iced::widget::container::Style {
                    background: Some(iced::Background::Color(Color::from_rgb(0.13, 0.13, 0.16))),
                    border: iced::Border {
                        radius: 8.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            );
        }

        let results_pane = if self.segs.is_empty() {
            scrollable(
                container(
                    text(if self.running {
                        "Working... results appear here."
                    } else {
                        "Subtitles will appear here."
                    })
                    .size(13)
                    .color(Color::from_rgb(0.45, 0.45, 0.5)),
                )
                .center(Length::Fill),
            )
            .height(Length::Fill)
        } else {
            scrollable(results).height(Length::Fill)
        };

        let log_pane = scrollable(text(self.logs.clone()).size(11).font(Font::MONOSPACE))
            .height(Length::Fixed(110.0));

        let mut main = column![
            title,
            subtitle,
            horizontal_rule(6),
            model_row,
            row![pick_btn, file_label].spacing(10).align_y(Alignment::Center),
            context_input,
            glossary_label,
            glossary,
            row![start_btn, cancel_btn, bar].spacing(10).align_y(Alignment::Center),
            stage_text,
            export_row,
            horizontal_rule(6),
            results_pane,
            log_pane,
        ]
        .spacing(10)
        .padding(16);

        if let Some(e) = err {
            main = main.push(e);
        }

        container(main)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}
