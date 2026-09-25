use crate::pipeline::{self, Kind, Params, Prog, Queue, Seg};
use crate::srt;
use anyhow::Result;
use iced::widget::{button, column, container, horizontal_rule, progress_bar, radio, row, scrollable, text, text_editor, text_input};
use iced::{clipboard, font, Alignment, Color, Element, Font, Length, Subscription, Task, Theme};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const NOTO_SANS: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
const NOTO_SANS_ARABIC: &[u8] = include_bytes!("../assets/fonts/NotoSansArabic-Regular.ttf");
const AR_FONT: Font = Font::with_name("Noto Sans Arabic");

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
            // automation hook for headless testing (TARJAMA_AUTOTEST=1 + TARJAMA_AUTOTEST_FILE=path)
            if std::env::var("TARJAMA_AUTOTEST").as_deref() == Ok("1") {
                if let Ok(f) = std::env::var("TARJAMA_AUTOTEST_FILE") {
                    app.file = Some(PathBuf::from(f));
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
    cancel_flag: Arc<AtomicBool>,
    queue: Queue,
    stage: String,
    progress: f32,
    logs: String,
    segs: Vec<Seg>,
    err: Option<String>,
    exported: bool,
    autostart: bool,
    autotest: bool,
    autotest_at: Option<std::time::Instant>,
    boot: bool,
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
            cancel_flag: Arc::new(AtomicBool::new(false)),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            stage: String::from("Idle - choose a file to begin"),
            progress: 0.0,
            logs: String::new(),
            segs: Vec::new(),
            err: None,
            exported: false,
            autostart: false,
            autotest: false,
            autotest_at: None,
            boot: true,
        }
    }

    fn log_line(&mut self, s: impl Into<String>) {
        let s: String = s.into();
        println!("{s}");
        self.logs.push_str(&s);
        self.logs.push('\n');
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::QuitNow => iced::exit(),
            Message::Noop => Task::none(),
            Message::ModelSel(k) => {
                self.model = k;
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
                self.err = None;
                self.segs.clear();
                self.exported = false;
                self.progress = 0.0;
                self.stage = "Starting...".into();
                self.running = true;
                self.cancel_flag = Arc::new(AtomicBool::new(false));
                let params = Params {
                    input: file,
                    model: self.model,
                    context: self.context.clone(),
                    glossary: self.glossary.text(),
                };
                let q = self.queue.clone();
                let cancel = self.cancel_flag.clone();
                std::thread::spawn(move || pipeline::run(params, q, cancel));
                Task::none()
            }
            Message::Cancel => {
                if self.running {
                    self.cancel_flag.store(true, Ordering::Relaxed);
                    self.log_line("Cancel requested - stopping after current step...");
                }
                Task::none()
            }
            Message::Poll => {
                // drain worker queue
                let msgs: Vec<Prog> = {
                    let mut g = self.queue.lock().unwrap();
                    g.drain(..).collect()
                };
                let mut task = Task::none();
                for m in msgs {
                    match m {
                        Prog::Stage(s) => {
                            self.stage = s;
                            self.progress = 0.0;
                        }
                        Prog::Asr(p) => {
                            self.progress = p;
                            self.stage = "Transcribing Turkish speech...".into();
                        }
                        Prog::Mt(a, b) => {
                            self.progress = if b > 0 { a as f32 / b as f32 } else { 0.0 };
                            self.stage = format!("Translating to Arabic ({a}/{b})...");
                        }
                        Prog::Log(s) => self.log_line(s),
                        Prog::Done(segs) => {
                            self.running = false;
                            self.progress = 1.0;
                            self.stage = format!("Done - {} subtitle lines ready", segs.len());
                            self.segs = segs;
                            self.log_line(format!(
                                "DONE: {} lines produced",
                                self.segs.len()
                            ));
                            if self.autotest {
                                // exercise the export button handler end-to-end,
                                // linger a few seconds so headless screenshots
                                // capture the results pane, then quit
                                let _ = self.update(Message::ExportAll);
                                let _ = std::fs::write(
                                    "/tmp/tarjama_autotest_ok",
                                    format!("{}\n", self.segs.len()),
                                );
                                self.log_line("AUTOTEST COMPLETE");
                                self.autotest_at = Some(std::time::Instant::now());
                            }
                        }
                        Prog::Failed(e) => {
                            self.running = false;
                            self.err = Some(e.clone());
                            self.stage = "Failed".into();
                            self.log_line(format!("ERROR: {e}"));
                        }
                    }
                }
                if let Some(t) = self.autotest_at {
                    if t.elapsed() > std::time::Duration::from_secs(7) {
                        return iced::exit();
                    }
                }
                if self.boot {
                    self.boot = false;
                }
                if self.autostart
                    && self.file.is_some()
                    && !self.running
                    && self.segs.is_empty()
                    && self.err.is_none()
                {
                    self.autostart = false;
                    task = self.update(Message::Start);
                }
                task
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
                    (format!("{}.ar.srt", stem), srt::srt_ar(&self.segs)),
                    (format!("{}.bilingual.srt", stem), srt::srt_bilingual(&self.segs)),
                    (format!("{}.ar.vtt", stem), srt::vtt_ar(&self.segs)),
                    (format!("{}.ar.txt", stem), srt::txt_ar(&self.segs)),
                    (format!("{}.tr.txt", stem), srt::txt_tr(&self.segs)),
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
                let txt = crate::srt::txt_ar(&self.segs);
                Task::batch(vec![clipboard::write(txt)])
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
            None => text("no file chosen").size(13).color(Color::from_rgb(0.5, 0.5, 0.55)),
        };
        let pick_btn = button("1. Choose video / audio file")
            .padding([8, 14])
            .on_press(Message::PickFile);

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
                .push(
                    button("All formats").padding([6, 10]).on_press(Message::ExportAll),
                )
                .push(button("Copy AR text").padding([6, 10]).on_press(Message::CopyAll))
                .push(
                    button("Open folder").padding([6, 10]).on_press(Message::OpenFolder),
                );
        }

        let mut results = column![].spacing(6);
        for (i, s) in self.segs.iter().enumerate() {
            let time = format!("{:02}:{:02} -> {:02}:{:02}", s.t0 / 60000, (s.t0 / 1000) % 60, s.t1 / 60000, (s.t1 / 1000) % 60);
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
                container(text(if self.running { "Working... results appear here." } else { "Subtitles will appear here." })
                    .size(13)
                    .color(Color::from_rgb(0.45, 0.45, 0.5)))
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
