use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use eframe::egui::{Color32, CornerRadius, Stroke, Vec2};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tts::Tts;
use ureq;

const CONFIG_FILE_PATH: &str = "config.json";
static DARK_MODE: AtomicBool = AtomicBool::new(false);

// Exportando estáticos com visibilidade pública (`pub`)
pub static YAMNET_MODEL: &[u8] = include_bytes!("../resources/yamnet.onnx");
pub static CLASS_MAP_CSV: &str = include_str!("../resources/yamnet_class_map.csv");

// ============================================================================
// PALETA DE CORES (AMARELO, CINZA E AZUL ESCURO)
// ============================================================================

mod theme {
    use eframe::egui::Color32;

    pub const BG_DEEP: Color32 = Color32::from_rgb(9, 12, 29);
    pub const BG_MID: Color32 = Color32::from_rgb(20, 24, 48);
    pub const GLASS_LOW: Color32 = Color32::from_rgb(17, 24, 48);
    pub const GLASS_HIGH: Color32 = Color32::from_rgb(31, 42, 78);
    pub const BORDER_SOFT: Color32 = Color32::from_rgb(108, 126, 190);
    pub const YELLOW_MAIN: Color32 = Color32::from_rgb(255, 201, 92);
    pub const YELLOW_AMBER: Color32 = Color32::from_rgb(255, 143, 106);
    pub const GRAY_DARK: Color32 = Color32::from_rgb(100, 116, 150);
    pub const NEON_PURPLE: Color32 = Color32::from_rgb(167, 139, 250);
    pub const NEON_CYAN: Color32 = Color32::from_rgb(61, 214, 204);
    pub const NEON_PINK: Color32 = Color32::from_rgb(244, 114, 182);
    pub const NEON_GREEN: Color32 = Color32::from_rgb(52, 211, 153);
    pub const NEON_RED: Color32 = Color32::from_rgb(251, 113, 133);
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(241, 245, 249);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(166, 178, 204);
}

// ============================================================================
// ESTRUTURAS DE CONFIGURAÇÃO E MENSAGENS
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub rms_threshold: f32,
    pub score_threshold: f32,
    pub alpha_smoothing: f32,
    pub cooldown_secs: u64,
    pub enabled_categories: HashSet<usize>,
    #[serde(default)]
    pub dark_mode: bool,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub custom_names: BTreeMap<usize, String>,
    #[serde(default)]
    pub alert_enabled: bool,
    #[serde(default)]
    pub alert_channel: String,
    #[serde(default)]
    pub alert_webhook_url: String,
    #[serde(default)]
    pub alert_recipient: String,
    #[serde(default)]
    pub alert_score_threshold: f32,
    #[serde(default)]
    pub alert_categories: HashSet<usize>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 0.04,
            score_threshold: 0.30,
            alpha_smoothing: 0.30,
            cooldown_secs: 2,
            enabled_categories: (0..521).collect(),
            dark_mode: true,
            paused: false,
            custom_names: BTreeMap::new(),
            alert_enabled: false,
            alert_channel: "webhook".to_string(),
            alert_webhook_url: String::new(),
            alert_recipient: String::new(),
            alert_score_threshold: 0.70,
            alert_categories: HashSet::new(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        if Path::new(CONFIG_FILE_PATH).exists() {
            if let Ok(content) = fs::read_to_string(CONFIG_FILE_PATH) {
                if let Ok(config) = serde_json::from_str::<AppConfig>(&content) {
                    println!(
                        "📂 Configurações carregadas com sucesso do arquivo '{}'",
                        CONFIG_FILE_PATH
                    );
                    return config;
                }
            }
        }
        println!(
            "⚠️ Não foi possível carregar '{}'. Usando configurações padrão.",
            CONFIG_FILE_PATH
        );
        Self::default()
    }

    pub fn save(&self) {
        if let Ok(json_string) = serde_json::to_string_pretty(self) {
            if let Err(e) = fs::write(CONFIG_FILE_PATH, json_string) {
                eprintln!("❌ Erro ao salvar configurações no disco: {}", e);
            }
        }
    }
}

pub struct ToastMessage {
    pub title: String,
    pub body: String,
    pub accent: Color32,
    pub created_at: Instant,
}

pub struct DetectionLog {
    pub class_id: usize,
    pub sound_name: String,
    pub score: f32,
    pub rms: f32,
    pub timestamp: String,
    pub created_at: Instant,
}

// ============================================================================
// APLICAÇÃO GUI (EGUI)
// ============================================================================

pub struct EarbuddyApp {
    config: Arc<RwLock<AppConfig>>,
    class_labels: Vec<String>,
    logs: Vec<DetectionLog>,
    rx_detections: Receiver<DetectionLog>,
    search_query: String,
    current_rms: Arc<RwLock<f32>>,

    started_at: Instant,
    peak_rms: f32,
    peak_hold_until: Instant,
    training_name: String,
    selected_training_class: Option<usize>,
    toast: Option<ToastMessage>,
}

impl EarbuddyApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        class_labels: Vec<String>,
        config: Arc<RwLock<AppConfig>>,
        rx_detections: Receiver<DetectionLog>,
        current_rms: Arc<RwLock<f32>>,
    ) -> Self {
        install_fonts(&cc.egui_ctx);
        let initial_dark = config.read().unwrap().dark_mode;
        DARK_MODE.store(initial_dark, Ordering::Relaxed);
        setup_visuals(&cc.egui_ctx, initial_dark);

        Self {
            config,
            class_labels,
            logs: Vec::new(),
            rx_detections,
            search_query: String::new(),
            current_rms,
            started_at: Instant::now(),
            peak_rms: 0.0,
            peak_hold_until: Instant::now(),
            training_name: String::new(),
            selected_training_class: None,
            toast: None,
        }
    }
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        "space_grotesk".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../resources/fonts/SpaceGrotesk-Regular.ttf"
        ))),
    );
    fonts.font_data.insert(
        "space_grotesk_bold".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../resources/fonts/SpaceGrotesk-Bold.ttf"
        ))),
    );
    fonts.font_data.insert(
        "jetbrains_mono".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../resources/fonts/JetBrainsMono-Regular.ttf"
        ))),
    );

    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "space_grotesk".to_owned());

    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "jetbrains_mono".to_owned());

    fonts.families.insert(
        egui::FontFamily::Name("bold".into()),
        vec!["space_grotesk_bold".to_owned(), "space_grotesk".to_owned()],
    );

    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(22.0, FontFamily::Name("bold".into())),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(13.5, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(13.0, FontFamily::Name("bold".into())),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.5, FontFamily::Monospace),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(11.0, FontFamily::Proportional),
    );
    ctx.set_style(style);
}

fn setup_visuals(ctx: &egui::Context, dark: bool) {
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.override_text_color = Some(primary_text());
    visuals.window_fill = theme::BG_MID;
    visuals.panel_fill = Color32::TRANSPARENT;
    visuals.extreme_bg_color = theme::BG_DEEP;
    visuals.faint_bg_color = theme::GLASS_LOW;

    let rounding = CornerRadius::same(10);

    visuals.widgets.noninteractive.bg_fill = theme::GLASS_LOW;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, theme::BORDER_SOFT);
    visuals.widgets.noninteractive.corner_radius = rounding;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, primary_text());

    visuals.widgets.inactive.bg_fill = theme::GLASS_LOW;
    visuals.widgets.inactive.weak_bg_fill = theme::GLASS_LOW;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, theme::BORDER_SOFT);
    visuals.widgets.inactive.corner_radius = rounding;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, primary_text());

    visuals.widgets.hovered.bg_fill = theme::GLASS_HIGH;
    visuals.widgets.hovered.weak_bg_fill = theme::GLASS_HIGH;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.5_f32, theme::YELLOW_MAIN);
    visuals.widgets.hovered.corner_radius = rounding;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, primary_text());

    visuals.widgets.active.bg_fill = theme::GLASS_HIGH;
    visuals.widgets.active.weak_bg_fill = theme::GLASS_HIGH;
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, theme::YELLOW_AMBER);
    visuals.widgets.active.corner_radius = rounding;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, primary_text());

    visuals.widgets.open.bg_fill = theme::GLASS_HIGH;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, theme::BORDER_SOFT);
    visuals.widgets.open.corner_radius = rounding;

    visuals.selection.bg_fill = Color32::from_rgba_premultiplied(250, 204, 21, 90);
    visuals.selection.stroke = Stroke::new(1.0_f32, primary_text());

    visuals.window_corner_radius = CornerRadius::same(14);
    visuals.menu_corner_radius = CornerRadius::same(10);

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 10.0);
    style.spacing.button_padding = Vec2::new(14.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(14);
    style.spacing.menu_margin = egui::Margin::same(10);
    style.spacing.slider_width = 220.0;
    style.spacing.indent = 18.0;
    ctx.set_style(style);
}

fn primary_text() -> Color32 {
    if DARK_MODE.load(Ordering::Relaxed) {
        Color32::from_rgb(226, 232, 240)
    } else {
        theme::TEXT_PRIMARY
    }
}

fn muted_text() -> Color32 {
    if DARK_MODE.load(Ordering::Relaxed) {
        Color32::from_rgb(148, 163, 184)
    } else {
        theme::TEXT_MUTED
    }
}

fn glass_card<R>(
    ui: &mut egui::Ui,
    accent: Color32,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    egui::Frame::group(ui.style())
        .fill(theme::GLASS_LOW)
        .stroke(Stroke::new(1.0_f32, theme::BORDER_SOFT))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(egui::Margin::same(14))
        .outer_margin(egui::Margin::same(4))
        .show(ui, |ui| {
            let rect = ui.max_rect();
            let painter = ui.painter();
            let top = egui::Rect::from_min_size(rect.min, Vec2::new(rect.width(), 2.0));
            painter.rect_filled(top, CornerRadius::same(2), accent.linear_multiply(0.7));
            add_contents(ui)
        })
        .inner
}

fn paint_background(ctx: &egui::Context, dark: bool) {
    let screen = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    let top = if dark {
        Color32::from_rgb(8, 10, 27)
    } else {
        Color32::from_rgb(246, 247, 255)
    };
    let bottom = if dark {
        Color32::from_rgb(24, 18, 54)
    } else {
        Color32::from_rgb(225, 235, 255)
    };
    for i in 0..32 {
        let t = i as f32 / 31.0;
        let y0 = screen.min.y + screen.height() * (i as f32 / 32.0);
        let y1 = screen.min.y + screen.height() * ((i + 1) as f32 / 32.0);
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(screen.min.x, y0), egui::pos2(screen.max.x, y1)),
            CornerRadius::ZERO,
            lerp_color(top, bottom, t),
        );
    }
    let t = ctx.input(|i| i.time) as f32;
    draw_radial_blob(
        &painter,
        egui::pos2(
            screen.min.x + screen.width() * (0.13 + 0.03 * (t * 0.22).sin()),
            screen.min.y + screen.height() * 0.12,
        ),
        180.0,
        theme::NEON_PURPLE,
        if dark { 34 } else { 22 },
    );
    draw_radial_blob(
        &painter,
        egui::pos2(
            screen.max.x - screen.width() * 0.12,
            screen.max.y - screen.height() * 0.12,
        ),
        220.0,
        theme::NEON_CYAN,
        if dark { 28 } else { 20 },
    );
    let grid = if dark {
        Color32::from_rgba_premultiplied(120, 130, 220, 14)
    } else {
        Color32::from_rgba_premultiplied(85, 105, 180, 12)
    };
    let step = 48.0;
    let mut x = screen.min.x;
    while x < screen.max.x {
        painter.line_segment(
            [egui::pos2(x, screen.min.y), egui::pos2(x, screen.max.y)],
            Stroke::new(1.0_f32, grid),
        );
        x += step;
    }
    let mut y = screen.min.y;
    while y < screen.max.y {
        painter.line_segment(
            [egui::pos2(screen.min.x, y), egui::pos2(screen.max.x, y)],
            Stroke::new(1.0_f32, grid),
        );
        y += step;
    }
}

fn draw_radial_blob(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    color: Color32,
    max_alpha: u8,
) {
    let layers = 10;
    for i in 0..layers {
        let f = i as f32 / layers as f32;
        let r = radius * (1.0 - f);
        let alpha = (max_alpha as f32 * (1.0 - f) * (1.0 - f)) as u8;
        let c = Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), alpha);
        painter.circle_filled(center, r, c);
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| -> u8 {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_premultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

fn animated_header(ui: &mut egui::Ui, ctx: &egui::Context) {
    let t = ctx.input(|i| i.time) as f32;
    let pulse = 0.5 + 0.5 * (t * 1.8).sin();

    let glow = lerp_color(theme::NEON_PURPLE, theme::NEON_CYAN, pulse);

    ui.horizontal(|ui| {
        ui.add_space(6.0);

        let (dot_rect, _) = ui.allocate_exact_size(Vec2::new(18.0, 18.0), egui::Sense::hover());
        let center = dot_rect.center();
        let painter = ui.painter();
        painter.circle_filled(center, 5.0 + 2.0 * pulse, theme::NEON_GREEN);
        painter.circle_stroke(
            center,
            9.0 + 3.0 * pulse,
            Stroke::new(1.5_f32, theme::NEON_GREEN.linear_multiply(0.35)),
        );

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("🎧 EarBuddy")
                .size(26.0)
                .strong()
                .color(glow),
        );
        ui.label(
            egui::RichText::new("· Monitoramento em Tempo Real")
                .size(14.0)
                .color(muted_text()),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new("● LIVE")
                    .size(12.0)
                    .strong()
                    .color(theme::NEON_GREEN),
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("YAMNet · ONNX")
                    .size(12.0)
                    .color(muted_text()),
            );
        });
    });
}

fn custom_rms_bar(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    rms: f32,
    threshold: f32,
    peak: f32,
) -> egui::Response {
    let desired_size = Vec2::new(ui.available_width(), 26.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter();

    painter.rect_filled(rect, CornerRadius::same(8), theme::BG_MID);
    painter.rect_stroke(
        rect,
        CornerRadius::same(8),
        Stroke::new(1.0_f32, theme::BORDER_SOFT),
        egui::StrokeKind::Inside,
    );

    let norm = (rms / 0.1).clamp(0.0, 1.0);
    let fill_w = rect.width() * norm;

    let segments = 40usize;
    let seg_w = rect.width() / segments as f32;
    let active_segs = (norm * segments as f32).ceil() as usize;
    let t = ctx.input(|i| i.time) as f32;

    for i in 0..segments {
        let x0 = rect.min.x + i as f32 * seg_w;
        let seg_rect = egui::Rect::from_min_size(
            egui::pos2(x0 + 1.0, rect.min.y + 4.0),
            Vec2::new(seg_w - 2.0, rect.height() - 8.0),
        );
        let f = i as f32 / segments as f32;
        let base = if f < 0.4 {
            lerp_color(theme::NEON_GREEN, theme::NEON_CYAN, f / 0.4)
        } else if f < 0.75 {
            lerp_color(theme::NEON_CYAN, theme::NEON_PURPLE, (f - 0.4) / 0.35)
        } else {
            lerp_color(theme::NEON_PURPLE, theme::NEON_PINK, (f - 0.75) / 0.25)
        };
        let color = if i < active_segs {
            if i + 1 == active_segs {
                let p = 0.7 + 0.3 * (t * 6.0).sin();
                base.linear_multiply(p)
            } else {
                base
            }
        } else {
            Color32::from_rgba_premultiplied(base.r(), base.g(), base.b(), 22)
        };
        painter.rect_filled(seg_rect, CornerRadius::same(2), color);
    }

    let th_x = rect.min.x + rect.width() * (threshold / 0.1).clamp(0.0, 1.0);
    painter.line_segment(
        [
            egui::pos2(th_x, rect.min.y + 2.0),
            egui::pos2(th_x, rect.max.y - 2.0),
        ],
        Stroke::new(2.0_f32, theme::NEON_RED),
    );

    let peak_norm = (peak / 0.1).clamp(0.0, 1.0);
    let peak_x = rect.min.x + rect.width() * peak_norm;
    painter.line_segment(
        [
            egui::pos2(peak_x, rect.min.y + 3.0),
            egui::pos2(peak_x, rect.max.y - 3.0),
        ],
        Stroke::new(2.0_f32, Color32::WHITE),
    );

    let label = format!("RMS  {:.4}", rms);
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::monospace(12.0),
        primary_text(),
    );

    let _ = fill_w;
    response
}

fn categorize_label(label: &str) -> &'static str {
    let l = label.to_lowercase();
    if l.contains("dog")
        || l.contains("cat")
        || l.contains("bird")
        || l.contains("animal")
        || l.contains("bark")
        || l.contains("meow")
        || l.contains("roar")
        || l.contains("frog")
        || l.contains("duck")
        || l.contains("chicken")
        || l.contains("owl")
        || l.contains("livestock")
    {
        "🐾 Animais"
    } else if l.contains("siren")
        || l.contains("alarm")
        || l.contains("fire")
        || l.contains("glass")
        || l.contains("explosion")
        || l.contains("gunshot")
        || l.contains("scream")
        || l.contains("shout")
        || l.contains("danger")
        || l.contains("warning")
        || l.contains("emergency")
        || l.contains("bell")
    {
        "🚨 Alertas e Perigo"
    } else if l.contains("speech")
        || l.contains("voice")
        || l.contains("whisper")
        || l.contains("laugh")
        || l.contains("cry")
        || l.contains("clap")
        || l.contains("footstep")
        || l.contains("cough")
        || l.contains("sneeze")
        || l.contains("human")
        || l.contains("singing")
    {
        "🗣️ Voz e Pessoas"
    } else if l.contains("music")
        || l.contains("piano")
        || l.contains("guitar")
        || l.contains("drum")
        || l.contains("violin")
        || l.contains("trumpet")
        || l.contains("instrument")
    {
        "🎵 Música e Instrumentos"
    } else if l.contains("car")
        || l.contains("vehicle")
        || l.contains("engine")
        || l.contains("train")
        || l.contains("aircraft")
        || l.contains("horn")
        || l.contains("traffic")
        || l.contains("motorcycle")
    {
        "🚗 Veículos e Trânsito"
    } else {
        "📦 Outros e Ambiente"
    }
}

fn build_grouped_labels(labels: &[String]) -> BTreeMap<String, Vec<(usize, String)>> {
    let mut groups: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();

    for (idx, label) in labels.iter().enumerate() {
        let group_name = categorize_label(label).to_string();
        groups
            .entry(group_name)
            .or_default()
            .push((idx, label.clone()));
    }

    for items in groups.values_mut() {
        items.sort_by(|a, b| a.1.to_lowercase().cmp(&b.1.to_lowercase()));
    }

    groups
}

impl eframe::App for EarbuddyApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.85, 0.87, 0.90, 1.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.screen_rect().width() <= 0.0 || ctx.screen_rect().height() <= 0.0 {
            ctx.request_repaint();
            return;
        }

        while let Ok(log) = self.rx_detections.try_recv() {
            self.toast = Some(ToastMessage {
                title: "🔊 SOM DETECTADO".to_string(),
                body: format!(
                    "{} · {:.0}% de confiança",
                    log.sound_name,
                    log.score * 100.0
                ),
                accent: if log.score >= 0.70 {
                    theme::NEON_PINK
                } else {
                    theme::NEON_CYAN
                },
                created_at: Instant::now(),
            });
            self.selected_training_class = Some(log.class_id);
            self.training_name = self
                .config
                .read()
                .unwrap()
                .custom_names
                .get(&log.class_id)
                .cloned()
                .unwrap_or_default();
            self.logs.insert(0, log);
            if self.logs.len() > 100 {
                self.logs.pop();
            }
        }

        ctx.request_repaint_after(Duration::from_millis(33));
        let dark_mode = self.config.read().unwrap().dark_mode;
        DARK_MODE.store(dark_mode, Ordering::Relaxed);
        setup_visuals(ctx, dark_mode);

        paint_background(ctx, dark_mode);

        let live_rms = *self.current_rms.read().unwrap();
        let now = Instant::now();
        if live_rms >= self.peak_rms || now > self.peak_hold_until {
            if live_rms >= self.peak_rms {
                self.peak_hold_until = now + Duration::from_millis(900);
            }
            self.peak_rms = live_rms.max(self.peak_rms * 0.985);
        } else {
            self.peak_rms *= 0.985;
        }

        let mut config_changed = false;

        egui::TopBottomPanel::top("top_bar")
            .frame(
                egui::Frame::NONE
                    .fill(Color32::TRANSPARENT)
                    .inner_margin(egui::Margin::symmetric(18, 14)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    animated_header(ui, ctx);
                    ui.add_space(12.0);
                    let mut cfg = self.config.write().unwrap();
                    let paused = cfg.paused;
                    let pause_color = if paused {
                        theme::NEON_GREEN
                    } else {
                        theme::NEON_PINK
                    };
                    let pause_label = if paused {
                        "▶ RETOMAR DETECÇÃO"
                    } else {
                        "Ⅱ PAUSAR DETECÇÃO"
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(pause_label)
                                    .strong()
                                    .color(primary_text()),
                            )
                            .fill(Color32::from_rgba_premultiplied(
                                pause_color.r(),
                                pause_color.g(),
                                pause_color.b(),
                                46,
                            ))
                            .stroke(Stroke::new(1.5_f32, pause_color))
                            .corner_radius(CornerRadius::same(10)),
                        )
                        .clicked()
                    {
                        cfg.paused = !paused;
                        if cfg.paused {
                            self.toast = None;
                        }
                        config_changed = true;
                    }
                    ui.label(
                        egui::RichText::new(if paused {
                            "Microfone em espera"
                        } else {
                            "Detecção ativa"
                        })
                        .size(11.0)
                        .strong()
                        .color(pause_color),
                    );
                });
            });

        let compact_layout = ctx.screen_rect().width() < 760.0;
        egui::SidePanel::left("settings_panel")
            .resizable(!compact_layout)
            .default_width(if compact_layout { 220.0 } else { 330.0 })
            .frame(
                egui::Frame::NONE
                    .fill(Color32::TRANSPARENT)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                glass_card(ui, theme::NEON_PURPLE, |ui| {
                    ui.label(
                        egui::RichText::new("⚙️ Parâmetros do Sistema")
                            .size(16.0)
                            .strong()
                            .color(primary_text()),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(
                            "Ajuste em tempo real — as configurações são salvas automaticamente.",
                        )
                        .size(11.0)
                        .color(muted_text()),
                    );

                    ui.add_space(10.0);
                    {
                        let mut cfg = self.config.write().unwrap();
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(if cfg.dark_mode {
                                    "◐ CYBERPUNK"
                                } else {
                                    "◌ HOLOGRÁFICO"
                                })
                                .strong()
                                .color(theme::NEON_CYAN),
                            );
                            if ui
                                .button(if cfg.dark_mode {
                                    "Modo claro"
                                } else {
                                    "Modo escuro"
                                })
                                .clicked()
                            {
                                cfg.dark_mode = !cfg.dark_mode;
                                config_changed = true;
                            }
                        });
                        ui.label(
                            egui::RichText::new("Tema adaptável para ambientes claros e baixa luz.")
                                .size(10.5)
                                .color(muted_text()),
                        );
                    }
                    ui.add_space(8.0);
                    egui::Frame::NONE
                        .fill(Color32::from_rgba_premultiplied(
                            theme::NEON_CYAN.r(),
                            theme::NEON_CYAN.g(),
                            theme::NEON_CYAN.b(),
                            22,
                        ))
                        .corner_radius(CornerRadius::same(10))
                        .inner_margin(egui::Margin::same(9))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("✦ RAG LOCAL ATIVO")
                                    .size(11.0)
                                    .strong()
                                    .color(theme::NEON_CYAN),
                            );
                            ui.label(
                                egui::RichText::new(
                                    "Recupera rótulos e nomes treinados para contextualizar cada detecção, sem enviar áudio para a nuvem.",
                                )
                                .size(10.5)
                                .color(muted_text()),
                            );
                        });
                    ui.add_space(12.0);

                    let mut cfg = self.config.write().unwrap();

                    ui.label(
                        egui::RichText::new("🎚️ SENSIBILIDADE")
                            .size(11.0)
                            .strong()
                            .color(theme::NEON_CYAN),
                    );
                    ui.add_space(4.0);

                    if ui
                        .add(
                            egui::Slider::new(&mut cfg.rms_threshold, 0.005..=0.20)
                                .text("Gatilho RMS")
                                .logarithmic(true),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut cfg.score_threshold, 0.05..=0.95)
                                .text("Confiança mínima"),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }

                    ui.add_space(10.0);
                    ui.label(
                        egui::RichText::new("🌊 SUAVIZAÇÃO & TEMPO")
                            .size(11.0)
                            .strong()
                            .color(theme::NEON_CYAN),
                    );
                    ui.add_space(4.0);

                    if ui
                        .add(
                            egui::Slider::new(&mut cfg.alpha_smoothing, 0.05..=1.00)
                                .text("Suavização (α)"),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut cfg.cooldown_secs, 1..=30)
                                .text("Cooldown (s)"),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                });

                ui.add_space(6.0);

                glass_card(ui, theme::NEON_PINK, |ui| {
                    let mut cfg = self.config.write().unwrap();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("⚡ ALERTAS EXTERNOS")
                                .size(12.0)
                                .strong()
                                .color(theme::NEON_PINK),
                        );
                        let active = cfg.alert_enabled;
                        if ui
                            .selectable_label(
                                active,
                                if active { "ATIVO" } else { "DESATIVADO" },
                            )
                            .clicked()
                        {
                            cfg.alert_enabled = !active;
                            config_changed = true;
                        }
                    });
                    ui.label(
                        egui::RichText::new("Toasts locais + disparo por webhook para SMS ou WhatsApp.")
                            .size(10.5)
                            .color(muted_text()),
                    );
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut cfg.alert_webhook_url)
                                .hint_text("URL do webhook (Twilio, n8n ou Make)")
                                .desired_width(f32::INFINITY),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut cfg.alert_recipient)
                                .hint_text("Destino: telefone ou número WhatsApp")
                                .desired_width(f32::INFINITY),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label("Canal:");
                        for (key, label) in
                            [("sms", "SMS"), ("whatsapp", "WhatsApp"), ("webhook", "Webhook")]
                        {
                            if ui.selectable_label(cfg.alert_channel == key, label).clicked() {
                                cfg.alert_channel = key.to_string();
                                config_changed = true;
                            }
                        }
                    });
                    if ui
                        .add(
                            egui::Slider::new(&mut cfg.alert_score_threshold, 0.30..=0.99)
                                .text("Confiança para alertar"),
                        )
                        .changed()
                    {
                        config_changed = true;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "{} sons selecionados para alertas",
                            cfg.alert_categories.len()
                        ))
                        .size(10.0)
                        .color(muted_text()),
                    );
                });

                ui.add_space(6.0);

                glass_card(ui, theme::NEON_CYAN, |ui| {
                    ui.label(
                        egui::RichText::new("📊 Nível do Microfone")
                            .size(14.0)
                            .strong()
                            .color(primary_text()),
                    );
                    ui.add_space(6.0);

                    let cfg = self.config.read().unwrap();
                    custom_rms_bar(ui, ctx, live_rms, cfg.rms_threshold, self.peak_rms);

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if cfg.paused {
                            ui.label(
                                egui::RichText::new("Ⅱ PAUSADO")
                                    .strong()
                                    .color(theme::NEON_PINK),
                            );
                        }
                        ui.label(
                            egui::RichText::new(format!("⚡ Pico: {:.4}", self.peak_rms))
                                .monospace()
                                .size(11.0)
                                .color(muted_text()),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                let active = !cfg.paused && live_rms > cfg.rms_threshold;
                                let (label, color) = if cfg.paused {
                                    ("Ⅱ PAUSADO", theme::NEON_PINK)
                                } else if active {
                                    ("● ATIVO", theme::NEON_GREEN)
                                } else {
                                    ("○ SILÊNCIO", muted_text())
                                };
                                ui.label(
                                    egui::RichText::new(label)
                                        .size(11.0)
                                        .strong()
                                        .color(color),
                                );
                            },
                        );
                    });
                });

                ui.add_space(6.0);

                glass_card(ui, theme::NEON_PINK, |ui| {
                    ui.label(
                        egui::RichText::new("💡 DICA")
                            .size(11.0)
                            .strong()
                            .color(theme::NEON_PINK),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Aumente o Gatilho RMS para reduzir falsos positivos em ambientes barulhentos.",
                        )
                        .size(11.5)
                        .color(muted_text()),
                    );
                });

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    let uptime = self.started_at.elapsed();
                    let h = uptime.as_secs() / 3600;
                    let m = (uptime.as_secs() % 3600) / 60;
                    let s = uptime.as_secs() % 60;
                    ui.label(
                        egui::RichText::new(format!(
                            "⏱  Uptime  {:02}:{:02}:{:02}",
                            h, m, s
                        ))
                        .monospace()
                        .size(11.0)
                        .color(muted_text()),
                    );
                });
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(Color32::TRANSPARENT)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.columns(2, |cols| {
                    // Coluna 1: Categorias
                    glass_card(&mut cols[0], theme::NEON_PURPLE, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("🔔 Categorias Notificáveis")
                                    .size(15.0)
                                    .strong()
                                    .color(primary_text()),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let cfg = self.config.read().unwrap();
                                    let sel = cfg.enabled_categories.len();
                                    let total = self.class_labels.len();
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{}/{}",
                                            sel, total
                                        ))
                                        .size(12.0)
                                        .strong()
                                        .color(theme::NEON_CYAN),
                                    );
                                },
                            );
                        });

                        ui.add_space(8.0);

                        egui::Frame::NONE
                            .fill(theme::BG_MID)
                            .stroke(Stroke::new(1.0_f32, theme::BORDER_SOFT))
                            .corner_radius(CornerRadius::same(10))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("🔎")
                                            .color(theme::NEON_CYAN),
                                    );
                                    let te = egui::TextEdit::singleline(
                                        &mut self.search_query,
                                    )
                                    .desired_width(f32::INFINITY)
                                    .hint_text("Buscar no índice RAG: ex. campainha...")
                                    .frame(false);
                                    ui.add(te);
                                });
                            });

                        ui.add_space(6.0);

                        let grouped_labels = build_grouped_labels(&self.class_labels);

                        egui::ScrollArea::vertical()
                            .id_salt("scroll_categorias")
                            .max_height(430.0)
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut cfg = self.config.write().unwrap();
                                let query = self.search_query.to_lowercase();

                                for (group_name, items) in &grouped_labels {
                                    let filtered_items: Vec<&(usize, String)> = items
                                        .iter()
                                        .filter(|(idx, label)| {
                                            query.is_empty()
                                                || label.to_lowercase().contains(&query)
                                                || cfg.custom_names.get(idx).map(|n| n.to_lowercase().contains(&query)).unwrap_or(false)
                                        })
                                        .collect();

                                    if filtered_items.is_empty() {
                                        continue;
                                    }

                                    ui.add_space(4.0);

                                    egui::CollapsingHeader::new(
                                        egui::RichText::new(format!(
                                            "{} ({})",
                                            group_name,
                                            filtered_items.len()
                                        ))
                                        .strong()
                                        .color(primary_text()),
                                    )
                                    .default_open(true)
                                    .show_background(true)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            if pill_button(ui, "☑ Marcar Grupo", theme::YELLOW_AMBER).clicked() {
                                                for &(idx, _) in &filtered_items {
                                                    cfg.enabled_categories.insert(*idx);
                                                }
                                                config_changed = true;
                                            }
                                            if pill_button(ui, "☒ Desmarcar Grupo", theme::GRAY_DARK).clicked() {
                                                for &(idx, _) in &filtered_items {
                                                    cfg.enabled_categories.remove(idx);
                                                }
                                                config_changed = true;
                                            }
                                        });
                                        ui.add_space(4.0);

                                        for &&(idx, ref label) in &filtered_items {
                                            let selected = cfg.enabled_categories.contains(&idx);
                                            let alert_selected = cfg.alert_categories.contains(&idx);
                                            let shown = cfg.custom_names.get(&idx).cloned().unwrap_or_else(|| label.clone());
                                            egui::Frame::NONE
                                                .fill(if selected { Color32::from_rgb(20, 62, 78) } else { Color32::from_rgb(22, 30, 58) })
                                                .stroke(Stroke::new(1.0_f32, if selected { theme::NEON_CYAN } else { Color32::TRANSPARENT }))
                                                .corner_radius(CornerRadius::same(8))
                                                .inner_margin(egui::Margin::symmetric(8, 5))
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        let label_text = if selected { format!("✓  {}", shown) } else { format!("○  {}", shown) };
                                                        if ui.add(egui::Button::new(egui::RichText::new(label_text).color(if selected { Color32::WHITE } else { Color32::from_rgb(225, 232, 247) })).fill(Color32::TRANSPARENT).stroke(Stroke::NONE)).clicked() {
                                                            if selected { cfg.enabled_categories.remove(&idx); } else { cfg.enabled_categories.insert(idx); }
                                                            config_changed = true;
                                                        }
                                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                            let alert_label = if alert_selected { "🔔 ALERTA" } else { "🔕" };
                                                            if ui.add(egui::Button::new(egui::RichText::new(alert_label).size(10.0).color(if alert_selected { Color32::WHITE } else { Color32::from_rgb(205, 216, 238) })).fill(Color32::TRANSPARENT).stroke(Stroke::new(1.0_f32, if alert_selected { theme::NEON_PINK } else { theme::BORDER_SOFT }))).clicked() {
                                                                if alert_selected { cfg.alert_categories.remove(&idx); } else { cfg.alert_categories.insert(idx); }
                                                                config_changed = true;
                                                            }
                                                        });
                                                    });
                                                });
                                        }
                                    });
                                }
                            });
                    });

                    // Coluna 2: Histórico + treinamento
                    glass_card(&mut cols[1], theme::NEON_CYAN, |ui| {
                        egui::Frame::NONE.fill(Color32::from_rgb(47, 35, 76)).corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::same(10)).show(ui, |ui| {
                            ui.label(egui::RichText::new("◉ MODO DE TREINAMENTO").size(11.0).strong().color(theme::NEON_PURPLE));
                            if let Some(class_id) = self.selected_training_class {
                                let raw = self.class_labels.get(class_id).map(String::as_str).unwrap_or("som detectado");
                                ui.label(egui::RichText::new(format!("Rotule ‘{}’ para melhorar o contexto das próximas leituras.", raw)).size(10.5).color(muted_text()));
                                ui.horizontal(|ui| {
                                    ui.add(egui::TextEdit::singleline(&mut self.training_name).hint_text("Nome personalizado, ex. Campainha da porta").desired_width(ui.available_width() - 90.0));
                                    if ui.button("Salvar").clicked() && !self.training_name.trim().is_empty() {
                                        self.config.write().unwrap().custom_names.insert(class_id, self.training_name.trim().to_string());
                                        config_changed = true;
                                    }
                                });
                            } else {
                                ui.label(egui::RichText::new("Quando um som aparecer, ele ficará disponível aqui para receber um nome humano.").size(10.5).color(muted_text()));
                            }
                        });
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("🚨 Últimas Detecções")
                                    .size(15.0)
                                    .strong()
                                    .color(primary_text()),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} eventos",
                                            self.logs.len()
                                        ))
                                        .size(12.0)
                                        .strong()
                                        .color(theme::NEON_CYAN),
                                    );
                                },
                            );
                        });

                        ui.add_space(8.0);

                        if self.logs.is_empty() {
                            ui.add_space(60.0);
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    egui::RichText::new("🎧")
                                        .size(38.0)
                                        .color(theme::NEON_PURPLE),
                                );
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new("Aguardando sons...")
                                        .size(13.0)
                                        .color(muted_text()),
                                );
                                ui.label(
                                    egui::RichText::new(
                                        "As detecções aparecerão aqui em tempo real.",
                                    )
                                    .size(11.0)
                                    .color(muted_text()),
                                );
                            });
                            ui.add_space(60.0);
                        } else {
                            egui::ScrollArea::vertical()
                                .id_salt("scroll_historico")
                                .max_height(500.0)
                                .auto_shrink([false; 2])
                                .show(ui, |ui| {
                                    let now = Instant::now();
                                    for log in &self.logs {
                                        render_log_entry(ui, log, now);
                                    }
                                });
                        }
                    });
                });
            });

        if let Some(toast) = &self.toast {
            if toast.created_at.elapsed() < Duration::from_secs(5) {
                egui::Area::new("alert_toast".into())
                    .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-22.0, 78.0))
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(Color32::from_rgba_premultiplied(
                                toast.accent.r(),
                                toast.accent.g(),
                                toast.accent.b(),
                                238,
                            ))
                            .stroke(Stroke::new(1.5_f32, Color32::WHITE))
                            .corner_radius(CornerRadius::same(14))
                            .inner_margin(egui::Margin::symmetric(16, 12))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(&toast.title)
                                        .size(13.0)
                                        .strong()
                                        .color(Color32::WHITE),
                                );
                                ui.label(
                                    egui::RichText::new(&toast.body)
                                        .size(11.0)
                                        .color(Color32::from_rgb(248, 250, 252)),
                                );
                            });
                    });
                ctx.request_repaint_after(Duration::from_millis(100));
            } else {
                self.toast = None;
            }
        }

        if config_changed {
            self.config.read().unwrap().save();
        }
    }
}

fn pill_button(ui: &mut egui::Ui, text: &str, accent: Color32) -> egui::Response {
    let btn = egui::Button::new(
        egui::RichText::new(text)
            .size(12.0)
            .strong()
            .color(primary_text()),
    )
    .fill(Color32::from_rgba_premultiplied(
        accent.r(),
        accent.g(),
        accent.b(),
        40,
    ))
    .stroke(Stroke::new(1.0_f32, accent))
    .corner_radius(CornerRadius::same(20));
    ui.add(btn)
}

fn render_log_entry(ui: &mut egui::Ui, log: &DetectionLog, now: Instant) {
    let age = now.saturating_duration_since(log.created_at).as_millis() as f32;
    let anim_ms = 350.0;
    let progress = (age / anim_ms).clamp(0.0, 1.0);
    let alpha = (progress * 255.0) as u8;
    let slide_x = (1.0 - progress) * 16.0;

    let (accent, tier) = if log.score >= 0.7 {
        (theme::NEON_RED, "ALTA")
    } else if log.score >= 0.45 {
        (theme::NEON_PURPLE, "MÉDIA")
    } else {
        (theme::NEON_CYAN, "BAIXA")
    };

    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.add_space(slide_x);
        egui::Frame::NONE
            .fill(Color32::from_rgba_premultiplied(
                accent.r(),
                accent.g(),
                accent.b(),
                (28.0 * (alpha as f32 / 255.0)) as u8,
            ))
            .stroke(Stroke::new(
                1.0_f32,
                Color32::from_rgba_premultiplied(accent.r(), accent.g(), accent.b(), alpha),
            ))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.set_min_width((ui.available_width() - slide_x).max(0.0));
                ui.horizontal(|ui| {
                    let (bar_rect, _) =
                        ui.allocate_exact_size(Vec2::new(3.0, 40.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(bar_rect, CornerRadius::same(2), accent);

                    ui.add_space(6.0);

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(&log.sound_name)
                                    .size(14.0)
                                    .strong()
                                    .color(primary_text()),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new(format!("⏱ {}", log.timestamp))
                                            .monospace()
                                            .size(11.0)
                                            .color(muted_text()),
                                    );
                                },
                            );
                        });

                        ui.add_space(2.0);

                        ui.horizontal(|ui| {
                            egui::Frame::NONE
                                .fill(Color32::from_rgba_premultiplied(
                                    accent.r(),
                                    accent.g(),
                                    accent.b(),
                                    60,
                                ))
                                .corner_radius(CornerRadius::same(10))
                                .inner_margin(egui::Margin::symmetric(8, 2))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} · {:.0}%",
                                            tier,
                                            log.score * 100.0
                                        ))
                                        .size(10.5)
                                        .strong()
                                        .color(accent),
                                    );
                                });

                            ui.label(
                                egui::RichText::new(format!("RMS {:.4}", log.rms))
                                    .monospace()
                                    .size(10.5)
                                    .color(muted_text()),
                            );

                            let bar_w = 90.0;
                            let (br, _) =
                                ui.allocate_exact_size(Vec2::new(bar_w, 6.0), egui::Sense::hover());
                            let p = ui.painter();
                            p.rect_filled(br, CornerRadius::same(3), theme::BG_MID);
                            let fill = egui::Rect::from_min_size(
                                br.min,
                                Vec2::new(bar_w * log.score, 6.0),
                            );
                            p.rect_filled(fill, CornerRadius::same(3), accent);
                        });
                    });
                });
            });
    });
}

// ============================================================================
// WORKER DE ÁUDIO & INFERÊNCIA
// ============================================================================

pub fn start_audio_worker(
    config: Arc<RwLock<AppConfig>>,
    class_labels: Vec<String>,
    tx_detections: Sender<DetectionLog>,
    current_rms: Arc<RwLock<f32>>,
) {
    thread::spawn(move || {
        let mut tts = Tts::default().ok();
        if let Some(ref mut speaker) = tts {
            if let Ok(voices) = speaker.voices() {
                if let Some(voice) = voices.into_iter().find(|v| {
                    let lang = v.language().to_string().to_lowercase();
                    lang == "pt-br" || lang.starts_with("pt-br") || lang == "pt_br"
                }) {
                    let _ = speaker.set_voice(&voice);
                }
            }
            let _ = speaker.set_rate(0.92);
        }

        let mut session = match (|| -> Result<Session, ort::Error> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_memory(YAMNET_MODEL)
        })() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Erro ao inicializar modelo ONNX: {:?}", e);
                return;
            }
        };

        let (tx, rx) = channel::<Vec<f32>>();
        let host = cpal::default_host();

        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                eprintln!("❌ Nenhum microfone padrão encontrado.");
                return;
            }
        };

        let config_cpal = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "❌ Falha ao obter configuração do microfone (Verifique permissões do Android): {:?}",
                    e
                );
                return;
            }
        };

        let sample_rate = config_cpal.sample_rate().0;

        let stream = match device.build_input_stream(
            &config_cpal.into(),
            move |data: &[f32], _| {
                let _ = tx.send(data.to_vec());
            },
            |err| eprintln!("❌ Erro no microfone: {}", err),
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Erro ao abrir stream de áudio: {:?}", e);
                return;
            }
        };

        if let Err(e) = stream.play() {
            eprintln!("❌ Erro ao iniciar captura de áudio: {:?}", e);
            return;
        }

        let mut raw_buffer: Vec<f32> = Vec::new();
        let mut resampled_16k_buffer: Vec<f32> = Vec::new();
        let step_ratio = (sample_rate as f32 / 16000.0).round() as usize;

        let mut smoothed_scores: Vec<f32> = Vec::new();
        let mut last_tts_time = Instant::now() - Duration::from_secs(60);

        while let Ok(samples) = rx.recv() {
            raw_buffer.extend(samples);

            while raw_buffer.len() >= step_ratio {
                let block: Vec<f32> = raw_buffer.drain(..step_ratio).collect();
                let avg_sample: f32 = block.iter().sum::<f32>() / step_ratio as f32;
                resampled_16k_buffer.push(avg_sample);
            }

            if resampled_16k_buffer.len() >= 15600 {
                let chunk: Vec<f32> = resampled_16k_buffer[..15600].to_vec();
                resampled_16k_buffer.drain(..7800);

                let rms: f32 =
                    (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();

                if let Ok(mut current_rms_guard) = current_rms.write() {
                    *current_rms_guard = rms;
                }

                let current_cfg = config.read().unwrap().clone();

                if current_cfg.paused {
                    smoothed_scores.clear();
                    if let Ok(mut current_rms_guard) = current_rms.write() {
                        *current_rms_guard = 0.0;
                    }
                    continue;
                }

                if rms > current_cfg.rms_threshold {
                    if let Ok(input_tensor) = Tensor::from_array((vec![15600usize], chunk)) {
                        if let Ok(outputs) = session.run(inputs![input_tensor]) {
                            if let Ok((_shape, current_scores)) =
                                outputs[0].try_extract_tensor::<f32>()
                            {
                                if smoothed_scores.is_empty() {
                                    smoothed_scores = current_scores.to_vec();
                                } else {
                                    let alpha = current_cfg.alpha_smoothing;
                                    for (smooth, &current) in
                                        smoothed_scores.iter_mut().zip(current_scores.iter())
                                    {
                                        *smooth = (alpha * current) + ((1.0 - alpha) * (*smooth));
                                    }
                                }

                                let mut max_score = -1.0f32;
                                let mut top_class_id = 0;

                                for (idx, &score) in smoothed_scores.iter().enumerate() {
                                    let class_id = idx % 521;
                                    if score > max_score {
                                        max_score = score;
                                        top_class_id = class_id;
                                    }
                                }

                                if max_score >= current_cfg.score_threshold
                                    && current_cfg.enabled_categories.contains(&top_class_id)
                                {
                                    let raw_sound_name = class_labels
                                        .get(top_class_id)
                                        .cloned()
                                        .unwrap_or_else(|| format!("Classe {}", top_class_id));
                                    let sound_name = current_cfg
                                        .custom_names
                                        .get(&top_class_id)
                                        .cloned()
                                        .unwrap_or(raw_sound_name);

                                    let now = Instant::now();
                                    if now.duration_since(last_tts_time)
                                        >= Duration::from_secs(current_cfg.cooldown_secs)
                                    {
                                        let _ = tx_detections.send(DetectionLog {
                                            class_id: top_class_id,
                                            sound_name: sound_name.clone(),
                                            score: max_score,
                                            rms,
                                            timestamp: chrono_time_str(),
                                            created_at: Instant::now(),
                                        });

                                        if current_cfg.alert_enabled
                                            && current_cfg.alert_categories.contains(&top_class_id)
                                            && max_score >= current_cfg.alert_score_threshold
                                            && !current_cfg.alert_webhook_url.trim().is_empty()
                                        {
                                            let url = current_cfg.alert_webhook_url.clone();
                                            let channel = current_cfg.alert_channel.clone();
                                            let recipient = current_cfg.alert_recipient.clone();
                                            let text = format!(
                                                "EarBuddy: {} detectado com {:.0}% de confiança",
                                                sound_name,
                                                max_score * 100.0
                                            );
                                            thread::spawn(move || {
                                                let payload = serde_json::json!({"channel": channel, "to": recipient, "message": text});
                                                let _ = ureq::post(&url)
                                                    .header("Content-Type", "application/json")
                                                    .send(payload.to_string());
                                            });
                                        }

                                        if let Some(ref mut v) = tts {
                                            let _ = v.speak(
                                                &format!("Som detectado. {}.", sound_name),
                                                true,
                                            );
                                        }

                                        last_tts_time = now;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    smoothed_scores.clear();
                }
            }
        }
    });
}

pub fn chrono_time_str() -> String {
    let now = std::time::SystemTime::now();
    let datetime = now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        (datetime / 3600) % 24,
        (datetime / 60) % 60,
        datetime % 60
    )
}

pub fn load_class_labels(csv_content: &str) -> Vec<String> {
    csv_content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() == 3 {
                Some(parts[2].trim().trim_matches('"').to_string())
            } else {
                None
            }
        })
        .collect()
}