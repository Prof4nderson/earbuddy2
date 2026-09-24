// Importa `android_activity` apenas quando o target for Android
#[cfg(target_os = "android")]
use android_activity::AndroidApp;

// Ponto de entrada exclusivo para Android
#[cfg(target_os = "android")]
#[no_mangle]
pub fn android_main(app: AndroidApp) {
    let mut options = eframe::NativeOptions::default();
    options.android_app = Some(app);

    let config = std::sync::Arc::new(std::sync::RwLock::new(earbuddy::AppConfig::load()));
    let class_labels = earbuddy::load_class_labels(earbuddy::CLASS_MAP_CSV);
    let (tx_detections, rx_detections) = std::sync::mpsc::channel();
    let current_rms = std::sync::Arc::new(std::sync::RwLock::new(0.0f32));

    earbuddy::start_audio_worker(
        config.clone(),
        class_labels.clone(),
        tx_detections,
        current_rms.clone(),
    );

    let _ = eframe::run_native(
        "EarBuddy",
        options,
        Box::new(move |cc| {
            Ok(Box::new(earbuddy::EarbuddyApp::new(
                cc,
                class_labels,
                config,
                rx_detections,
                current_rms,
            )))
        }),
    );
}

// Ponto de entrada exclusivo para Desktop (Windows, Linux, macOS)
#[cfg(not(target_os = "android"))]
fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };

    let config = std::sync::Arc::new(std::sync::RwLock::new(earbuddy::AppConfig::load()));
    let class_labels = earbuddy::load_class_labels(earbuddy::CLASS_MAP_CSV);
    let (tx_detections, rx_detections) = std::sync::mpsc::channel();
    let current_rms = std::sync::Arc::new(std::sync::RwLock::new(0.0f32));

    earbuddy::start_audio_worker(
        config.clone(),
        class_labels.clone(),
        tx_detections,
        current_rms.clone(),
    );

    let _ = eframe::run_native(
        "EarBuddy",
        options,
        Box::new(move |cc| {
            Ok(Box::new(earbuddy::EarbuddyApp::new(
                cc,
                class_labels,
                config,
                rx_detections,
                current_rms,
            )))
        }),
    );
}