//! MCP stdio client for Niri's background-input IPC. No compositor runs in this process.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{self, DirBuilder};
use std::io::{self, BufRead, BufReader, Cursor, Read, Write};
use std::net::Shutdown;
use std::os::fd::AsFd;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use base64::Engine;
use niri_ipc::socket::Socket;
use niri_ipc::{AgentEvent, AgentScreenshotInfo, AgentUnicodeInput, Request, Response, Window};
use serde_json::{json, Value};

#[cfg(feature = "dbus")]
mod accessibility;
mod session;

pub fn run(
    isolated: bool,
    desktop: bool,
    headless: Option<crate::cli::HeadlessSize>,
    workspace: Option<String>,
) -> Result<()> {
    if isolated {
        return session::isolate(headless.unwrap_or(crate::cli::HeadlessSize(1280, 800)));
    }
    let automatic_workspace = workspace.as_deref() == Some("");
    if automatic_workspace {
        std::env::remove_var("NIRI_MCP_WORKSPACE");
    } else if let Some(workspace) = &workspace {
        std::env::set_var("NIRI_MCP_WORKSPACE", workspace);
    }
    if desktop || workspace.is_some() || std::env::var_os("NIRI_MCP_WORKSPACE").is_some() {
        session::desktop_environment()?;
    }
    let input = BufReader::new(Input::new()?);
    match headless {
        Some(size) => session::run(size, input),
        None => {
            #[cfg(feature = "dbus")]
            if let Err(error) = accessibility::bounded(accessibility::enable()) {
                tracing::warn!(?error, "could not enable desktop accessibility");
            }
            serve(input, automatic_workspace)
        }
    }
}

struct Input {
    file: fs::File,
    #[cfg(target_os = "linux")]
    signals: std::os::fd::OwnedFd,
}

impl Input {
    fn new() -> io::Result<Self> {
        Ok(Self {
            file: fs::File::from(io::stdin().as_fd().try_clone_to_owned()?),
            #[cfg(target_os = "linux")]
            signals: crate::utils::signals::termination_fd()?,
        })
    }
}

impl Read for Input {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            loop {
                let mut fds = [
                    libc::pollfd {
                        fd: self.file.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                    libc::pollfd {
                        fd: self.signals.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                ];
                let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, -1) };
                if ready < 0 {
                    let error = io::Error::last_os_error();
                    if error.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(error);
                }
                if fds[1].revents != 0 {
                    return Err(io::Error::from(io::ErrorKind::ConnectionAborted));
                }
                break;
            }
        }
        self.file.read(bytes)
    }
}

const MAX_REQUEST: u64 = 1024 * 1024;
const MAX_IMAGE: usize = 2 * 1024 * 1024;
const MAX_SIDE: u32 = 1920;
const MAX_CONTEXTS: usize = 32;

#[derive(Default)]
struct Control {
    active: Option<String>,
    pending: HashMap<String, bool>,
    shutdown: Option<UnixStream>,
    closed: bool,
}

thread_local! {
    static CONNECTION: RefCell<Option<Socket>> = const { RefCell::new(None) };
    static CONTROL: RefCell<Option<Arc<Mutex<Control>>>> = const { RefCell::new(None) };
}

pub(super) fn cancelled() -> bool {
    CONTROL.with_borrow(|control| {
        control.as_ref().is_some_and(|control| {
            let control = control.lock().unwrap();
            control.closed
                || control
                    .active
                    .as_ref()
                    .is_some_and(|id| control.pending.get(id) == Some(&true))
        })
    })
}

fn request(request: Request) -> Result<Response> {
    ensure!(!cancelled(), "request cancelled");
    CONNECTION.with_borrow_mut(|connection| {
        if connection.is_none() {
            let mut socket = Socket::connect()?;
            socket.set_timeout(Some(Duration::from_secs(15)))?;
            CONTROL.with_borrow(|control| -> Result<()> {
                control
                    .as_ref()
                    .context("MCP control is not initialized")?
                    .lock()
                    .unwrap()
                    .shutdown = Some(socket.shutdown_handle()?);
                Ok(())
            })?;
            ensure!(!cancelled(), "request cancelled");
            socket
                .send(if let Ok(workspace) = std::env::var("NIRI_MCP_WORKSPACE") {
                    Request::AgentWorkspace { workspace }
                } else {
                    Request::AgentSession
                })?
                .map_err(anyhow::Error::msg)?;
            *connection = Some(socket);
        }
        match connection.as_mut().unwrap().send(request) {
            Ok(reply) => reply.map_err(anyhow::Error::msg),
            Err(error) => {
                *connection = None;
                CONTROL.with_borrow(|control| {
                    if let Some(socket) = control
                        .as_ref()
                        .and_then(|c| c.lock().unwrap().shutdown.take())
                    {
                        let _ = socket.shutdown(Shutdown::Both);
                    }
                });
                Err(error.into())
            }
        }
    })
}

fn workspace_status() -> Result<Option<niri_ipc::AgentWorkspace>> {
    let Ok(workspace) = std::env::var("NIRI_MCP_WORKSPACE") else {
        return Ok(None);
    };
    match request(Request::AgentWorkspaceStatus { workspace })? {
        Response::AgentWorkspace(info) => Ok(Some(info)),
        _ => bail!("unexpected agent workspace response"),
    }
}

fn thread_workspace(params: &Value) -> Result<String> {
    let thread = std::env::var("T3CODE_THREAD_ID")
        .ok()
        .or_else(|| params["_meta"]["threadId"].as_str().map(str::to_owned))
        .or_else(|| std::env::var("CODEX_THREAD_ID").ok())
        .context(
            "Thread identity is missing. Use --workspace NAME to choose an explicit workspace.",
        )?;
    ensure!(
        !thread.is_empty() && thread.len() <= 4096,
        "invalid thread identity"
    );
    // A fixed hash keeps names stable across MCP restarts and bounds delegated thread IDs.
    let hash = thread.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    Ok(format!("agents-{hash:016x}"))
}

fn windows() -> Result<Vec<Window>> {
    match request(Request::Windows)? {
        Response::Windows(mut windows) => {
            if let Some(workspace) = workspace_status()? {
                windows.retain(|w| w.workspace_id == Some(workspace.id));
            }
            Ok(windows)
        }
        _ => bail!("unexpected window-list reply"),
    }
}

fn window(id: u64) -> Result<Window> {
    windows()?
        .into_iter()
        .find(|w| w.id == id)
        .context("window no longer exists")
}

enum Target {
    Window(Window),
    Layer(niri_ipc::LayerSurface),
}

impl Target {
    fn id(&self) -> u64 {
        match self {
            Self::Window(w) => w.id,
            Self::Layer(l) => l.id,
        }
    }

    #[cfg(feature = "dbus")]
    fn pid(&self) -> Option<i32> {
        match self {
            Self::Window(w) => w.pid,
            Self::Layer(l) => l.pid,
        }
    }

    #[cfg(feature = "dbus")]
    fn title(&self) -> Option<&str> {
        match self {
            Self::Window(w) => w.title.as_deref(),
            Self::Layer(l) => Some(&l.namespace),
        }
    }

    fn app_id(&self) -> Option<&str> {
        match self {
            Self::Window(w) => w.app_id.as_deref(),
            Self::Layer(l) => Some(&l.namespace),
        }
    }
}

fn layers() -> Result<Vec<niri_ipc::LayerSurface>> {
    if std::env::var_os("NIRI_MCP_WORKSPACE").is_some() {
        return Ok(Vec::new());
    }
    match request(Request::Layers)? {
        Response::Layers(layers) => Ok(layers),
        _ => bail!("unexpected layer-list reply"),
    }
}

fn target_id(args: &Value) -> Result<u64> {
    match (args.get("window_id"), args.get("layer_id")) {
        (Some(_), None) => integer(args, "window_id"),
        (None, Some(_)) => integer(args, "layer_id"),
        _ => bail!("provide exactly one of window_id or layer_id"),
    }
}

fn target(args: &Value) -> Result<Target> {
    let id = target_id(args)?;
    if args.get("layer_id").is_some() {
        Ok(Target::Layer(
            layers()?
                .into_iter()
                .find(|l| l.id == id)
                .context("layer surface no longer exists")?,
        ))
    } else {
        Ok(Target::Window(window(id)?))
    }
}

fn argument<'a>(args: &'a Value, name: &str) -> Result<&'a Value> {
    args.get(name).with_context(|| format!("missing {name}"))
}

fn number(args: &Value, name: &str) -> Result<f64> {
    argument(args, name)?
        .as_f64()
        .filter(|n| n.is_finite())
        .with_context(|| format!("{name} must be finite"))
}

fn integer(args: &Value, name: &str) -> Result<u64> {
    argument(args, name)?
        .as_u64()
        .with_context(|| format!("{name} must be a non-negative integer"))
}

fn string<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    argument(args, name)?
        .as_str()
        .with_context(|| format!("{name} must be a string"))
}

fn boolean(args: &Value, name: &str, default: bool) -> Result<bool> {
    args.get(name).map_or(Ok(default), |v| {
        v.as_bool()
            .with_context(|| format!("{name} must be a boolean"))
    })
}

fn tool(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    read_only: bool,
) -> Value {
    json!({"name": name, "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required, "additionalProperties": false},
        "annotations": {"readOnlyHint": read_only, "destructiveHint": !read_only, "openWorldHint": true}})
}

fn tools() -> Vec<Value> {
    let id = json!({"type":"integer", "minimum":0});
    let coordinate = json!({"type":"number", "description":"Pixel coordinate in the latest returned screenshot of this window."});
    let text = json!({"type":"string"});
    let modifiers = json!({"type":"array","items":{"enum":["Ctrl","Shift","Alt","Super"]},"maxItems":4,"uniqueItems":true});
    let tools = vec![
        tool("get_session_state", "Read the host, workspace and pause state.", json!({}), &[], true),
        tool("request_handoff", "Pause automation for human help. Tell the user what to do in the agent workspace, then wait for their confirmation.", json!({"reason":text}), &["reason"], false),
        tool("resume_session", "Resume workspace automation only after the user explicitly returns control, for example by saying done in chat. Inspect the app again before continuing.", json!({}), &[], false),
        tool("list_layers", "List mapped layer-shell surfaces with stable IDs, namespaces, process IDs and output-local geometry. Use layer_id to target one explicitly.", json!({}), &[], true),
        tool("list_windows", "List Niri windows, stable IDs, process IDs, layout and real focus state.", json!({}), &[], true),
        tool("get_app_state", "Inspect one exact window, its accessibility tree and optionally its screenshot. Screenshot and input preserve real focus. Accessibility actions require the returned snapshot_id and element_index.", json!({"window_id":id,"include_screenshot":{"type":"boolean"},"max_nodes":{"type":"integer","minimum":1,"maximum":2000},"max_depth":{"type":"integer","minimum":1,"maximum":64}}), &["window_id"], true),
        tool("screenshot", "Capture one window without notifications, clipboard writes or focus changes. Images are bounded to 1920 pixels per side and 2 MiB. Coordinates for pointer tools use the returned image pixels.", json!({"window_id":id}), &["window_id"], true),
        tool("launch_app", "Launch an app and confirm new windows on the agent workspace. Other modes queue the command. Use the installed browser launcher with --new-window and its normal profile unless isolation was requested.", json!({"command":{"type":"array","items":text,"minItems":1}}), &["command"], false),
        tool("move_pointer", "Hover within one exact window using its latest screenshot coordinates, without moving the real pointer.", json!({"window_id":id,"x":coordinate,"y":coordinate}), &["window_id","x","y"], false),
        tool("click", "Click within an explicitly targeted window. Coordinates use its latest screenshot. Optional modifiers are held only for this gesture. No foreground fallback.", json!({"window_id":id,"x":coordinate,"y":coordinate,"button":{"enum":["left","middle","right"]},"click_count":{"type":"integer","minimum":1,"maximum":3},"modifiers":modifiers}), &["window_id","x","y"], false),
        tool("scroll", "Scroll within one exact window at the specified screenshot coordinates. Deltas are logical pixels, positive down and right. Optional modifiers are held only for this gesture.", json!({"window_id":id,"x":coordinate,"y":coordinate,"vertical":{"type":"number"},"horizontal":{"type":"number"},"modifiers":modifiers}), &["window_id","x","y","vertical"], false),
        tool("drag", "Press, move and release within one exact window using screenshot coordinates. Optional modifiers are held only for this gesture.", json!({"window_id":id,"start_x":coordinate,"start_y":coordinate,"end_x":coordinate,"end_y":coordinate,"modifiers":modifiers}), &["window_id","start_x","start_y","end_x","end_y"], false),
        tool("type_text", "Type literal text into one window's current editable element without changing real focus. A different foreground window of the same Wayland client causes refusal. Prefer set_value when an editable accessibility element is available.", json!({"window_id":id,"text":text,"unicode_input":{"enum":["keymap","ctrl_shift_u"]}}), &["window_id","text"], false),
        tool("press_key", "Press and release physical keyboard keys, for example Ctrl+A, Return, Escape or Alt+Left. Letter positions use the US evdev layout. No focus changes or global input fallback.", json!({"window_id":id,"key":text}), &["window_id","key"], false),
    ];
    #[cfg(feature = "dbus")]
    let tools = {
        let mut tools = tools;
        let element = json!({"window_id":id,"snapshot_id":id,"element_index":id,"action":text});
        tools.push(tool("perform_action", "Invoke a named AT-SPI action from the latest snapshot. Mutating tools invalidate that snapshot. This may trigger application behavior such as opening dialogs.", element, &["window_id","snapshot_id","element_index","action"], false));
        tools.push(tool("set_value", "Replace the text or numeric value of an element from the latest accessibility snapshot. Mutating tools invalidate that snapshot.", json!({"window_id":id,"snapshot_id":id,"element_index":id,"value":text}), &["window_id","snapshot_id","element_index","value"], false));
        tools
    };
    tools
        .into_iter()
        .map(|mut tool| {
            let schema = &mut tool["inputSchema"];
            if schema["properties"].get("window_id").is_some() {
                schema["properties"]["layer_id"] = id.clone();
                schema["required"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|v| v != "window_id");
                schema["oneOf"] = json!([{"required":["window_id"]},{"required":["layer_id"]}]);
                let description = tool["description"]
                    .as_str()
                    .unwrap()
                    .replace("window", "surface");
                tool["description"] = json!(description);
            }
            tool
        })
        .collect()
}

struct CaptureDir(PathBuf);

impl Drop for CaptureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Capture {
    info: AgentScreenshotInfo,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
struct Geometry {
    native_width: u32,
    native_height: u32,
    width: u32,
    height: u32,
    scale: f64,
    geometry_x: i32,
    geometry_y: i32,
}

fn capture(id: u64) -> Result<Capture> {
    let path = std::env::temp_dir().join(format!("niri-mcp-{:032x}", fastrand::u128(..)));
    DirBuilder::new().mode(0o700).create(&path)?;
    let dir = CaptureDir(path);
    let png = dir.0.join("window.png");
    let reply = request(Request::AgentScreenshot {
        workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
        window_id: id,
        path: png
            .to_str()
            .context("temporary path is not UTF-8")?
            .to_owned(),
        show_pointer: false,
    })?;
    let Response::AgentScreenshot(info) = reply else {
        bail!("unexpected window capture reply")
    };
    // Bound allocations before decoding an unusually large client surface.
    ensure!(
        u64::from(info.width) * u64::from(info.height) <= 32 * 1024 * 1024,
        "window is too large to capture"
    );
    let mut bytes = Vec::new();
    fs::File::open(png)?
        .take(128 * 1024 * 1024)
        .read_to_end(&mut bytes)?;
    Ok(Capture { info, bytes })
}

impl Capture {
    fn payload(self) -> Result<(Value, Value, Geometry)> {
        let Self { info, mut bytes } = self;
        let ratio = (f64::from(MAX_SIDE) / f64::from(info.width.max(info.height))).min(1.0);
        let mut width = (f64::from(info.width) * ratio).floor().max(1.0) as u32;
        let mut height = (f64::from(info.height) * ratio).floor().max(1.0) as u32;
        if ratio < 1.0 || bytes.len() > MAX_IMAGE {
            let mut reader = png::Decoder::new(Cursor::new(&bytes)).read_info()?;
            let mut pixels = vec![
                0;
                reader
                    .output_buffer_size()
                    .context("PNG buffer is too large")?
            ];
            let frame = reader.next_frame(&mut pixels)?;
            ensure!(
                frame.color_type == png::ColorType::Rgba && frame.bit_depth == png::BitDepth::Eight,
                "expected RGBA8 window capture"
            );
            loop {
                let mut resized = Vec::with_capacity(width as usize * height as usize * 4);
                for y in 0..height {
                    for x in 0..width {
                        let source_x = u64::from(x) * u64::from(info.width) / u64::from(width);
                        let source_y = u64::from(y) * u64::from(info.height) / u64::from(height);
                        let offset = (source_y * u64::from(info.width) + source_x) as usize * 4;
                        resized.extend_from_slice(&pixels[offset..offset + 4]);
                    }
                }
                bytes.clear();
                crate::utils::write_png_rgba8(&mut bytes, width, height, &resized)?;
                if bytes.len() <= MAX_IMAGE {
                    break;
                }
                ensure!(width > 1 || height > 1, "could not bound screenshot size");
                width = (width / 2).max(1);
                height = (height / 2).max(1);
            }
        }
        let geometry = Geometry {
            native_width: info.width,
            native_height: info.height,
            width,
            height,
            scale: info.scale,
            geometry_x: info.geometry_x,
            geometry_y: info.geometry_y,
        };
        let metadata = json!({"width":width,"height":height,"native_width":info.width,"native_height":info.height,"scale":info.scale,"geometry_x":info.geometry_x,"geometry_y":info.geometry_y});
        let image = json!({"type":"image","mimeType":"image/png","data":base64::engine::general_purpose::STANDARD.encode(bytes)});
        Ok((metadata, image, geometry))
    }
}

#[derive(Default)]
struct Server {
    connection: Option<Socket>,
    last_used: Option<Instant>,
    handoff: bool,
    captures: HashMap<u64, Geometry>,
    #[cfg(feature = "dbus")]
    accessibility: accessibility::Accessibility,
}

impl Server {
    fn call_with_connection(&mut self, name: &str, args: &Value) -> Result<Value> {
        CONNECTION.with_borrow_mut(|connection| *connection = self.connection.take());
        let result = (|| {
            let shutdown = CONNECTION.with_borrow(|connection| {
                connection.as_ref().map(Socket::shutdown_handle).transpose()
            })?;
            CONTROL.with_borrow(|control| {
                control.as_ref().unwrap().lock().unwrap().shutdown = shutdown;
            });
            self.call(name, args)
        })();
        self.connection = CONNECTION.with_borrow_mut(Option::take);
        CONTROL.with_borrow(|control| {
            control.as_ref().unwrap().lock().unwrap().shutdown = None;
        });
        result
    }

    fn invalidate(&mut self) {
        #[cfg(feature = "dbus")]
        self.accessibility.invalidate();
    }

    fn point(&mut self, id: u64, x: f64, y: f64) -> Result<(f64, f64)> {
        let geometry = *self
            .captures
            .get(&id)
            .context("take a screenshot or get_app_state with a screenshot before pointer input")?;
        ensure!(
            x >= 0.0 && y >= 0.0 && x < f64::from(geometry.width) && y < f64::from(geometry.height),
            "point is outside the window screenshot"
        );
        Ok((
            (x * f64::from(geometry.native_width) / f64::from(geometry.width)
                - f64::from(geometry.geometry_x))
                / geometry.scale,
            (y * f64::from(geometry.native_height) / f64::from(geometry.height)
                - f64::from(geometry.geometry_y))
                / geometry.scale,
        ))
    }

    fn input(&mut self, id: u64, events: Vec<AgentEvent>) -> Result<()> {
        self.invalidate();
        let result = request(Request::AgentInput {
            workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
            window_id: id,
            geometry: events
                .iter()
                .any(|e| matches!(e, AgentEvent::Motion { .. }))
                .then(|| {
                    self.captures.get(&id).map(|g| AgentScreenshotInfo {
                        width: g.native_width,
                        height: g.native_height,
                        scale: g.scale,
                        geometry_x: g.geometry_x,
                        geometry_y: g.geometry_y,
                    })
                })
                .flatten(),
            events,
        });
        if result.is_err() {
            // A motion can fail after a press in an earlier drag request.
            let _ = request(Request::AgentInput {
                workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
                window_id: id,
                geometry: None,
                events: vec![AgentEvent::KeyboardLeave, AgentEvent::PointerLeave],
            });
        }
        result.map(|_| ())
    }

    fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
        let status = workspace_status()?;
        self.handoff = status
            .as_ref()
            .is_some_and(|w| w.paused && w.reason.is_some() && !w.handoff_for_other_session);
        if !matches!(
            name,
            "get_session_state" | "request_handoff" | "resume_session"
        ) && status.as_ref().is_some_and(|w| w.paused)
        {
            self.invalidate();
            let status = status.as_ref().unwrap();
            if status.handoff_for_other_session {
                bail!("Another agent session is waiting for human help: {}. Wait for that handoff to finish in its thread without asking the user to resume this thread.", status.reason.as_deref().unwrap_or("human assistance"));
            }
            bail!("Workspace automation is paused. Check get_session_state and wait for the user to return control before calling resume_session.");
        }
        let mut content = Vec::new();
        let mut pending_capture = None;
        let data = match name {
            "get_session_state" => {
                let mut state =
                    status.map_or_else(|| Ok(json!({"workspace":null})), serde_json::to_value)?;
                state["host"] = std::fs::read_to_string("/proc/sys/kernel/hostname")
                    .map(|name| json!(name.trim()))
                    .unwrap_or(Value::Null);
                state
            }
            "request_handoff" | "resume_session" => {
                status.context("human handoff requires workspace mode")?;
                let workspace = std::env::var("NIRI_MCP_WORKSPACE")?;
                self.invalidate();
                let control = if name == "request_handoff" {
                    Request::AgentHandoff {
                        workspace,
                        reason: string(args, "reason")?.to_owned(),
                    }
                } else {
                    Request::AgentResume { workspace }
                };
                match request(control)? {
                    Response::AgentWorkspace(info) => {
                        self.handoff =
                            info.paused && info.reason.is_some() && !info.handoff_for_other_session;
                        serde_json::to_value(info)?
                    }
                    _ => bail!("unexpected workspace control response"),
                }
            }
            "list_windows" => json!({"windows":windows()?}),
            "list_layers" => json!({"layers":layers()?}),
            "launch_app" => {
                let command: Vec<String> =
                    serde_json::from_value(argument(args, "command")?.clone())?;
                ensure!(
                    command.first().is_some_and(|s| !s.is_empty()),
                    "command requires an executable"
                );
                self.invalidate();
                let command = [
                    "env".to_owned(),
                    "NO_AT_BRIDGE=0".to_owned(),
                    "GTK_A11Y=atspi".to_owned(),
                    "ACCESSIBILITY_ENABLED=1".to_owned(),
                ]
                .into_iter()
                .chain(command)
                .collect();
                match request(Request::AgentSpawn {
                    command,
                    workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
                })? {
                    Response::AgentSpawn(windows) => json!({"window_ids":windows}),
                    Response::Handled if status.is_none() => {
                        json!({"queued":true,"note":"Confirm application startup with list_windows."})
                    }
                    Response::Handled => bail!("The compositor did not confirm the app launch. Restart Niri to use workspace launch confirmation."),
                    _ => bail!("unexpected application launch response"),
                }
            }
            "screenshot" | "get_app_state" => {
                let window = target(args)?;
                let id = window.id();
                request(Request::AgentInput {
                    geometry: None,
                    workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
                    window_id: id,
                    events: vec![],
                })?;
                let mut data = match &window {
                    Target::Window(w) => json!({"window":w}),
                    Target::Layer(l) => json!({"layer":l}),
                };
                if name == "get_app_state" {
                    self.invalidate();
                    #[cfg(feature = "dbus")]
                    match self.accessibility.snapshot(&window, args) {
                        Ok(snapshot) => data
                            .as_object_mut()
                            .unwrap()
                            .extend(snapshot.as_object().unwrap().clone()),
                        Err(error) => data["accessibility_error"] = json!(format!("{error:#}")),
                    }
                    #[cfg(not(feature = "dbus"))]
                    {
                        data["accessibility_error"] = json!("Niri was built without D-Bus support");
                    }
                }
                if name == "screenshot" || boolean(args, "include_screenshot", true)? {
                    let (metadata, image, geometry) = capture(id)?.payload()?;
                    pending_capture = Some((id, geometry));
                    data["screenshot"] = metadata;
                    content.push(image);
                }
                data
            }
            "move_pointer" | "click" | "scroll" | "drag" => {
                let id = target(args)?.id();
                let mut events = modifiers(args)?;
                let release = events.split_off(events.len() / 2);
                if name == "drag" {
                    let (start_x, start_y) = (number(args, "start_x")?, number(args, "start_y")?);
                    let (end_x, end_y) = (number(args, "end_x")?, number(args, "end_y")?);
                    let (sx, sy) = self.point(id, start_x, start_y)?;
                    let (ex, ey) = self.point(id, end_x, end_y)?;
                    events.extend([
                        AgentEvent::Motion { x: sx, y: sy },
                        AgentEvent::Button {
                            button: 272,
                            pressed: true,
                        },
                    ]);
                    self.input(id, events)?;
                    for step in 1..=20 {
                        let t = f64::from(step) / 20.0;
                        std::thread::sleep(Duration::from_millis(10));
                        let motion = vec![AgentEvent::Motion {
                            x: sx + (ex - sx) * t,
                            y: sy + (ey - sy) * t,
                        }];
                        self.input(id, motion)?;
                    }
                    events = vec![AgentEvent::Button {
                        button: 272,
                        pressed: false,
                    }];
                } else {
                    let (x, y) = self.point(id, number(args, "x")?, number(args, "y")?)?;
                    events.push(AgentEvent::Motion { x, y });
                    if name == "click" {
                        let button = match args
                            .get("button")
                            .map(Value::as_str)
                            .unwrap_or(Some("left"))
                        {
                            Some("left") => 272,
                            Some("right") => 273,
                            Some("middle") => 274,
                            _ => bail!("unsupported button"),
                        };
                        let count = args
                            .get("click_count")
                            .map_or(Ok(1), |_| integer(args, "click_count"))?;
                        ensure!(
                            (1..=3).contains(&count),
                            "click_count must be between 1 and 3"
                        );
                        for _ in 0..count {
                            events.extend([
                                AgentEvent::Button {
                                    button,
                                    pressed: true,
                                },
                                AgentEvent::Button {
                                    button,
                                    pressed: false,
                                },
                            ]);
                        }
                    } else if name == "scroll" {
                        let vertical = number(args, "vertical")?;
                        let horizontal = args
                            .get("horizontal")
                            .map_or(Ok(0.0), |_| number(args, "horizontal"))?;
                        ensure!(
                            vertical.abs() <= 10000.0 && horizontal.abs() <= 10000.0,
                            "scroll delta is too large"
                        );
                        events.push(AgentEvent::Axis {
                            vertical,
                            horizontal,
                        });
                    }
                }
                events.extend(release);
                self.input(id, events)?;
                json!({"delivered":true})
            }
            "press_key" | "type_text" => {
                let window = target(args)?;
                let id = window.id();
                let events = if name == "type_text" {
                    let text = string(args, "text")?.to_owned();
                    ensure!(text.len() <= 64 * 1024, "text exceeds 64 KiB");
                    let app_id = window.app_id().unwrap_or_default().to_lowercase();
                    let chromium = [
                        "chrome", "chromium", "helium", "brave", "vivaldi", "edge", "electron",
                    ]
                    .iter()
                    .any(|name| app_id.contains(name));
                    let unicode_input = match args.get("unicode_input").map(Value::as_str) {
                        Some(Some("keymap")) => AgentUnicodeInput::Keymap,
                        Some(Some("ctrl_shift_u")) => AgentUnicodeInput::CtrlShiftU,
                        None if !chromium => AgentUnicodeInput::Keymap,
                        None => AgentUnicodeInput::CtrlShiftU,
                        _ => bail!("unsupported Unicode input method"),
                    };
                    vec![AgentEvent::Text {
                        text,
                        unicode_input,
                    }]
                } else {
                    shortcut(string(args, "key")?)?
                };
                self.input(id, events)?;
                json!({"delivered":true})
            }
            #[cfg(feature = "dbus")]
            "perform_action" | "set_value" => {
                let window = target(args)?;
                request(Request::AgentInput {
                    geometry: None,
                    workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
                    window_id: window.id(),
                    events: vec![],
                })?;
                self.accessibility.act(&window, name, args)?;
                json!({"performed":true})
            }
            _ => bail!("unknown tool: {name}"),
        };
        if matches!(name, "get_app_state" | "screenshot") {
            request(Request::AgentInput {
                geometry: None,
                workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
                window_id: target_id(args)?,
                events: vec![],
            })?;
        }
        if let Some((id, geometry)) = pending_capture {
            self.captures.insert(id, geometry);
        }
        if !matches!(name, "get_session_state" | "request_handoff")
            && workspace_status()?.is_some_and(|w| w.paused)
        {
            self.invalidate();
            bail!(
                "Workspace automation paused during this request. Check get_session_state before continuing."
            );
        }
        content.push(json!({"type":"text","text":serde_json::to_string(&data)?}));
        Ok(json!({"content":content,"structuredContent":data,"isError":false}))
    }
}

fn modifiers(args: &Value) -> Result<Vec<AgentEvent>> {
    let Some(value) = args.get("modifiers") else {
        return Ok(Vec::new());
    };
    let names: Vec<String> = serde_json::from_value(value.clone())?;
    ensure!(
        names
            .iter()
            .all(|s| matches!(s.as_str(), "Ctrl" | "Shift" | "Alt" | "Super")),
        "unsupported modifier"
    );
    if names.is_empty() {
        Ok(Vec::new())
    } else {
        shortcut(&names.join("+"))
    }
}

fn shortcut(shortcut: &str) -> Result<Vec<AgentEvent>> {
    let mut keys = Vec::new();
    for part in shortcut.split('+') {
        let name = part.to_ascii_lowercase();
        let key = match name.as_str() {
            "ctrl" | "control" => 29,
            "shift" => 42,
            "alt" => 56,
            "super" | "meta" | "win" => 125,
            "return" | "enter" => 28,
            "escape" | "esc" => 1,
            "tab" => 15,
            "space" => 57,
            "backspace" => 14,
            "delete" => 111,
            "insert" => 110,
            "home" => 102,
            "end" => 107,
            "pageup" => 104,
            "pagedown" => 109,
            "left" => 105,
            "right" => 106,
            "up" => 103,
            "down" => 108,
            "minus" => 12,
            "equal" | "plus" => 13,
            _ => {
                let mut found = None;
                for (row, offset) in [
                    ("1234567890", 2),
                    ("qwertyuiop", 16),
                    ("asdfghjkl", 30),
                    ("zxcvbnm", 44),
                ] {
                    if name.len() == 1 {
                        found = found.or_else(|| row.find(&name).map(|i| i as u32 + offset));
                    }
                }
                if let Some(function) = name.strip_prefix('f').and_then(|s| s.parse::<u32>().ok()) {
                    found = match function {
                        1..=10 => Some(58 + function),
                        11..=12 => Some(76 + function),
                        _ => None,
                    };
                }
                found.with_context(|| format!("unsupported key: {part}"))?
            }
        };
        ensure!(
            !keys.contains(&key) && keys.len() < 8,
            "duplicate key or chord too large"
        );
        keys.push(key);
    }
    let mut events: Vec<_> = keys
        .iter()
        .map(|keycode| AgentEvent::Key {
            keycode: *keycode,
            pressed: true,
        })
        .collect();
    events.extend(keys.into_iter().rev().map(|keycode| AgentEvent::Key {
        keycode,
        pressed: false,
    }));
    Ok(events)
}

/// Run a newline-delimited JSON-RPC MCP server until stdin closes.
fn serve(mut input: impl BufRead + Send + 'static, automatic_workspace: bool) -> Result<()> {
    let control = Arc::new(Mutex::new(Control::default()));
    CONTROL.with_borrow_mut(|value| *value = Some(control.clone()));
    let (send, receive) = mpsc::channel();
    let reader_control = control.clone();
    std::thread::spawn(move || {
        loop {
            let mut line = Vec::new();
            let result = (&mut input)
                .take(MAX_REQUEST + 1)
                .read_until(b'\n', &mut line);
            match result {
                Ok(0) | Err(_) => break,
                Ok(_) if line.len() as u64 > MAX_REQUEST => {
                    let _ = send.send(Err(anyhow::anyhow!("MCP request exceeds 1 MiB")));
                    break;
                }
                Ok(_) => (),
            }
            if let Ok(message) = serde_json::from_slice::<Value>(&line) {
                let mut control = reader_control.lock().unwrap();
                if message["method"] == "notifications/cancelled" {
                    let id = message["params"]["requestId"].to_string();
                    if let Some(cancelled) = control.pending.get_mut(&id) {
                        *cancelled = true;
                        if control.active.as_ref() == Some(&id) {
                            if let Some(socket) = &control.shutdown {
                                let _ = socket.shutdown(Shutdown::Both);
                            }
                        }
                    }
                    continue;
                }
                if let Some(id) = message.get("id") {
                    control.pending.insert(id.to_string(), false);
                }
            }
            if send.send(Ok(line)).is_err() {
                break;
            }
        }
        let mut control = reader_control.lock().unwrap();
        control.closed = true;
        if let Some(socket) = &control.shutdown {
            let _ = socket.shutdown(Shutdown::Both);
        }
    });
    let specs = tools();
    if !automatic_workspace {
        request(Request::Windows).context("the selected compositor does not support agent IPC")?;
    }
    let explicit_workspace = std::env::var("NIRI_MCP_WORKSPACE").ok();
    let mut servers: HashMap<String, Server> = HashMap::new();
    if let Some(connection) = CONNECTION.with_borrow_mut(Option::take) {
        servers.insert(
            explicit_workspace.clone().unwrap_or_default(),
            Server {
                connection: Some(connection),
                ..Server::default()
            },
        );
    }
    control.lock().unwrap().shutdown = None;
    let mut output = io::stdout().lock();
    for line in receive {
        let line = line?;
        let parsed = serde_json::from_slice::<Value>(&line);
        let active = parsed
            .as_ref()
            .ok()
            .and_then(|message| message.get("id"))
            .map(Value::to_string);
        control.lock().unwrap().active = active.clone();
        let mut active_server = None;
        let mut response = match parsed {
            Err(_) => {
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Invalid JSON"}})
            }
            Ok(message) => {
                let id = message.get("id").cloned().unwrap_or(Value::Null);
                if message.get("jsonrpc") != Some(&json!("2.0"))
                    || !message.get("method").is_some_and(Value::is_string)
                    || !(id.is_null() || id.is_string() || id.is_number())
                {
                    json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid request"}})
                } else if message.get("id").is_none() {
                    continue;
                } else {
                    let method = message["method"].as_str().unwrap();
                    let params = &message["params"];
                    let result = match method {
                        "initialize" => {
                            let requested = params["protocolVersion"].as_str().unwrap_or("");
                            let protocol =
                                if ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
                                    .contains(&requested)
                                {
                                    requested
                                } else {
                                    "2025-11-25"
                                };
                            Ok(
                                json!({"protocolVersion":protocol,"capabilities":{"tools":{}},"serverInfo":{"name":"niri","version":env!("CARGO_PKG_VERSION")},"instructions":"Start with list_windows or list_layers, then get_app_state for an explicit window_id or layer_id. Use exactly one target selector. Layer accessibility matches PID and namespace against an accessible window name, when available. Pointer coordinates use its latest returned screenshot. Input and captures preserve real focus and never fall back to global input. Workspace mode exposes only its own windows. Open browser windows with --new-window to reuse existing sign-ins. Use request_handoff when human help is needed and tell the user what to do on the host and workspace returned by get_session_state. Stay paused until the user explicitly returns control in chat, then call resume_session. If handoff_for_other_session is true, wait for that thread instead of asking the user to resume this thread. Inspect the app again before continuing. Launch apps through launch_app. Accessibility actions require snapshot_id and element_index from get_app_state. Every mutating tool invalidates the accessibility snapshot. Browser accessibility may require --force-renderer-accessibility."}),
                            )
                        }
                        "ping" => Ok(json!({})),
                        "tools/list" => Ok(json!({"tools":specs})),
                        "tools/call" => {
                            let name = params["name"].as_str().unwrap_or("");
                            let empty = json!({});
                            let args = params.get("arguments").unwrap_or(&empty);
                            let spec = specs.iter().find(|s| s["name"] == name);
                            let validation: Result<()> = (|| {
                                let spec = spec.context("unknown tool")?;
                                let args =
                                    args.as_object().context("arguments must be an object")?;
                                for required in spec["inputSchema"]["required"].as_array().unwrap()
                                {
                                    ensure!(
                                        args.contains_key(required.as_str().unwrap()),
                                        "missing {}",
                                        required.as_str().unwrap()
                                    );
                                }
                                for key in args.keys() {
                                    ensure!(
                                        spec["inputSchema"]["properties"].get(key).is_some(),
                                        "unknown argument: {key}"
                                    );
                                }
                                Ok(())
                            })();
                            match validation {
                                Err(error) => Err((-32602, format!("{error:#}"))),
                                Ok(()) => {
                                    let result = (|| {
                                        let workspace = if automatic_workspace {
                                            Some(thread_workspace(params)?)
                                        } else {
                                            explicit_workspace.clone()
                                        };
                                        if let Some(workspace) = &workspace {
                                            std::env::set_var("NIRI_MCP_WORKSPACE", workspace);
                                        }
                                        let key = workspace.unwrap_or_default();
                                        active_server = Some(key.clone());
                                        if !servers.contains_key(&key)
                                            && servers.len() >= MAX_CONTEXTS
                                        {
                                            let oldest = servers.iter()
                                                .filter(|(_, server)| !server.handoff || server.connection.is_none())
                                                .min_by_key(|(_, server)| server.last_used)
                                                .map(|(key, _)| key.clone())
                                                .context("Too many pending human handoffs. Finish one before opening another thread workspace.")?;
                                            servers.remove(&oldest);
                                        }
                                        let server = servers.entry(key).or_default();
                                        server.last_used = Some(Instant::now());
                                        server.call_with_connection(name, args)
                                    })();
                                    Ok(result.unwrap_or_else(|error: anyhow::Error| json!({"content":[{"type":"text","text":format!("{error:#}")}],"isError":true})))
                                }
                            }
                        }
                        _ => Err((-32601, "Unknown method".to_owned())),
                    };
                    match result {
                        Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
                        Err((code, message)) => {
                            json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
                        }
                    }
                }
            }
        };
        let was_cancelled = {
            let mut control = control.lock().unwrap();
            let cancelled = control.closed
                || active
                    .as_ref()
                    .is_some_and(|id| control.pending.get(id) == Some(&true));
            control.active = None;
            if let Some(id) = active {
                control.pending.remove(&id);
            }
            if cancelled {
                control.shutdown = None;
            }
            cancelled
        };
        if was_cancelled {
            response = json!({"jsonrpc":"2.0","id":response["id"],"error":{"code":-32800,"message":"Request cancelled"}});
            CONNECTION.with_borrow_mut(|value| *value = None);
            if let Some(server) = active_server.and_then(|key| servers.get_mut(&key)) {
                server.connection = None;
            }
        }
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    for server in servers.values_mut() {
        server.invalidate();
    }
    CONNECTION.with_borrow_mut(|value| *value = None);
    control.lock().unwrap().shutdown = None;
    Ok(())
}
