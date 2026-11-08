//! Per-client "virtual focus" input for computer-use agents.
//!
//! Wayland input objects are per client: every client has its own `wl_pointer` and `wl_keyboard`
//! and only sees events sent to them. This module sends enter/motion/button/key events straight to
//! the target client's objects without touching the seat's pointer location, cursor, keyboard
//! focus or window activation. Other clients (including the one the user is working in) receive
//! nothing, so the user's pointer and focus stay where they are.
//!
//! Protocol events carry no surface after `enter`, so a client cannot distinguish two focus
//! streams. When the requested window holds real keyboard focus, agent keys share that stream.
//! Keyboard input is refused if another window of the same client has real focus. A real pointer
//! hovering the client is held back while the agent pointer drives it, and re-enters afterwards.
//! Human and agent input in the same window can mix. Separate clients avoid this interference.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use niri_config::{Color, CornerRadius, GradientInterpolation};
use niri_ipc::{AgentActivityKind, AgentEvent, AgentScreenshotInfo, AgentUnicodeInput};
use smithay::desktop::{
    find_popup_root_surface, layer_map_for_output, LayerSurface, PopupKind, PopupManager, Window,
    WindowSurfaceType,
};
use smithay::input::keyboard::{xkb, Keycode, KeymapFile};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::RegistrationToken;
use smithay::reexports::wayland_server::backend::{ClientId, ObjectId};
use smithay::reexports::wayland_server::protocol::wl_keyboard::{KeyState, KeymapFormat};
use smithay::reexports::wayland_server::protocol::wl_pointer::{Axis, ButtonState};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, DisplayHandle, Resource};
use smithay::utils::{IsAlive, Logical, Point, Rectangle, Serial, Size, SERIAL_COUNTER};
use smithay::wayland::compositor::{
    with_surface_tree_downward, SurfaceAttributes, TraversalAction,
};
use smithay::wayland::socket::ListeningSocketSource;
use tracing::warn;

use crate::layout::workspace::WorkspaceId;
use crate::niri::{ClientState, NewClient, Niri, State};
use crate::render_helpers::border::BorderRenderElement;
use crate::utils::get_monotonic_time;

#[derive(Debug, Default)]
pub struct AgentInputState {
    /// Idle input can be replaced by another session, but held input is exclusive.
    owner: Option<u64>,
    workspaces: HashMap<String, AgentWorkspaceState>,
    desktop_sessions: HashSet<String>,
    controls: HashMap<String, AgentControlState>,
    desktop_owners: HashMap<u64, String>,
    launch_sockets: Vec<(Arc<AgentWorkspaceLaunch>, RegistrationToken)>,
    indicator_visible: bool,
    /// Surface the agent pointer has entered.
    pointer: Option<WlSurface>,
    /// Surface the agent keyboard has entered, and its pressed evdev keycodes.
    keyboard: Option<(WlSurface, Vec<u32>)>,
    /// The target client holds the real keyboard focus: `keyboard` is the surface the real focus
    /// entered, and agent keys go there without agent enter/leave events.
    keyboard_shared: bool,
    /// Recently issued agent serials. Clients reuse input serials to request xdg-activation;
    /// those requests must not move the user's focus.
    serials: VecDeque<(Serial, Serial)>,
    /// Popups opened by agent input. They are mapped without a seat grab.
    popups: Vec<PopupKind>,
    /// When each client last received agent input.
    clients: HashMap<ClientId, Instant>,
    /// Real pointer `last_enter` serial at the time of the agent pointer enter. If it changes,
    /// the real pointer may have entered and left the agent's client, so enter again.
    pointer_real_enter: Option<Serial>,
    /// Real keyboard focus touched the agent keyboard's client; enter again before keys.
    keyboard_stale: bool,
    /// Client of the real keyboard focus.
    real_keyboard_client: Option<ClientId>,
    /// Screenshots waiting for their window to finish drawing.
    shots: Vec<PendingShot>,
    next_shot: u64,
    /// Windows (by id) that receive paced frame callbacks until the given time, even when not
    /// visible, so agent input and screenshots see up-to-date content.
    pumped: HashMap<u64, Instant>,
    pump_running: bool,
    /// Windows (by id) that recently received agent input or screenshots, and when. They match
    /// the `is-agent-driven` window rule.
    windows: HashMap<u64, Instant>,
    /// Client whose real pointer input is held back while the agent pointer drives it, and
    /// until when. The real pointer does not focus its surfaces meanwhile.
    pointer_hold: Option<(ClientId, Instant)>,
    /// Pointer buttons the agent holds.
    buttons: Vec<u32>,
    /// Agent pointer indicators by window id, shown while the window is agent-driven.
    cursors: HashMap<u64, AgentCursor>,
    expiry_running: bool,
}

#[derive(Debug)]
struct AgentWorkspaceState {
    name: String,
    id: u64,
    connections: HashSet<u64>,
}

#[derive(Debug, Default)]
struct AgentControlState {
    paused: bool,
    reason: Option<String>,
    handoff_owner: Option<u64>,
}

#[derive(Debug)]
pub struct AgentWorkspaceLaunch {
    pub(crate) name: String,
    pub(crate) workspace_id: WorkspaceId,
    pub(crate) existing: HashSet<ObjectId>,
    pub(crate) created: Instant,
}

enum AgentSurface {
    Window(Window, WlSurface),
    Layer(LayerSurface),
}

impl AgentSurface {
    fn root(&self) -> &WlSurface {
        match self {
            Self::Window(_, root) => root,
            Self::Layer(layer) => layer.wl_surface(),
        }
    }

    fn geometry_origin(&self) -> Point<i32, Logical> {
        match self {
            Self::Window(window, _) => window.geometry().loc,
            Self::Layer(_) => Point::from((0, 0)),
        }
    }

    fn contains(&self, candidate: &WlSurface) -> bool {
        let mut found = false;
        match self {
            Self::Window(window, _) => window.with_surfaces(|s, _| found |= s == candidate),
            Self::Layer(layer) => layer.with_surfaces(|s, _| found |= s == candidate),
        }
        found
    }

    fn surface_under(
        &self,
        point: Point<f64, Logical>,
    ) -> Option<(WlSurface, Point<i32, Logical>)> {
        match self {
            Self::Window(window, _) => window.surface_under(point, WindowSurfaceType::ALL),
            Self::Layer(layer) => layer.surface_under(point, WindowSurfaceType::ALL),
        }
    }
}

fn agent_window_ancestors(surface: &WlSurface) -> Vec<WlSurface> {
    let mut ancestors = Vec::new();
    let mut current = Some(surface.clone());
    while let Some(surface) = current {
        if ancestors.contains(&surface) {
            break;
        }
        current = smithay::wayland::compositor::with_states(&surface, |states| {
            states
                .data_map
                .get::<smithay::wayland::shell::xdg::XdgToplevelSurfaceData>()
                .and_then(|role| role.lock().unwrap().parent.clone())
        });
        ancestors.push(surface);
    }
    ancestors
}

impl Niri {
    pub fn check_desktop_session_access(&self, owner: u64) -> Result<(), String> {
        if let Some(name) = self.agent_input.desktop_owners.get(&owner) {
            if !self.agent_input.desktop_sessions.contains(name) {
                return Err("desktop session is finished".into());
            }
            if self.agent_paused(name) {
                return Err("desktop automation is paused".into());
            }
        }
        Ok(())
    }

    fn agent_paused(&self, name: &str) -> bool {
        self.agent_input
            .controls
            .get(name)
            .is_some_and(|c| c.paused)
    }

    fn agent_handoff_for_other_session(&self, name: &str, owner: Option<u64>) -> bool {
        self.agent_input
            .controls
            .get(name)
            .and_then(|c| c.handoff_owner)
            .is_some_and(|other| {
                owner != Some(other)
                    && (self.agent_input.desktop_owners.contains_key(&other)
                        || self
                            .agent_input
                            .workspaces
                            .values()
                            .any(|w| w.connections.contains(&other)))
            })
    }

    fn desktop_session_status(
        &self,
        name: &str,
        owner: u64,
    ) -> Option<niri_ipc::AgentDesktopSession> {
        self.agent_input.desktop_sessions.contains(name).then(|| {
            let control = &self.agent_input.controls[name];
            niri_ipc::AgentDesktopSession {
                name: name.to_owned(),
                paused: control.paused,
                reason: control.reason.clone(),
                handoff_for_other_session: self.agent_handoff_for_other_session(name, Some(owner)),
            }
        })
    }

    fn agent_handoff_pending(&self, entry: &AgentWorkspaceState) -> bool {
        let id = WorkspaceId::specific(entry.id);
        self.agent_paused(&entry.name)
            && (self.agent_input.desktop_sessions.contains(&entry.name)
                || self
                    .layout
                    .find_workspace_by_id(id)
                    .is_some_and(|(_, workspace)| workspace.windows().next().is_some())
                || self.unmapped_windows.keys().any(|surface| {
                    self.agent_launch_for_surface(surface)
                        .is_some_and(|launch| launch.workspace_id == id)
                }))
    }

    pub fn agent_cleanup_needed(&self, name: &str) -> bool {
        (self.agent_input.desktop_sessions.contains(name) && !self.agent_paused(name))
            || self
                .agent_input
                .workspaces
                .get(&name.to_ascii_lowercase())
                .is_some_and(|entry| !self.agent_handoff_pending(entry))
    }

    pub fn keep_agent_window(&mut self, window_id: u64, scope: Option<&str>) -> Result<(), String> {
        self.check_agent_workspace_access(window_id, scope)?;
        let surface = self
            .layout
            .windows()
            .find(|(_, w)| w.id().get() == window_id)
            .map(|(_, w)| w.toplevel().wl_surface().clone())
            .ok_or("window no longer exists")?;
        self.set_agent_window_retained(&surface);
        Ok(())
    }

    pub fn keep_agent_surface(&mut self, surface: &WlSurface) {
        let root = self.find_root_shell_surface(surface);
        if self.agent_launch_for_surface(&root).is_some_and(|launch| {
            self.agent_input
                .workspaces
                .get(&launch.name)
                .is_some_and(|entry| !self.agent_paused(&entry.name))
        }) {
            self.set_agent_window_retained(&root);
        }
    }

    fn set_agent_window_retained(&mut self, surface: &WlSurface) {
        for surface in agent_window_ancestors(surface) {
            if let Some((window, _)) = self.layout.find_window_and_output_mut(&surface) {
                window.is_agent_retained = true;
            }
        }
    }

    pub fn finish_agent_workspace(&mut self, name: &str) -> Result<(), String> {
        let name = name.to_ascii_lowercase();
        let Some(entry) = self.agent_input.workspaces.get(&name) else {
            return Ok(());
        };
        if self.agent_handoff_pending(entry) {
            return Ok(());
        }
        let id = WorkspaceId::specific(entry.id);
        if let Some((_, workspace)) = self.layout.find_workspace_by_id(id) {
            for window in workspace.windows() {
                let surface = window.toplevel().wl_surface();
                if !agent_window_ancestors(surface).iter().any(|surface| {
                    self.layout
                        .find_window_and_output(surface)
                        .is_some_and(|(w, _)| w.is_agent_retained)
                }) && self
                    .agent_launch_for_surface(surface)
                    .is_some_and(|launch| {
                        launch.workspace_id == id && !launch.existing.contains(&surface.id())
                    })
                {
                    window.toplevel().send_close();
                }
            }
        }
        self.agent_input.workspaces.remove(&name);
        self.layout.release_agent_workspace(id);
        let configured = self
            .layout
            .find_workspace_by_id(id)
            .is_some_and(|(_, workspace)| {
                workspace.name().is_some_and(|name| {
                    self.config
                        .borrow()
                        .workspaces
                        .iter()
                        .any(|w| w.name.0.eq_ignore_ascii_case(name))
                })
            });
        if !configured {
            self.layout.unname_workspace_by_id(id);
        }
        self.queue_redraw_all();
        Ok(())
    }

    pub fn agent_workspace_changed(
        &mut self,
        name: String,
        connected: bool,
        owner: u64,
    ) -> Result<niri_ipc::AgentWorkspace, String> {
        let name = name.to_ascii_lowercase();
        if connected {
            if name.trim().is_empty() || name.chars().count() > 64 {
                return Err("workspace name must contain 1 to 64 characters".into());
            }
            if let Some(entry) = self.agent_input.workspaces.get_mut(&name) {
                if self
                    .layout
                    .find_workspace_by_id(WorkspaceId::specific(entry.id))
                    .is_some()
                {
                    entry.connections.insert(owner);
                    return self.agent_workspace_status(&name, Some(owner));
                }
            }
            self.layout
                .ensure_named_workspace_at_bottom(&niri_config::Workspace {
                    name: niri_config::workspace::WorkspaceName(name.clone()),
                    open_on_output: Vec::new(),
                    layout: None,
                });
            let id = self
                .layout
                .find_workspace_by_name(&name)
                .unwrap()
                .1
                .id()
                .get();
            self.agent_input.controls.entry(name.clone()).or_default();
            let entry = self
                .agent_input
                .workspaces
                .entry(name.clone())
                .or_insert_with(|| AgentWorkspaceState {
                    name: name.clone(),
                    id,
                    connections: HashSet::new(),
                });
            entry.id = id;
            entry.connections.insert(owner);
            self.agent_workspace_status(&name, Some(owner))
        } else {
            let entry = self
                .agent_input
                .workspaces
                .get_mut(&name)
                .ok_or("agent workspace is not reserved")?;
            entry.connections.remove(&owner);
            self.agent_workspace_status(&name, Some(owner))
        }
    }

    pub fn agent_workspace_status(
        &self,
        name: &str,
        owner: Option<u64>,
    ) -> Result<niri_ipc::AgentWorkspace, String> {
        let entry = self
            .agent_input
            .workspaces
            .get(&name.to_ascii_lowercase())
            .ok_or("agent workspace is not reserved")?;
        let (index, workspace) = self
            .layout
            .find_workspace_by_id(WorkspaceId::specific(entry.id))
            .ok_or("agent workspace no longer exists, reconnect the session")?;
        let control = &self.agent_input.controls[&entry.name];
        let info = niri_ipc::AgentWorkspace {
            name: workspace
                .name()
                .cloned()
                .unwrap_or_else(|| (index + 1).to_string()),
            id: entry.id,
            paused: control.paused,
            reason: control.reason.clone(),
            handoff_for_other_session: owner.is_some_and(|owner| {
                self.agent_handoff_for_other_session(&entry.name, Some(owner))
            }),
        };
        Ok(info)
    }

    pub(crate) fn check_agent_workspace_access(
        &self,
        window_id: u64,
        scope: Option<&str>,
    ) -> Result<(), String> {
        let workspace = self
            .layout
            .workspaces()
            .find(|(_, _, w)| w.windows().any(|w| w.id().get() == window_id))
            .map(|(_, _, w)| w.id().get());
        if let Some(scope) = scope {
            if Some(self.agent_workspace_status(scope, None)?.id) != workspace {
                return Err("window is outside the agent workspace".into());
            }
        }
        if self
            .agent_input
            .workspaces
            .values()
            .any(|w| Some(w.id) == workspace && self.agent_paused(&w.name))
        {
            return Err("workspace automation is paused".into());
        }
        Ok(())
    }

    pub fn agent_workspace_for_client(&self, surface: &WlSurface) -> Option<WorkspaceId> {
        self.agent_launch_for_surface(surface)
            .map(|launch| launch.workspace_id)
            .filter(|id| {
                self.agent_input
                    .workspaces
                    .values()
                    .any(|entry| entry.id == id.get())
            })
    }

    #[allow(
        clippy::mutable_key_type,
        reason = "ObjectId hashes identity, independent of its liveness flag"
    )]
    pub fn agent_existing_surfaces(&self) -> HashSet<ObjectId> {
        // Surface identity survives unmapping, unlike the layout's window ID.
        self.unmapped_windows
            .keys()
            .map(Resource::id)
            .chain(
                self.layout
                    .windows()
                    .map(|(_, w)| w.toplevel().wl_surface().id()),
            )
            .collect()
    }

    pub fn agent_launch_display(
        &mut self,
        launch: Arc<AgentWorkspaceLaunch>,
    ) -> Result<OsString, String> {
        let source =
            ListeningSocketSource::with_name(&format!("niri-agent-{:016x}", fastrand::u64(..)))
                .map_err(|e| e.to_string())?;
        let name = source.socket_name().to_os_string();
        let weak = Arc::downgrade(&launch);
        let token = self
            .event_loop
            .insert_source(source, move |client, _, state| {
                let Some(launch) = weak.upgrade() else {
                    return;
                };
                state.niri.insert_client(NewClient {
                    client,
                    restricted: false,
                    credentials_unknown: false,
                    agent_launch: Some(launch),
                });
            })
            .map_err(|e| e.to_string())?;
        self.agent_input.launch_sockets.push((launch, token));
        Ok(name)
    }

    pub fn agent_launch_for_surface(
        &self,
        surface: &WlSurface,
    ) -> Option<Arc<AgentWorkspaceLaunch>> {
        let mut ancestor = Some(surface.clone());
        let mut seen = Vec::new();
        while let Some(candidate) = ancestor {
            if seen.contains(&candidate) {
                break;
            }
            seen.push(candidate.clone());
            if let Some(launch) = smithay::wayland::compositor::with_states(&candidate, |states| {
                states
                    .data_map
                    .get::<Mutex<Arc<AgentWorkspaceLaunch>>>()
                    .map(|launch| launch.lock().unwrap().clone())
            }) {
                return Some(launch);
            }
            ancestor = smithay::wayland::compositor::with_states(&candidate, |states| {
                states
                    .data_map
                    .get::<smithay::wayland::shell::xdg::XdgToplevelSurfaceData>()
                    .and_then(|role| role.lock().unwrap().parent.clone())
            });
        }
        surface
            .client()?
            .get_data::<ClientState>()?
            .agent_launch
            .clone()
    }

    fn agent_surface(&self, id: u64) -> Option<AgentSurface> {
        self.layout
            .windows()
            .find(|(_, m)| m.id().get() == id)
            .map(|(_, m)| AgentSurface::Window(m.window.clone(), m.toplevel().wl_surface().clone()))
            .or_else(|| {
                self.mapped_layer_surfaces
                    .values()
                    .find(|m| m.id().get() == id)
                    .map(|m| AgentSurface::Layer(m.surface().clone()))
            })
    }
}

#[derive(Debug)]
struct PendingShot {
    owner: u64,
    workspace: Option<String>,
    id: u64,
    root: WlSurface,
    window_id: u64,
    path: String,
    show_pointer: bool,
    done: async_channel::Sender<Result<AgentScreenshotInfo, String>>,
    started: Instant,
    last_commit: Instant,
}

/// A client counts as agent-driven for this long after its last agent input.
const AGENT_DRIVEN_TIMEOUT: Duration = Duration::from_secs(30);

/// The real pointer is held back from the agent's client for this long after agent pointer
/// input (or as long as the agent holds a button, up to `AGENT_DRIVEN_TIMEOUT`).
const AGENT_POINTER_HOLD: Duration = Duration::from_millis(600);

/// Frame callbacks are paced for windows touched by the agent for this long.
const AGENT_PUMP_DURATION: Duration = Duration::from_secs(2);
/// Frame callback interval of the pump (about 60 Hz).
const AGENT_PUMP_INTERVAL: Duration = Duration::from_millis(16);
/// A screenshot is taken once the window has not committed for this long...
const AGENT_SHOT_QUIET: Duration = Duration::from_millis(50);
/// ...or after this long at most.
const AGENT_SHOT_MAX_WAIT: Duration = Duration::from_millis(500);

/// Agent pointer indicator animations.
const CURSOR_APPEAR: Duration = Duration::from_millis(160);
const CURSOR_GLIDE: Duration = Duration::from_millis(180);
const CURSOR_RIPPLE: Duration = Duration::from_millis(520);

/// Agent pointer indicator colors: an indigo-to-violet gradient.
const CURSOR_FROM: [u8; 3] = [0x81, 0x8c, 0xf8];
const CURSOR_TO: [u8; 3] = [0xc0, 0x84, 0xfc];

/// The agent's pointer as drawn on its window: a gradient dot with a white outline and a soft
/// halo. It glides to new positions, and a ring ripples out when a button is pressed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentCursor {
    shown_at: Instant,
    /// Glide from `from` to `to` (relative to the window geometry), started at `moved_at`.
    from: Point<f64, Logical>,
    to: Point<f64, Logical>,
    moved_at: Instant,
    pressed: bool,
    ripple: Option<(Point<f64, Logical>, Instant)>,
}

fn ease_out_cubic(t: f64) -> f64 {
    1. - (1. - t).powi(3)
}

fn progress(since: Instant, now: Instant, duration: Duration) -> f64 {
    (now.saturating_duration_since(since).as_secs_f64() / duration.as_secs_f64()).clamp(0., 1.)
}

fn cursor_color([r, g, b]: [u8; 3], alpha: f32) -> Color {
    Color::from_rgba8_unpremul(r, g, b, (alpha * 255.) as u8)
}

impl AgentCursor {
    fn new(pos: Point<f64, Logical>, now: Instant) -> Self {
        Self {
            shown_at: now,
            from: pos,
            to: pos,
            moved_at: now,
            pressed: false,
            ripple: None,
        }
    }

    fn position(&self, now: Instant) -> Point<f64, Logical> {
        let t = ease_out_cubic(progress(self.moved_at, now, CURSOR_GLIDE));
        self.from + (self.to - self.from).upscale(t)
    }

    fn move_to(&mut self, pos: Point<f64, Logical>, now: Instant) {
        self.from = self.position(now);
        self.to = pos;
        self.moved_at = now;
    }

    fn set_pressed(&mut self, pressed: bool, now: Instant) {
        if pressed && !self.pressed {
            self.ripple = Some((self.to, now));
        }
        self.pressed = pressed;
    }

    pub fn is_animating(&self, now: Instant) -> bool {
        progress(self.shown_at, now, CURSOR_APPEAR) < 1.
            || progress(self.moved_at, now, CURSOR_GLIDE) < 1.
            || self
                .ripple
                .is_some_and(|(_, at)| progress(at, now, CURSOR_RIPPLE) < 1.)
    }

    /// Renders the indicator for a window whose visual geometry is at `origin`.
    pub fn render(
        &self,
        origin: Point<f64, Logical>,
        scale: f32,
        push: &mut dyn FnMut(BorderRenderElement),
    ) {
        let now = Instant::now();
        let appear = ease_out_cubic(progress(self.shown_at, now, CURSOR_APPEAR));

        // A circle (or a ring when `width > 0`) centered at `center`.
        let circle = |center: Point<f64, Logical>,
                      diameter: f64,
                      from: Color,
                      to: Color,
                      width: f32,
                      alpha: f32| {
            let size = Size::from((diameter, diameter));
            BorderRenderElement::new(
                size,
                Rectangle::from_size(size),
                GradientInterpolation::default(),
                from,
                to,
                45f32.to_radians(),
                Rectangle::from_size(size),
                width,
                CornerRadius::from(diameter as f32 / 2.),
                scale,
                alpha,
            )
            .with_location(origin + center - Point::from((diameter / 2., diameter / 2.)))
        };

        let center = self.position(now);
        let grow = 0.6 + 0.4 * appear;
        let alpha = appear as f32;
        let white = Color::from_rgba8_unpremul(255, 255, 255, 245);
        let shade = Color::from_rgba8_unpremul(0, 0, 0, 70);

        // Topmost first: dot, white outline, dark hairline for light backgrounds.
        let from = cursor_color(CURSOR_FROM, 1.);
        let to = cursor_color(CURSOR_TO, 1.);
        push(circle(center, 10. * grow, from, to, 0., alpha));
        push(circle(center, 14. * grow, white, white, 0., alpha));
        push(circle(center, 16. * grow, shade, shade, 0., alpha));

        if let Some((at, started)) = self.ripple {
            let t = progress(started, now, CURSOR_RIPPLE);
            if t < 1. {
                let e = ease_out_cubic(t);
                let from = cursor_color(CURSOR_FROM, 0.9);
                let to = cursor_color(CURSOR_TO, 0.9);
                let width = (2.5 - 1.5 * e) as f32;
                push(circle(at, 14. + 42. * e, from, to, width, (1. - e) as f32));
            }
        }

        // The halo tightens while a button is held.
        let (halo, halo_alpha) = if self.pressed {
            (24., 0.3)
        } else {
            (34., 0.16)
        };
        let from = cursor_color(CURSOR_FROM, halo_alpha);
        let to = cursor_color(CURSOR_TO, halo_alpha);
        push(circle(center, halo * grow, from, to, 0., alpha));
    }
}

/// Environment variable marking processes launched by an agent. Their windows (and those of
/// their descendants, which inherit the environment) open without taking focus.
const AGENT_LAUNCH_ENV: &[u8] = b"NIRI_AGENT_LAUNCH=1";

/// Whether the client process was started with `NIRI_AGENT_LAUNCH=1`.
pub fn is_agent_launched(client: &Client, dh: &DisplayHandle) -> bool {
    if client
        .get_data::<ClientState>()
        .is_some_and(|data| data.agent_launch.is_some())
    {
        return true;
    }
    let Ok(credentials) = client.get_credentials(dh) else {
        return false;
    };
    let Ok(environ) = std::fs::read(format!("/proc/{}/environ", credentials.pid)) else {
        return false;
    };
    environ
        .split(|b| *b == 0)
        .any(|var| var == AGENT_LAUNCH_ENV)
}

/// Whether a surface tree has frame callbacks queued, i.e. the client is waiting to draw.
fn has_pending_frame_callbacks(root: &WlSurface) -> bool {
    let pending = std::cell::Cell::new(false);
    with_surface_tree_downward(
        root,
        (),
        |_, _, _| TraversalAction::DoChildren(()),
        |_, states, _| {
            let mut attrs = states.cached_state.get::<SurfaceAttributes>();
            if !attrs.current().frame_callbacks.is_empty() {
                pending.set(true);
            }
        },
        |_, _, _| !pending.get(),
    );
    pending.get()
}

impl AgentInputState {
    fn input_is_held(&self) -> bool {
        !self.buttons.is_empty()
            || self
                .keyboard
                .as_ref()
                .is_some_and(|(_, keys)| !keys.is_empty())
    }

    fn next_serial(&mut self) -> u32 {
        let serial = SERIAL_COUNTER.next_serial();
        if let Some((_, end)) = self.serials.back_mut() {
            if u32::from(*end).wrapping_add(1) == u32::from(serial) {
                *end = serial;
                return serial.into();
            }
        }
        self.serials.push_back((serial, serial));
        serial.into()
    }

    /// Whether `serial` was sent to a client by agent input.
    pub fn issued(&self, serial: Serial) -> bool {
        self.serials
            .iter()
            .any(|(start, end)| serial.is_no_older_than(start) && end.is_no_older_than(&serial))
    }

    /// Whether the real pointer must not focus surfaces of `client` right now.
    pub fn holds_pointer(&self, client: &ClientId) -> bool {
        self.pointer_hold
            .as_ref()
            .is_some_and(|(c, until)| c == client && *until > Instant::now())
    }

    /// Whether `client` received agent input recently.
    pub fn is_agent_driven(&self, client: &Client) -> bool {
        self.clients
            .get(&client.id())
            .is_some_and(|t| t.elapsed() < AGENT_DRIVEN_TIMEOUT)
    }

    /// Called on every real keyboard focus change.
    pub fn real_keyboard_focus_changed(&mut self, new: Option<ClientId>) {
        let old = std::mem::replace(&mut self.real_keyboard_client, new.clone());
        let agent_client = self
            .keyboard
            .as_ref()
            .and_then(|(s, _)| s.client())
            .map(|c| c.id());
        if agent_client.is_some() && (agent_client == old || agent_client == new) {
            self.keyboard_stale = true;
        }
    }

    /// Remember a popup whose grab request carried an agent serial.
    pub fn add_popup(&mut self, popup: PopupKind) {
        self.popups.retain(|p| p.alive());
        self.popups.push(popup);
    }

    fn is_popup(&self, surface: &WlSurface) -> bool {
        self.popups.iter().any(|p| p.wl_surface() == surface)
    }

    /// Topmost live agent popup of the window with toplevel `root`.
    fn top_popup(&self, root: &WlSurface) -> Option<WlSurface> {
        self.popups
            .iter()
            .rev()
            .filter(|p| p.alive())
            .find(|p| find_popup_root_surface(p).ok().as_ref() == Some(root))
            .map(|p| p.wl_surface().clone())
    }

    /// Take all agent popups of the window with toplevel `root`, topmost first.
    fn take_popups(&mut self, root: &WlSurface) -> Vec<PopupKind> {
        let (taken, kept): (Vec<_>, Vec<_>) = self
            .popups
            .drain(..)
            .partition(|p| p.alive() && find_popup_root_surface(p).ok().as_ref() == Some(root));
        self.popups = kept.into_iter().filter(|p| p.alive()).collect();
        taken.into_iter().rev().collect()
    }
}

const KEY_LEFTCTRL: u32 = 29;
const KEY_LEFTSHIFT: u32 = 42;
const KEY_U: u32 = 22;
const KEY_SPACE: u32 = 57;

fn now_ms() -> u32 {
    get_monotonic_time().as_millis() as u32
}

/// The most significant kind of activity in an agent input request.
fn activity_kind(events: &[AgentEvent]) -> Option<AgentActivityKind> {
    events
        .iter()
        .map(|e| match e {
            AgentEvent::Key { .. } | AgentEvent::Text { .. } | AgentEvent::KeyboardLeave => {
                AgentActivityKind::Keyboard
            }
            AgentEvent::Button { .. } => AgentActivityKind::Click,
            AgentEvent::Axis { .. } => AgentActivityKind::Scroll,
            AgentEvent::Motion { .. } | AgentEvent::PointerLeave => AgentActivityKind::Pointer,
        })
        .max_by_key(|kind| match kind {
            AgentActivityKind::Keyboard => 3,
            AgentActivityKind::Click => 2,
            AgentActivityKind::Scroll => 1,
            _ => 0,
        })
}

pub(crate) fn is_agent_cleanup(events: &[AgentEvent]) -> bool {
    !events.is_empty()
        && events
            .iter()
            .all(|event| matches!(event, AgentEvent::KeyboardLeave | AgentEvent::PointerLeave))
}

fn same_client(surface: &WlSurface, client: &Client) -> bool {
    surface.client().is_some_and(|c| c.id() == client.id())
}

impl State {
    pub fn control_desktop_session(
        &mut self,
        name: String,
        action: niri_ipc::AgentDesktopAction,
        owner: u64,
    ) -> Result<Option<niri_ipc::AgentDesktopSession>, String> {
        use niri_ipc::AgentDesktopAction;
        if name.is_empty() || name.len() > 64 || name != name.to_ascii_lowercase() {
            return Err("invalid desktop session identity".into());
        }
        match action {
            AgentDesktopAction::Status => {
                self.niri
                    .agent_input
                    .controls
                    .entry(name.clone())
                    .or_default();
                self.niri.agent_input.desktop_sessions.insert(name.clone());
                self.niri
                    .agent_input
                    .desktop_owners
                    .insert(owner, name.clone());
            }
            AgentDesktopAction::Finish => {
                self.finish_agent_session(&name)?;
            }
            action => {
                let reason = match action {
                    AgentDesktopAction::Handoff { reason } => Some(reason),
                    AgentDesktopAction::Resume => None,
                    _ => unreachable!(),
                };
                if !self.niri.agent_input.desktop_sessions.contains(&name) {
                    return Err("desktop session is not open".into());
                }
                if self.niri.agent_input.workspaces.contains_key(&name) {
                    self.set_agent_workspace_control(&name, reason.is_some(), reason, Some(owner))?;
                } else {
                    self.set_agent_control(&name, reason.is_some(), reason, Some(owner))?;
                }
            }
        }
        self.refresh_agent_workspace_control();
        Ok(self.niri.desktop_session_status(&name, owner))
    }

    fn set_agent_control(
        &mut self,
        name: &str,
        paused: bool,
        reason: Option<String>,
        owner: Option<u64>,
    ) -> Result<(), String> {
        if reason
            .as_ref()
            .is_some_and(|reason| reason.trim().is_empty() || reason.chars().count() > 240)
        {
            return Err("handoff reason must contain 1 to 240 characters".into());
        }
        if self.niri.agent_handoff_for_other_session(name, owner) {
            return Err("handoff belongs to another agent session".into());
        }
        let control = self
            .niri
            .agent_input
            .controls
            .get_mut(name)
            .ok_or("agent session is not open")?;
        control.handoff_owner = if paused && reason.is_some() {
            owner
        } else {
            None
        };
        control.paused = paused;
        control.reason = reason;
        if paused {
            self.cancel_desktop_requests(name);
        }
        Ok(())
    }

    fn cancel_desktop_requests(&mut self, name: &str) {
        let owners: Vec<_> = self
            .niri
            .agent_input
            .desktop_owners
            .iter()
            .filter(|(_, session)| session.as_str() == name)
            .map(|(owner, _)| *owner)
            .collect();
        for owner in owners {
            self.agent_cancel_requests(owner);
        }
    }

    pub fn finish_agent_session(&mut self, name: &str) -> Result<bool, String> {
        let name = name.to_ascii_lowercase();
        if self.niri.agent_input.desktop_sessions.contains(&name) && self.niri.agent_paused(&name) {
            return Ok(false);
        }
        if !self.finish_agent_workspace(&name)? {
            return Ok(false);
        }
        self.cancel_desktop_requests(&name);
        self.niri.agent_input.desktop_sessions.remove(&name);
        self.niri.agent_input.controls.remove(&name);
        self.refresh_agent_workspace_control();
        Ok(true)
    }

    pub fn retire_empty_agent_handoffs(&mut self) {
        let names: Vec<_> = self
            .niri
            .agent_input
            .workspaces
            .iter()
            .filter(|(_, entry)| {
                self.niri.agent_paused(&entry.name)
                    && entry.connections.is_empty()
                    && !self.niri.agent_handoff_pending(entry)
            })
            .map(|(name, _)| name.clone())
            .collect();
        for name in names {
            let _ = self.finish_agent_workspace(&name);
        }
    }

    pub fn finish_agent_workspace(&mut self, name: &str) -> Result<bool, String> {
        let Some(entry) = self
            .niri
            .agent_input
            .workspaces
            .get(&name.to_ascii_lowercase())
        else {
            return Ok(true);
        };
        if self.niri.agent_handoff_pending(entry) {
            return Ok(false);
        }
        let owners: Vec<_> = entry.connections.iter().copied().collect();
        for owner in owners {
            self.agent_disconnect(owner);
        }
        self.niri.finish_agent_workspace(name).map(|()| true)
    }

    pub fn set_agent_workspace_control(
        &mut self,
        name: &str,
        paused: bool,
        reason: Option<String>,
        owner: Option<u64>,
    ) -> Result<niri_ipc::AgentWorkspace, String> {
        let name = name.to_ascii_lowercase();
        if !self.niri.agent_input.workspaces.contains_key(&name) {
            return Err("agent workspace is not reserved".into());
        }
        self.set_agent_control(&name, paused, reason, owner)?;
        let info = self.niri.agent_workspace_status(&name, owner)?;
        let windows: Vec<_> = self
            .niri
            .layout
            .find_workspace_by_id(WorkspaceId::specific(info.id))
            .map(|(_, workspace)| {
                workspace
                    .windows()
                    .map(|w| (w.id().get(), w.toplevel().wl_surface().clone()))
                    .collect()
            })
            .unwrap_or_default();
        if paused {
            let contains = |surface: &WlSurface| {
                windows
                    .iter()
                    .any(|(_, root)| *root == self.niri.find_root_shell_surface(surface))
            };
            let pointer = self
                .niri
                .agent_input
                .pointer
                .as_ref()
                .filter(|s| contains(s))
                .and_then(WlSurface::client);
            let release_keyboard = self
                .niri
                .agent_input
                .keyboard
                .as_ref()
                .is_some_and(|(s, _)| contains(s));
            let keep_keyboard = self
                .niri
                .agent_input
                .keyboard
                .as_ref()
                .filter(|_| !release_keyboard)
                .map(|(s, _)| s.clone());
            if let Some(client) = pointer {
                self.agent_release_other_target(&client, |surface| {
                    keep_keyboard.as_ref() == Some(surface)
                });
            }
            if release_keyboard {
                self.agent_release_keyboard();
            }
            self.niri
                .agent_input
                .cursors
                .retain(|id, _| !windows.iter().any(|(window_id, _)| id == window_id));
            let (canceled, pending): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.niri.agent_input.shots)
                    .into_iter()
                    .partition(|shot| windows.iter().any(|(id, _)| *id == shot.window_id));
            self.niri.agent_input.shots = pending;
            for shot in canceled {
                let _ = shot
                    .done
                    .send_blocking(Err("workspace automation is paused".into()));
            }
        }
        for (_, surface) in windows {
            if let Some((window, _)) = self.niri.layout.find_window_and_output_mut(&surface) {
                window.set_urgent(info.reason.is_some());
            }
        }
        self.niri.queue_redraw_all();
        self.niri.agent_workspace_status(&name, owner)
    }

    pub fn refresh_agent_workspace_control(&mut self) {
        let visible = self
            .niri
            .agent_input
            .desktop_sessions
            .iter()
            .any(|name| !self.niri.agent_paused(name))
            || self
                .niri
                .layout
                .workspaces()
                .any(|(monitor, index, workspace)| {
                    monitor.is_some_and(|m| m.active_workspace_idx() == index)
                        && self.niri.agent_input.workspaces.values().any(|w| {
                            w.id == workspace.id().get() && !self.niri.agent_paused(&w.name)
                        })
                });
        if visible != self.niri.agent_input.indicator_visible {
            self.niri.agent_input.indicator_visible = visible;
            self.niri.agent_indicator.session_changed(visible);
            self.niri.queue_redraw_all();
        }
    }

    pub fn agent_release_input(&mut self, owner: u64) {
        if self.niri.agent_input.owner == Some(owner) {
            if let Some(client) = self
                .niri
                .agent_input
                .pointer
                .as_ref()
                .and_then(WlSurface::client)
            {
                self.agent_release_other_target(&client, |_| false);
            }
            self.agent_release_keyboard();
            self.niri.agent_input.pointer = None;
            self.niri.agent_input.buttons.clear();
            self.niri.agent_input.owner = None;
            self.niri.agent_input.pointer_hold = None;
            self.update_pointer_contents();
            self.niri.queue_redraw_all();
        }
    }

    pub fn agent_disconnect(&mut self, owner: u64) {
        self.niri.agent_input.desktop_owners.remove(&owner);
        self.agent_cancel_requests(owner);
    }

    fn agent_cancel_requests(&mut self, owner: u64) {
        self.agent_release_input(owner);
        self.niri.agent_input.shots.retain(|shot| {
            if shot.owner != owner {
                return true;
            }
            let _ = shot.done.send_blocking(Err("agent disconnected".into()));
            false
        });
    }

    pub fn agent_input(
        &mut self,
        owner: u64,
        window_id: u64,
        events: Vec<AgentEvent>,
    ) -> Result<(), String> {
        if !is_agent_cleanup(&events) {
            self.niri.check_desktop_session_access(owner)?;
        }
        if events.is_empty() {
            self.niri.check_agent_workspace_access(window_id, None)?;
            if self.niri.is_locked() {
                return Err("session is locked".into());
            }
            self.niri
                .agent_surface(window_id)
                .ok_or_else(|| format!("no window with id {window_id}"))?;
            return Ok(());
        }
        if self
            .niri
            .agent_input
            .owner
            .is_some_and(|current| current != owner)
            && self.niri.agent_input.input_is_held()
        {
            return Err("agent input is busy with another session's gesture".into());
        }
        self.niri.agent_input.owner = Some(owner);
        let result = self.agent_deliver_input(window_id, events);
        if result.is_err() {
            self.agent_release_input(owner);
        }
        result
    }

    fn agent_deliver_input(
        &mut self,
        window_id: u64,
        events: Vec<AgentEvent>,
    ) -> Result<(), String> {
        let target = self
            .niri
            .agent_surface(window_id)
            .ok_or_else(|| format!("no window with id {window_id}"))?;
        let toplevel = target.root().clone();
        if !is_agent_cleanup(&events) {
            self.niri.check_agent_workspace_access(window_id, None)?;
            if self.niri.is_locked() {
                return Err("session is locked".into());
            }
        }
        if events.iter().any(|event| {
            matches!(event, AgentEvent::Key { keycode, .. } if keycode.checked_add(8).is_none())
        }) {
            return Err(String::from("keycode exceeds the XKB keycode range"));
        }

        let client = toplevel
            .client()
            .ok_or_else(|| String::from("window has no client"))?;

        let pointer = self.niri.seat.get_pointer().unwrap();
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let real_pointer_on_client = pointer
            .current_focus()
            .is_some_and(|s| same_client(&s, &client));
        let keyboard_shared = keyboard
            .current_focus()
            .is_some_and(|s| same_client(&s, &client));
        let real_focus_on_target = keyboard
            .current_focus()
            .is_some_and(|s| target.contains(&s));
        if !real_focus_on_target && !events.is_empty() {
            if let AgentSurface::Layer(layer) = &target {
                if let Some(mapped) = self.niri.mapped_layer_surfaces.get_mut(layer) {
                    mapped.agent_focus_suppressed = true;
                }
            }
        }
        let belongs_to_window = |candidate: &WlSurface| target.contains(candidate);
        if keyboard_shared
            && !real_focus_on_target
            && events
                .iter()
                .any(|event| matches!(event, AgentEvent::Key { .. } | AgentEvent::Text { .. }))
        {
            return Err(String::from(
                "another window of this client has the real keyboard focus",
            ));
        }
        self.agent_mark_window(window_id);

        self.agent_release_other_target(&client, belongs_to_window);

        let state = &mut self.niri.agent_input;
        // Older serials cannot authorize activation once both real focus enters have passed
        // them. Keep their provenance too when the user explicitly enables invalid serials.
        if !self
            .niri
            .config
            .borrow()
            .debug
            .honor_xdg_activation_with_invalid_serial
        {
            let keyboard_enter = keyboard.last_enter();
            let pointer_enter = pointer.last_enter();
            state.serials.retain(|(_, end)| {
                keyboard_enter.is_some_and(|enter| end.is_no_older_than(&enter))
                    || pointer_enter.is_some_and(|enter| end.is_no_older_than(&enter))
            });
        }
        state
            .clients
            .retain(|_, t| t.elapsed() < AGENT_DRIVEN_TIMEOUT);
        state.clients.insert(client.id(), Instant::now());
        if state.keyboard_shared != keyboard_shared {
            // Real focus entered or left the client since; its own enter/leave superseded ours.
            state.keyboard = None;
            state.keyboard_shared = keyboard_shared;
        }

        // Virtual input can leave a synthetic keymap active on the shared seat. Agent keycodes
        // refer to the configured keyboard, so restore it before lookup and modifier calculation,
        // then synchronize the target client (including after our per-client temporary keymaps).
        if events
            .iter()
            .any(|e| matches!(e, AgentEvent::Key { .. } | AgentEvent::Text { .. }))
        {
            self.restore_configured_keymap();
            self.agent_send_seat_keymap(&client);
        }

        let kind = activity_kind(&events);

        // Protocol events carry no surface after enter, so a client cannot tell two pointers
        // apart. Hold the real pointer back from the client while the agent pointer drives it:
        // the real pointer leaves the client's surfaces now, and re-enters them once the hold
        // expires.
        let pointer_events = events.iter().any(|e| {
            matches!(
                e,
                AgentEvent::Motion { .. }
                    | AgentEvent::Button { .. }
                    | AgentEvent::Axis { .. }
                    | AgentEvent::PointerLeave
            )
        });
        if pointer_events {
            // During a real pointer grab (the user dragging in the client) the real pointer stays
            // on the client; the agent's pointer events then mix with the user's.
            self.agent_hold_pointer(&client);
        }

        let mut pointer_stale = self.niri.agent_input.pointer.is_some()
            && (real_pointer_on_client
                || self.niri.agent_input.pointer_real_enter != pointer.last_enter());

        let geometry_loc = target.geometry_origin().to_f64();

        for event in events {
            match event {
                AgentEvent::Motion { x, y } => {
                    let point = Point::<f64, Logical>::from((x, y)) + geometry_loc;
                    let (surface, loc) = target
                        .surface_under(point)
                        .ok_or_else(|| format!("no surface at ({x}, {y})"))?;
                    let local = point - loc.to_f64();

                    let entered = self.niri.agent_input.pointer.clone();
                    let reenter = pointer_stale || entered.as_ref() != Some(&surface);
                    pointer_stale = false;
                    for ptr in pointer.client_pointers(&client) {
                        if reenter {
                            if let Some(old) = entered.as_ref().filter(|s| s.is_alive()) {
                                ptr.leave(self.niri.agent_input.next_serial(), old);
                            }
                            ptr.enter(
                                self.niri.agent_input.next_serial(),
                                &surface,
                                local.x,
                                local.y,
                            );
                        }
                        ptr.motion(now_ms(), local.x, local.y);
                        if ptr.version() >= 5 {
                            ptr.frame();
                        }
                    }
                    self.niri.agent_input.pointer = Some(surface);
                    self.niri.agent_input.pointer_real_enter = pointer.last_enter();
                    let (pos, now) = (Point::from((x, y)), Instant::now());
                    self.niri
                        .agent_input
                        .cursors
                        .entry(window_id)
                        .and_modify(|c| c.move_to(pos, now))
                        .or_insert_with(|| AgentCursor::new(pos, now));
                }
                AgentEvent::Button { button, pressed } => {
                    if self.niri.agent_input.pointer.is_none() {
                        return Err(String::from("Button before Motion"));
                    }
                    let on_popup = self
                        .niri
                        .agent_input
                        .pointer
                        .as_ref()
                        .is_some_and(|s| self.niri.agent_input.is_popup(s));
                    if pressed && !on_popup {
                        for popup in self.niri.agent_input.take_popups(&toplevel) {
                            let _ = PopupManager::dismiss_popup(&toplevel, &popup);
                        }
                    }

                    if let Some(cursor) = self.niri.agent_input.cursors.get_mut(&window_id) {
                        cursor.set_pressed(pressed, Instant::now());
                    }
                    let buttons = &mut self.niri.agent_input.buttons;
                    buttons.retain(|b| *b != button);
                    if pressed {
                        buttons.push(button);
                    }

                    let state = if pressed {
                        ButtonState::Pressed
                    } else {
                        ButtonState::Released
                    };
                    for ptr in pointer.client_pointers(&client) {
                        ptr.button(self.niri.agent_input.next_serial(), now_ms(), button, state);
                        if ptr.version() >= 5 {
                            ptr.frame();
                        }
                    }
                }
                AgentEvent::Axis {
                    vertical,
                    horizontal,
                } => {
                    if self.niri.agent_input.pointer.is_none() {
                        return Err(String::from("Axis before Motion"));
                    }
                    for ptr in pointer.client_pointers(&client) {
                        if vertical != 0. {
                            ptr.axis(now_ms(), Axis::VerticalScroll, vertical);
                        }
                        if horizontal != 0. {
                            ptr.axis(now_ms(), Axis::HorizontalScroll, horizontal);
                        }
                        if ptr.version() >= 5 {
                            ptr.frame();
                        }
                    }
                }
                AgentEvent::PointerLeave => {
                    self.niri.agent_input.cursors.remove(&window_id);
                    let buttons = std::mem::take(&mut self.niri.agent_input.buttons);
                    let surface = self.niri.agent_input.pointer.take();
                    for ptr in pointer.client_pointers(&client) {
                        for button in &buttons {
                            ptr.button(
                                self.niri.agent_input.next_serial(),
                                now_ms(),
                                *button,
                                ButtonState::Released,
                            );
                        }
                        if let Some(surface) = &surface {
                            ptr.leave(self.niri.agent_input.next_serial(), surface);
                        }
                        if ptr.version() >= 5 {
                            ptr.frame();
                        }
                    }
                }
                AgentEvent::Key { keycode, pressed } => {
                    self.agent_key(&client, &toplevel, keycode, pressed);
                }
                AgentEvent::Text {
                    text,
                    unicode_input,
                } => {
                    self.agent_text(&client, &toplevel, &text, unicode_input)?;
                }
                AgentEvent::KeyboardLeave => self.agent_release_keyboard(),
            }
        }

        // Give a shared keyboard the user's modifiers back once the agent released its keys.
        if self.niri.agent_input.keyboard_shared
            && self
                .niri
                .agent_input
                .keyboard
                .as_ref()
                .is_some_and(|(_, keys)| keys.is_empty())
        {
            let mods = keyboard.modifier_state().serialized;
            for kbd in keyboard.client_keyboards(&client) {
                kbd.modifiers(
                    self.niri.agent_input.next_serial(),
                    mods.depressed,
                    mods.latched,
                    mods.locked,
                    mods.layout_effective,
                );
            }
        }

        // Let the window draw the result promptly even if it is off-screen.
        self.agent_pump_window(window_id);

        if pointer_events {
            // Extend the hold from the end of the input (and while a button is held).
            self.agent_hold_pointer(&client);
        }

        if let Some(kind) = kind {
            self.ipc_agent_input_activity(window_id, kind);
        }

        Ok(())
    }

    /// Release synthetic input before switching windows, including within one client.
    fn agent_release_other_target(
        &mut self,
        client: &Client,
        belongs_to_window: impl Fn(&WlSurface) -> bool,
    ) {
        let pointer = self.niri.seat.get_pointer().unwrap();
        let state = &mut self.niri.agent_input;
        if state
            .pointer
            .as_ref()
            .is_some_and(|s| !s.is_alive() || !belongs_to_window(s))
        {
            if let Some(surface) = state.pointer.take() {
                if let Some(old_client) = surface.client() {
                    let buttons = std::mem::take(&mut state.buttons);
                    for ptr in pointer.client_pointers(&old_client) {
                        for button in &buttons {
                            ptr.button(
                                state.next_serial(),
                                now_ms(),
                                *button,
                                ButtonState::Released,
                            );
                        }
                        if surface.is_alive() {
                            ptr.leave(state.next_serial(), &surface);
                        }
                        if ptr.version() >= 5 {
                            ptr.frame();
                        }
                    }
                }
            }
            state.buttons.clear();
            state.pointer_real_enter = None;
            state.pointer_hold = None;
        }
        if state
            .pointer_hold
            .as_ref()
            .is_some_and(|(old, _)| *old != client.id())
        {
            state.pointer_hold = None;
        }
        if state
            .keyboard
            .as_ref()
            .is_some_and(|(s, _)| !s.is_alive() || !belongs_to_window(s))
        {
            self.agent_release_keyboard();
        }
    }

    fn agent_release_keyboard(&mut self) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let state = &mut self.niri.agent_input;
        if let Some((surface, keys)) = state.keyboard.take() {
            if let Some(old_client) = surface.client() {
                let shared = keyboard
                    .current_focus()
                    .is_some_and(|s| same_client(&s, &old_client));
                let mods = keyboard.modifier_state().serialized;
                for kbd in keyboard.client_keyboards(&old_client) {
                    for key in &keys {
                        kbd.key(state.next_serial(), now_ms(), *key, KeyState::Released);
                    }
                    if shared {
                        kbd.modifiers(
                            state.next_serial(),
                            mods.depressed,
                            mods.latched,
                            mods.locked,
                            mods.layout_effective,
                        );
                    } else if surface.is_alive() {
                        kbd.leave(state.next_serial(), &surface);
                    }
                }
            }
        }
    }

    /// Hold the real pointer back from `client` (see `AgentInputState::pointer_hold`).
    fn agent_hold_pointer(&mut self, client: &Client) {
        let state = &mut self.niri.agent_input;
        let duration = if state.buttons.is_empty() {
            AGENT_POINTER_HOLD
        } else {
            AGENT_DRIVEN_TIMEOUT
        };
        let until = Instant::now() + duration;
        state.pointer_hold = Some((client.id(), until));

        // Move the real pointer focus off the client's surfaces right away. Unlike
        // refresh_pointer_contents(), update_pointer_contents() also works during transitions.
        let pointer = self.niri.seat.get_pointer().unwrap();
        if self.update_pointer_contents() {
            pointer.frame(self);
            self.niri.queue_redraw_all();
        }

        let result =
            self.niri
                .event_loop
                .insert_source(Timer::from_deadline(until), |_, _, state| {
                    state.agent_pointer_hold_expired();
                    TimeoutAction::Drop
                });
        if let Err(err) = result {
            warn!("error scheduling agent pointer hold expiry: {err}");
        }
    }

    /// Release an expired pointer hold. refresh() then gives the real pointer its focus back.
    fn agent_pointer_hold_expired(&mut self) {
        let state = &mut self.niri.agent_input;
        let Some((client, until)) = state.pointer_hold.clone() else {
            return;
        };
        if until > Instant::now() {
            // Extended since this timer was scheduled.
            return;
        }
        state.pointer_hold = None;

        // If the real pointer is about to enter the client, leave the agent's surface first so
        // the client does not get two enters in a row.
        let pointer = self.niri.seat.get_pointer().unwrap();
        let under = self.niri.contents_under(pointer.current_location());
        let real_enters = under
            .surface
            .as_ref()
            .and_then(|(s, _)| s.client())
            .is_some_and(|c| c.id() == client);
        let agent_surface = self
            .niri
            .agent_input
            .pointer
            .take_if(|s| s.client().is_some_and(|c| c.id() == client));
        if let (true, Some(surface)) = (real_enters, &agent_surface) {
            if let Some(client) = surface.client() {
                for ptr in pointer.client_pointers(&client) {
                    ptr.leave(self.niri.agent_input.next_serial(), surface);
                    if ptr.version() >= 5 {
                        ptr.frame();
                    }
                }
            }
        } else {
            // Still entered.
            self.niri.agent_input.pointer = agent_surface;
        }
    }

    /// Enter the agent keyboard on the topmost agent popup of the window, or the toplevel, if it
    /// is not there already (or real focus touched the client since).
    fn agent_keyboard_enter(&mut self, client: &Client, toplevel: &WlSurface) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        if self.niri.agent_input.keyboard_shared {
            // The real focus has entered the client already; type where it is.
            let state = &mut self.niri.agent_input;
            if state.keyboard.is_none() {
                let focus = keyboard.current_focus().unwrap_or_else(|| toplevel.clone());
                state.keyboard = Some((focus, Vec::new()));
            }
            return;
        }
        // Like a popup grab: keys go to the topmost agent popup if there is one.
        let target = self
            .niri
            .agent_input
            .top_popup(toplevel)
            .unwrap_or_else(|| toplevel.clone());
        let entered = self
            .niri
            .agent_input
            .keyboard
            .as_ref()
            .map(|(s, _)| s.clone());
        if !self.niri.agent_input.keyboard_stale && entered.as_ref() == Some(&target) {
            return;
        }
        self.niri.agent_input.keyboard_stale = false;
        for kbd in keyboard.client_keyboards(client) {
            if let Some(old) = entered.as_ref().filter(|s| s.is_alive()) {
                kbd.leave(self.niri.agent_input.next_serial(), old);
            }
            kbd.enter(self.niri.agent_input.next_serial(), &target, Vec::new());
        }
        let pressed_keys = self
            .niri
            .agent_input
            .keyboard
            .take()
            .map(|(_, keys)| keys)
            .unwrap_or_default();
        self.niri.agent_input.keyboard = Some((target, pressed_keys.clone()));
        self.agent_send_modifiers(client, &pressed_keys);
    }

    /// Press or release an evdev key for the agent keyboard.
    fn agent_key(&mut self, client: &Client, toplevel: &WlSurface, keycode: u32, pressed: bool) {
        self.agent_keyboard_enter(client, toplevel);
        let keyboard = self.niri.seat.get_keyboard().unwrap();

        let (_, pressed_keys) = self.niri.agent_input.keyboard.as_mut().unwrap();
        if pressed {
            if !pressed_keys.contains(&keycode) {
                pressed_keys.push(keycode);
            }
        } else {
            pressed_keys.retain(|k| *k != keycode);
        }
        let pressed_keys = pressed_keys.clone();

        let state = if pressed {
            KeyState::Pressed
        } else {
            KeyState::Released
        };
        for kbd in keyboard.client_keyboards(client) {
            kbd.key(
                self.niri.agent_input.next_serial(),
                now_ms(),
                keycode,
                state,
            );
        }
        self.agent_send_modifiers(client, &pressed_keys);
    }

    fn agent_tap(&mut self, client: &Client, toplevel: &WlSurface, keycode: u32, shift: bool) {
        if shift {
            self.agent_key(client, toplevel, KEY_LEFTSHIFT, true);
        }
        self.agent_key(client, toplevel, keycode, true);
        self.agent_key(client, toplevel, keycode, false);
        if shift {
            self.agent_key(client, toplevel, KEY_LEFTSHIFT, false);
        }
    }

    /// Find an evdev keycode and whether Shift is needed to type `ch` with the seat keymap.
    fn agent_lookup_char(&mut self, ch: char) -> Option<(u32, bool)> {
        let keysym = match ch {
            '\n' | '\r' => xkb::Keysym::Return,
            '\t' => xkb::Keysym::Tab,
            '\u{8}' => xkb::Keysym::BackSpace,
            _ => xkb::utf32_to_keysym(ch as u32),
        };
        if keysym == xkb::Keysym::NoSymbol {
            return None;
        }
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        keyboard.with_xkb_state(self, |ctx| {
            let xkb = ctx.xkb().lock().unwrap();
            // SAFETY: the references do not outlive the lock.
            let keymap = unsafe { xkb.keymap() };
            let layout = unsafe { xkb.state() }.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE);
            let shift_mask = 1 << keymap.mod_get_index(xkb::MOD_NAME_SHIFT);
            let mut found = None;
            keymap.key_for_each(|keymap, code| {
                if found.is_some() || code.raw() < 8 {
                    return;
                }
                for level in 0..keymap.num_levels_for_key(code, layout).min(2) {
                    if !keymap
                        .key_get_syms_by_level(code, layout, level)
                        .contains(&keysym)
                    {
                        continue;
                    }
                    // Check that the level is reached with no modifiers or with Shift alone.
                    let mut masks = [0; 4];
                    let n = keymap.key_get_mods_for_level(code, layout, level, &mut masks);
                    let wanted = if level == 0 { 0 } else { shift_mask };
                    if masks[..n].contains(&wanted) || (level == 0 && n == 0) {
                        found = Some((code.raw() - 8, level == 1));
                        return;
                    }
                }
            });
            found
        })
    }

    /// Type text into the agent's target.
    fn agent_text(
        &mut self,
        client: &Client,
        toplevel: &WlSurface,
        text: &str,
        unicode_input: AgentUnicodeInput,
    ) -> Result<(), String> {
        let mut missing = String::new();
        for ch in text.chars() {
            if let Some((keycode, shift)) = self.agent_lookup_char(ch) {
                self.agent_type_missing(client, toplevel, &missing, unicode_input)?;
                missing.clear();
                self.agent_tap(client, toplevel, keycode, shift);
            } else {
                missing.push(ch);
            }
        }
        self.agent_type_missing(client, toplevel, &missing, unicode_input)
    }

    /// Type characters that the seat keymap lacks.
    fn agent_type_missing(
        &mut self,
        client: &Client,
        toplevel: &WlSurface,
        text: &str,
        unicode_input: AgentUnicodeInput,
    ) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        match unicode_input {
            AgentUnicodeInput::CtrlShiftU => {
                for ch in text.chars() {
                    self.agent_key(client, toplevel, KEY_LEFTCTRL, true);
                    self.agent_key(client, toplevel, KEY_LEFTSHIFT, true);
                    self.agent_tap(client, toplevel, KEY_U, false);
                    self.agent_key(client, toplevel, KEY_LEFTSHIFT, false);
                    self.agent_key(client, toplevel, KEY_LEFTCTRL, false);
                    for digit in format!("{:x}", ch as u32).chars() {
                        let (keycode, shift) = self
                            .agent_lookup_char(digit)
                            .ok_or_else(|| format!("seat keymap cannot type {digit:?}"))?;
                        self.agent_tap(client, toplevel, keycode, shift);
                    }
                    self.agent_tap(client, toplevel, KEY_SPACE, false);
                }
                Ok(())
            }
            AgentUnicodeInput::Keymap => {
                // A keymap with one key per distinct character, sent to this client only. The
                // client's modifiers are reset while it is active; the seat keymap is restored
                // right after, before any real input can reach the client.
                let chars: Vec<char> = text.chars().collect();
                for chunk in chars.chunks(200) {
                    let mut keymap_chars = chunk.to_vec();
                    keymap_chars.sort_unstable();
                    keymap_chars.dedup();
                    let keymap_file = self.agent_temp_keymap(&keymap_chars)?;
                    self.agent_keyboard_enter(client, toplevel);
                    let keyboard = self.niri.seat.get_keyboard().unwrap();
                    for kbd in keyboard.client_keyboards(client) {
                        let res = keymap_file.with_fd(kbd.version() >= 7, |fd, size| {
                            kbd.keymap(KeymapFormat::XkbV1, fd, size as u32)
                        });
                        if let Err(err) = res {
                            warn!("error sending agent temporary keymap: {err:?}");
                        }
                        kbd.modifiers(self.niri.agent_input.next_serial(), 0, 0, 0, 0);
                    }
                    for ch in chunk {
                        // Wire keycode = xkb keycode - 8; the temp keymap starts at xkb 9.
                        let keycode = keymap_chars.iter().position(|c| c == ch).unwrap() as u32 + 1;
                        for kbd in keyboard.client_keyboards(client) {
                            let serial = self.niri.agent_input.next_serial();
                            kbd.key(serial, now_ms(), keycode, KeyState::Pressed);
                            let serial = self.niri.agent_input.next_serial();
                            kbd.key(serial, now_ms(), keycode, KeyState::Released);
                        }
                    }
                    self.agent_send_seat_keymap(client);
                }
                Ok(())
            }
        }
    }

    fn agent_temp_keymap(&mut self, chars: &[char]) -> Result<KeymapFile, String> {
        let mut keycodes = String::new();
        let mut symbols = String::new();
        for (i, ch) in chars.iter().enumerate() {
            keycodes.push_str(&format!("<K{i}> = {};\n", i + 9));
            symbols.push_str(&format!("key <K{i}> {{ [ U{:04X} ] }};\n", *ch as u32));
        }
        let text = format!(
            "xkb_keymap {{\n\
             xkb_keycodes \"agent\" {{ minimum = 8; maximum = {}; {keycodes} }};\n\
             xkb_types \"agent\" {{ include \"complete\" }};\n\
             xkb_compatibility \"agent\" {{ include \"complete\" }};\n\
             xkb_symbols \"agent\" {{ {symbols} }};\n\
             }};\n",
            chars.len() + 9
        );
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_string(
            &context,
            text,
            xkb::KEYMAP_FORMAT_TEXT_V1,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .ok_or_else(|| String::from("error compiling temporary agent keymap"))?;
        Ok(KeymapFile::new(&keymap))
    }

    /// Screenshot a window without the clipboard and notification side effects of the regular
    /// screenshot action. `done` receives the result.
    ///
    /// Windows that are not visible only get frame callbacks about once per second, and an
    /// app may need several frames to show the result of input (key -> pty -> echo -> redraw).
    /// Agent-touched windows are therefore pumped with frame callbacks at ~60 Hz, and the
    /// screenshot is taken once the window stops committing (or after a maximum wait).
    pub fn agent_screenshot(
        &mut self,
        owner: u64,
        workspace: Option<String>,
        window_id: u64,
        path: String,
        show_pointer: bool,
        done: async_channel::Sender<Result<AgentScreenshotInfo, String>>,
    ) -> Result<(), String> {
        self.niri.check_desktop_session_access(owner)?;
        self.niri
            .check_agent_workspace_access(window_id, workspace.as_deref())?;
        if self.niri.is_locked() {
            return Err(String::from("session is locked"));
        }

        let root = self
            .niri
            .agent_surface(window_id)
            .map(|target| target.root().clone())
            .ok_or_else(|| format!("no window with id {window_id} on an output"))?;
        self.agent_mark_window(window_id);
        self.ipc_agent_input_activity(window_id, AgentActivityKind::Screenshot);

        let recently_pumped = self.niri.agent_input.pumped.contains_key(&window_id);
        if !recently_pumped && !has_pending_frame_callbacks(&root) {
            // Idle window: its current buffer is up to date.
            return self.agent_render_screenshot(
                window_id,
                workspace.as_deref(),
                path,
                show_pointer,
                done,
            );
        }

        self.agent_pump_window(window_id);

        let now = Instant::now();
        let state = &mut self.niri.agent_input;
        let id = state.next_shot;
        state.next_shot += 1;
        state.shots.push(PendingShot {
            owner,
            workspace,
            id,
            root,
            window_id,
            path,
            show_pointer,
            done,
            started: now,
            last_commit: now,
        });
        self.niri
            .event_loop
            .insert_source(
                Timer::from_duration(AGENT_SHOT_QUIET),
                move |_, _, state| {
                    if state.agent_try_complete_shot(id) {
                        TimeoutAction::Drop
                    } else {
                        TimeoutAction::ToDuration(AGENT_PUMP_INTERVAL)
                    }
                },
            )
            .map_err(|err| format!("error scheduling agent screenshot: {err}"))?;
        Ok(())
    }

    /// Called for every non-sync commit.
    pub fn agent_root_committed(&mut self, root: &WlSurface) {
        let now = Instant::now();
        for shot in &mut self.niri.agent_input.shots {
            if &shot.root == root {
                shot.last_commit = now;
            }
        }
    }

    /// Take the screenshot if the window has gone quiet or waited long enough. Returns whether
    /// the shot is finished.
    fn agent_try_complete_shot(&mut self, id: u64) -> bool {
        let shots = &mut self.niri.agent_input.shots;
        let Some(index) = shots.iter().position(|s| s.id == id) else {
            return true;
        };
        let shot = &shots[index];
        if shot.last_commit.elapsed() < AGENT_SHOT_QUIET
            && shot.started.elapsed() < AGENT_SHOT_MAX_WAIT
        {
            return false;
        }
        let shot = shots.remove(index);
        let done = shot.done.clone();
        if let Err(err) = self.agent_render_screenshot(
            shot.window_id,
            shot.workspace.as_deref(),
            shot.path,
            shot.show_pointer,
            shot.done,
        ) {
            let _ = done.send_blocking(Err(err));
        }
        true
    }

    /// Mark a window as agent-driven for a while.
    fn agent_mark_window(&mut self, window_id: u64) {
        let state = &mut self.niri.agent_input;
        state.windows.insert(window_id, Instant::now());
        if state.expiry_running {
            return;
        }
        state.expiry_running = true;
        // The timer only wakes up the event loop; refresh() drops expired windows.
        let result = self.niri.event_loop.insert_source(
            Timer::from_duration(AGENT_DRIVEN_TIMEOUT),
            |_, _, state| {
                let now = Instant::now();
                let next = state
                    .niri
                    .agent_input
                    .windows
                    .values()
                    .map(|t| (*t + AGENT_DRIVEN_TIMEOUT).saturating_duration_since(now))
                    .filter(|d| !d.is_zero())
                    .min();
                match next {
                    Some(next) => TimeoutAction::ToDuration(next),
                    None => {
                        state.niri.agent_input.expiry_running = false;
                        TimeoutAction::Drop
                    }
                }
            },
        );
        if let Err(err) = result {
            warn!("error scheduling agent window expiry: {err}");
            self.niri.agent_input.expiry_running = false;
        }
    }

    /// Keep sending paced frame callbacks to a window for a while, visible or not.
    pub fn agent_pump_window(&mut self, window_id: u64) {
        let state = &mut self.niri.agent_input;
        state
            .pumped
            .insert(window_id, Instant::now() + AGENT_PUMP_DURATION);
        if state.pump_running {
            return;
        }
        state.pump_running = true;
        self.agent_pump_tick();
        let result = self.niri.event_loop.insert_source(
            Timer::from_duration(AGENT_PUMP_INTERVAL),
            |_, _, state| {
                if state.agent_pump_tick() {
                    TimeoutAction::ToDuration(AGENT_PUMP_INTERVAL)
                } else {
                    state.niri.agent_input.pump_running = false;
                    TimeoutAction::Drop
                }
            },
        );
        if let Err(err) = result {
            warn!("error scheduling agent frame pump: {err}");
            self.niri.agent_input.pump_running = false;
        }
    }

    /// Send frame callbacks to pumped windows. Returns whether any window is still pumped.
    fn agent_pump_tick(&mut self) -> bool {
        let now = Instant::now();
        self.niri.agent_input.pumped.retain(|_, until| *until > now);
        if self.niri.agent_input.pumped.is_empty() {
            return false;
        }
        let pumped = &self.niri.agent_input.pumped;
        let time = get_monotonic_time();
        for (mon, mapped) in self.niri.layout.windows() {
            if !pumped.contains_key(&mapped.id().get()) {
                continue;
            }
            let Some(output) = mon.map(|mon| mon.output()) else {
                continue;
            };
            // The throttle keeps this from doubling up with regular vblank frame callbacks.
            mapped.window.send_frame(
                output,
                time,
                Some(AGENT_PUMP_INTERVAL - Duration::from_millis(1)),
                |_, _| Some(output.clone()),
            );
        }
        for output in self.niri.layout.outputs() {
            for layer in layer_map_for_output(output).layers() {
                if self
                    .niri
                    .mapped_layer_surfaces
                    .get(layer)
                    .is_some_and(|m| pumped.contains_key(&m.id().get()))
                {
                    layer.send_frame(
                        output,
                        time,
                        Some(AGENT_PUMP_INTERVAL - Duration::from_millis(1)),
                        |_, _| Some(output.clone()),
                    );
                }
            }
        }
        true
    }

    /// Render a window and write it as PNG from a worker thread.
    fn agent_render_screenshot(
        &mut self,
        window_id: u64,
        workspace: Option<&str>,
        path: String,
        show_pointer: bool,
        done: async_channel::Sender<Result<AgentScreenshotInfo, String>>,
    ) -> Result<(), String> {
        self.niri
            .check_agent_workspace_access(window_id, workspace)?;
        let (size, (scale, geometry), pixels) =
            self.agent_capture(window_id, show_pointer, true)?;
        std::thread::spawn(move || {
            let result = (|| {
                let mut buf = Vec::new();
                crate::utils::write_png_rgba8(
                    std::io::Cursor::new(&mut buf),
                    size.w as u32,
                    size.h as u32,
                    &pixels,
                )
                .map_err(|err| format!("error encoding PNG: {err:?}"))?;
                std::fs::write(&path, buf).map_err(|err| format!("error writing {path}: {err}"))?;
                Ok(AgentScreenshotInfo {
                    width: size.w as u32,
                    height: size.h as u32,
                    scale,
                    geometry_x: geometry.x,
                    geometry_y: geometry.y,
                })
            })();
            let _ = done.send_blocking(result);
        });
        Ok(())
    }

    pub fn agent_geometry(&mut self, window_id: u64) -> Result<AgentScreenshotInfo, String> {
        let (size, (scale, origin), _) = self.agent_capture(window_id, false, false)?;
        Ok(AgentScreenshotInfo {
            width: size.w as u32,
            height: size.h as u32,
            scale,
            geometry_x: origin.x,
            geometry_y: origin.y,
        })
    }

    fn agent_capture(
        &mut self,
        window_id: u64,
        show_pointer: bool,
        pixels: bool,
    ) -> Result<crate::niri::WindowPixels, String> {
        if self.niri.is_locked() {
            return Err(String::from("session is locked"));
        }

        let niri = &self.niri;
        let window = niri
            .layout
            .windows()
            .find(|(_, m)| m.id().get() == window_id)
            .and_then(|(mon, m)| Some((mon?.output().clone(), m)));
        let layer = niri.layout.outputs().find_map(|output| {
            layer_map_for_output(output).layers().find_map(|surface| {
                let mapped = niri.mapped_layer_surfaces.get(surface)?;
                (mapped.id().get() == window_id).then(|| (output.clone(), mapped))
            })
        });
        if window.is_none() && layer.is_none() {
            return Err(format!(
                "no window or layer with id {window_id} on an output"
            ));
        }

        self.backend
            .with_primary_renderer(|renderer| {
                if let Some((output, mapped)) = &window {
                    niri.render_window_pixels(renderer, output, mapped, show_pointer, pixels)
                } else {
                    let (output, mapped) = layer.as_ref().unwrap();
                    niri.render_layer_pixels(renderer, output, mapped, pixels)
                }
            })
            .ok_or_else(|| String::from("no renderer"))?
            .map_err(|err| format!("error rendering window: {err:?}"))
    }

    fn agent_send_seat_keymap(&mut self, client: &Client) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let keymap_file = keyboard.with_xkb_state(self, |ctx| {
            let xkb = ctx.xkb().lock().unwrap();
            // SAFETY: KeymapFile only serializes the keymap; no ref-count is kept.
            KeymapFile::new(unsafe { xkb.keymap() })
        });
        for kbd in keyboard.client_keyboards(client) {
            let res = keymap_file.with_fd(kbd.version() >= 7, |fd, size| {
                kbd.keymap(KeymapFormat::XkbV1, fd, size as u32)
            });
            if let Err(err) = res {
                warn!("error sending agent keymap: {err:?}");
            }
        }
        // A new keymap resets the client's layout, including on a shared keyboard.
        let pressed = self
            .niri
            .agent_input
            .keyboard
            .as_ref()
            .map(|(_, keys)| keys.clone())
            .unwrap_or_default();
        self.agent_send_modifiers(client, &pressed);
    }

    /// Send modifiers derived from the agent's own pressed keys (not the user's).
    fn agent_send_modifiers(&mut self, client: &Client, pressed: &[u32]) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let (depressed, latched, locked, group) = keyboard.with_xkb_state(self, |ctx| {
            let xkb = ctx.xkb().lock().unwrap();
            // SAFETY: the temporary state is dropped before the lock is released.
            let keymap = unsafe { xkb.keymap() };
            let layout = unsafe { xkb.state() }.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE);
            let mut state = xkb::State::new(keymap);
            // Use the selected layout without inheriting the user's modifiers.
            state.update_mask(0, 0, 0, 0, 0, layout);
            for key in pressed {
                state.update_key(Keycode::new(key + 8), xkb::KeyDirection::Down);
            }
            (
                state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
                state.serialize_mods(xkb::STATE_MODS_LATCHED),
                state.serialize_mods(xkb::STATE_MODS_LOCKED),
                state.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE),
            )
        });
        for kbd in keyboard.client_keyboards(client) {
            kbd.modifiers(
                self.niri.agent_input.next_serial(),
                depressed,
                latched,
                locked,
                group,
            );
        }
    }
}

impl Niri {
    /// Sync agent-driven state and the agent pointer indicator to mapped windows.
    pub fn refresh_mapped_agent_state(&mut self) {
        self.agent_input.launch_sockets.retain(|(launch, token)| {
            // Tokens, clients and windows keep their launch socket alive.
            let keep = launch.created.elapsed() < crate::handlers::XDG_ACTIVATION_TOKEN_TIMEOUT
                || Arc::strong_count(launch) > 1;
            if !keep {
                self.event_loop.remove(*token);
            }
            keep
        });
        let pending_workspaces: HashSet<_> = self
            .unmapped_windows
            .keys()
            .filter_map(|surface| self.agent_launch_for_surface(surface))
            .map(|launch| launch.workspace_id.get())
            .collect();
        let launches = &self.agent_input.launch_sockets;
        let controls = &self.agent_input.controls;
        self.agent_input.workspaces.retain(|name, entry| {
            let keep = controls.get(&entry.name).is_some_and(|c| c.paused)
                || !entry.connections.is_empty()
                || pending_workspaces.contains(&entry.id)
                || launches.iter().any(|(launch, _)| launch.name == *name)
                || self
                    .layout
                    .find_workspace_by_id(WorkspaceId::specific(entry.id))
                    .is_some_and(|(_, workspace)| workspace.windows().next().is_some());
            if !keep {
                let id = WorkspaceId::specific(entry.id);
                let configured =
                    self.layout
                        .find_workspace_by_id(id)
                        .is_some_and(|(_, workspace)| {
                            workspace.name().is_some_and(|name| {
                                self.config
                                    .borrow()
                                    .workspaces
                                    .iter()
                                    .any(|w| w.name.0.eq_ignore_ascii_case(name))
                            })
                        });
                self.layout.release_agent_workspace(id);
                if !configured {
                    self.layout.unname_workspace_by_id(id);
                }
            }
            keep
        });
        let state = &mut self.agent_input;
        state.controls.retain(|name, _| {
            state.workspaces.contains_key(name) || state.desktop_sessions.contains(name)
        });
        state
            .windows
            .retain(|_, t| t.elapsed() < AGENT_DRIVEN_TIMEOUT);
        let windows = &state.windows;
        state.cursors.retain(|id, _| windows.contains_key(id));
        let cursors = &state.cursors;

        let now = Instant::now();
        let mut redraw = Vec::new();
        self.layout.with_windows_mut(|mapped, output| {
            let id = mapped.id().get();
            mapped.set_is_agent_driven(windows.contains_key(&id));
            let cursor = cursors.get(&id).copied();
            let changed = mapped.set_agent_cursor(cursor);
            if changed || cursor.is_some_and(|c| c.is_animating(now)) {
                redraw.extend(output.cloned());
            }
        });
        for output in self.layout.outputs() {
            for layer in layer_map_for_output(output).layers() {
                if let Some(mapped) = self.mapped_layer_surfaces.get_mut(layer) {
                    let cursor = cursors.get(&mapped.id().get()).copied();
                    let changed = mapped.set_agent_cursor(cursor);
                    if changed || cursor.is_some_and(|c| c.is_animating(now)) {
                        redraw.push(output.clone());
                    }
                }
            }
        }
        for output in redraw {
            self.queue_redraw(&output);
        }
    }
}
