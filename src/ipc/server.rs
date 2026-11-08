use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{env, io, process};

use anyhow::Context;
use async_channel::{Receiver, Sender, TrySendError};
use calloop::futures::Scheduler;
use calloop::io::Async;
use directories::BaseDirs;
use futures_util::io::{AsyncReadExt, BufReader};
use futures_util::{select_biased, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, FutureExt as _};
use niri_config::OutputName;
use niri_ipc::state::{EventStreamState, EventStreamStatePart as _};
use niri_ipc::{
    Action, AgentActivityKind, Event, KeyboardLayouts, OutputConfigChanged, Overview, Reply,
    Request, Response, Screenshot, ScreenshotRequest, Timestamp, WindowLayout, Workspace,
};
use smithay::desktop::layer_map_for_output;
use smithay::input::pointer::{
    CursorIcon, CursorImageStatus, Focus, GrabStartData as PointerGrabStartData,
};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::{Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::rustix::fs::unlink;
use smithay::utils::SERIAL_COUNTER;
use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};

use crate::backend::IpcOutputMap;
use crate::handlers::image_copy_capture;
use crate::input::pick_window_grab::PickWindowGrab;
use crate::layout::workspace::WorkspaceId;
use crate::niri::State;
use crate::utils::{version, with_toplevel_role};
use crate::window::Mapped;

// If an event stream client fails to read events fast enough that we accumulate more than this
// number in our buffer, we drop that event stream client.
const EVENT_STREAM_BUFFER_SIZE: usize = 64;

pub struct IpcServer {
    /// Path to the IPC socket.
    ///
    /// This is `None` when creating `IpcServer` without a socket.
    pub socket_path: Option<PathBuf>,
    event_streams: Rc<RefCell<Vec<EventStreamSender>>>,
    event_stream_state: Rc<RefCell<EventStreamState>>,
}

struct ClientCtx {
    owner: u64,
    alive: Rc<Cell<bool>>,
    event_loop: LoopHandle<'static, State>,
    scheduler: Scheduler<()>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
    event_streams: Rc<RefCell<Vec<EventStreamSender>>>,
    event_stream_state: Rc<RefCell<EventStreamState>>,
    agent_session: bool,
    desktop_session: Option<String>,
    agent_workspace: Option<String>,
}

impl Drop for ClientCtx {
    fn drop(&mut self) {
        self.alive.set(false);
        let owner = self.owner;
        self.event_loop
            .insert_idle(move |state| state.agent_disconnect(owner));
        if let Some(name) = self.agent_workspace.take() {
            self.event_loop.insert_idle(move |state| {
                let _ = state.niri.agent_workspace_changed(name, false, owner);
                state.retire_empty_agent_handoffs();
                state.niri.queue_redraw_all();
            });
        }
        if self.agent_session {
            self.event_loop.insert_idle(|state| {
                if state.niri.agent_indicator.session_changed(false) {
                    state.niri.queue_redraw_all();
                }
            });
        }
    }
}

struct EventStreamClient {
    events: Receiver<Event>,
    disconnect: Receiver<()>,
    write: Box<dyn AsyncWrite + Unpin>,
}

struct EventStreamSender {
    events: Sender<Event>,
    disconnect: Sender<()>,
}

impl IpcServer {
    pub fn start(
        event_loop: &LoopHandle<'static, State>,
        wayland_socket_name: Option<&OsStr>,
    ) -> anyhow::Result<Self> {
        let _span = tracy_client::span!("Ipc::start");

        let socket_path = if let Some(wayland_socket_name) = wayland_socket_name {
            let wayland_socket_name = wayland_socket_name.to_string_lossy();
            let socket_name = format!("niri.{wayland_socket_name}.{}.sock", process::id());
            let mut socket_path = socket_dir();
            socket_path.push(socket_name);

            let listener = UnixListener::bind(&socket_path).context("error binding socket")?;
            listener
                .set_nonblocking(true)
                .context("error setting socket to non-blocking")?;

            let source = Generic::new(listener, Interest::READ, Mode::Level);
            event_loop
                .insert_source(source, |_, socket, state| {
                    match socket.accept() {
                        Ok((stream, _)) => on_new_ipc_client(state, stream),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => (),
                        Err(e) => return Err(e),
                    }

                    Ok(PostAction::Continue)
                })
                .unwrap();

            Some(socket_path)
        } else {
            None
        };

        Ok(Self {
            socket_path,
            event_streams: Rc::new(RefCell::new(Vec::new())),
            event_stream_state: Rc::new(RefCell::new(EventStreamState::default())),
        })
    }

    fn send_event(&self, event: Event) {
        let mut streams = self.event_streams.borrow_mut();
        let mut to_remove = Vec::new();
        for (idx, stream) in streams.iter_mut().enumerate() {
            match stream.events.try_send(event.clone()) {
                Ok(()) => (),
                Err(TrySendError::Closed(_)) => to_remove.push(idx),
                Err(TrySendError::Full(_)) => {
                    warn!(
                        "disconnecting IPC event stream client \
                         because it is reading events too slowly"
                    );
                    to_remove.push(idx);
                }
            }
        }

        for idx in to_remove.into_iter().rev() {
            let stream = streams.swap_remove(idx);
            let _ = stream.disconnect.send_blocking(());
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        if let Some(socket_path) = &self.socket_path {
            let _ = unlink(socket_path);
        }
    }
}

fn socket_dir() -> PathBuf {
    BaseDirs::new()
        .as_ref()
        .and_then(|x| x.runtime_dir())
        .map(|x| x.to_owned())
        .unwrap_or_else(env::temp_dir)
}

fn on_new_ipc_client(state: &mut State, stream: UnixStream) {
    let _span = tracy_client::span!("on_new_ipc_client");
    trace!("new IPC client connected");

    let stream = match state.niri.event_loop.adapt_io(stream) {
        Ok(stream) => stream,
        Err(err) => {
            warn!("error making IPC stream async: {err:?}");
            return;
        }
    };

    let ipc_server = state.niri.ipc_server.as_ref().unwrap();

    let ctx = ClientCtx {
        owner: fastrand::u64(..),
        alive: Rc::new(Cell::new(true)),
        event_loop: state.niri.event_loop.clone(),
        scheduler: state.niri.scheduler.clone(),
        ipc_outputs: state.backend.ipc_outputs(),
        event_streams: ipc_server.event_streams.clone(),
        event_stream_state: ipc_server.event_stream_state.clone(),
        agent_session: false,
        desktop_session: None,
        agent_workspace: None,
    };

    let future = async move {
        if let Err(err) = handle_client(ctx, stream).await {
            warn!("error handling IPC client: {err:?}");
        }
    };
    if let Err(err) = state.niri.scheduler.schedule(future) {
        warn!("error scheduling IPC stream future: {err:?}");
    }
}

async fn handle_client(
    mut ctx: ClientCtx,
    stream: Async<'static, UnixStream>,
) -> anyhow::Result<()> {
    let (read, mut write) = stream.split();
    let mut read = BufReader::new(read);

    loop {
        // Don't keep buf around to avoid clients wasting RAM by filling it with bogus data.
        let mut buf = Vec::new();
        let res = read.read_until(b'\n', &mut buf).await;
        match res {
            Ok(0) => return Ok(()),
            Ok(_) => (),
            // Normal client disconnection.
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
            Err(err) => {
                return Err(err).context("error reading request");
            }
        }

        let request = serde_json::from_slice(&buf)
            .context("error parsing request")
            .map_err(|err| err.to_string());
        let requested_error = matches!(request, Ok(Request::ReturnError));
        let requested_event_stream = matches!(request, Ok(Request::EventStream));
        let requested_agent_session = matches!(
            request,
            Ok(Request::AgentSession | Request::AgentWorkspace { .. })
        );
        if requested_agent_session {
            anyhow::ensure!(
                ctx.agent_workspace.is_none() && !ctx.agent_session,
                "agent session already registered"
            );
        }
        if let Ok(Request::AgentWorkspace { workspace }) = &request {
            ctx.agent_workspace = Some(workspace.clone());
        }
        if let Ok(Request::AgentDesktopSession { session, .. }) = &request {
            anyhow::ensure!(
                ctx.desktop_session
                    .as_ref()
                    .is_none_or(|name| name == session),
                "desktop session identity changed"
            );
            ctx.desktop_session = Some(session.clone());
        }
        if matches!(request, Ok(Request::AgentSession)) {
            ctx.agent_session = true;
            ctx.event_loop.insert_idle(|state| {
                if state.niri.agent_indicator.session_changed(true) {
                    state.niri.queue_redraw_all();
                }
            });
        }

        let (reply, payload, close_after_reply) = match request {
            Ok(Request::Screenshot(request)) => {
                let Some((reply, payload)) = process_screenshot(&ctx, request, &mut read).await
                else {
                    return Ok(());
                };
                (reply, payload, true)
            }
            Ok(request)
                if ctx.agent_session
                    || ctx.agent_workspace.is_some()
                    || ctx.desktop_session.is_some() =>
            {
                let mut disconnected = [0];
                let reply = select_biased! {
                    result = process(&ctx, request).fuse() => result,
                    _ = read.read(&mut disconnected).fuse() => return Ok(()),
                };
                (reply, None, false)
            }
            Ok(request) => (process(&ctx, request).await, None, false),
            Err(err) => (Err(err), None, false),
        };

        if let Err(err) = &reply {
            if !requested_error {
                warn!("error processing IPC request: {err:?}");
            }
        }

        buf.clear();
        serde_json::to_writer(&mut buf, &reply).context("error formatting reply")?;
        buf.push(b'\n');
        write.write_all(&buf).await.context("error writing reply")?;

        if let Some(payload) = payload {
            write
                .write_all(&payload)
                .await
                .context("error writing screenshot data")?;
        }

        if close_after_reply {
            return Ok(());
        }

        if requested_event_stream {
            let (events_tx, events_rx) = async_channel::bounded(EVENT_STREAM_BUFFER_SIZE);
            let (disconnect_tx, disconnect_rx) = async_channel::bounded(1);

            // Spawn a task for the client.
            let client = EventStreamClient {
                events: events_rx,
                disconnect: disconnect_rx,
                write: Box::new(write) as _,
            };
            let future = async move {
                if let Err(err) = handle_event_stream_client(client).await {
                    warn!("error handling IPC event stream client: {err:?}");
                }
            };
            if let Err(err) = ctx.scheduler.schedule(future) {
                warn!("error scheduling IPC event stream future: {err:?}");
            }

            // Send the initial state.
            {
                let state = ctx.event_stream_state.borrow();
                for event in state.replicate() {
                    events_tx
                        .try_send(event)
                        .expect("initial event burst had more events than buffer size");
                }
            }

            // Add it to the list.
            {
                let mut streams = ctx.event_streams.borrow_mut();
                let sender = EventStreamSender {
                    events: events_tx,
                    disconnect: disconnect_tx,
                };
                streams.push(sender);
            }

            return Ok(());
        }
    }
}

async fn process(ctx: &ClientCtx, request: Request) -> Reply {
    let response = match request {
        Request::ReturnError => return Err(String::from("example compositor error")),
        Request::Version => Response::Version(version()),
        Request::Outputs => {
            let ipc_outputs = ctx.ipc_outputs.lock().unwrap().clone();
            let outputs = ipc_outputs.values().cloned().map(|o| (o.name.clone(), o));
            Response::Outputs(outputs.collect())
        }
        Request::Workspaces => {
            let state = ctx.event_stream_state.borrow();
            let workspaces = state.workspaces.workspaces.values().cloned().collect();
            Response::Workspaces(workspaces)
        }
        Request::Windows => {
            let state = ctx.event_stream_state.borrow();
            let windows = state.windows.windows.values().cloned().collect();
            Response::Windows(windows)
        }
        Request::Layers => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let mut layers = Vec::new();
                for output in state.niri.global_space.outputs() {
                    let name = output.name();
                    let map = layer_map_for_output(output);
                    for surface in map.layers() {
                        let Some(mapped) = state.niri.mapped_layer_surfaces.get(surface) else {
                            continue;
                        };
                        let Some(geometry) = map.layer_geometry(surface) else {
                            continue;
                        };
                        let layer = match surface.layer() {
                            Layer::Background => niri_ipc::Layer::Background,
                            Layer::Bottom => niri_ipc::Layer::Bottom,
                            Layer::Top => niri_ipc::Layer::Top,
                            Layer::Overlay => niri_ipc::Layer::Overlay,
                        };
                        let keyboard_interactivity =
                            match surface.cached_state().keyboard_interactivity {
                                KeyboardInteractivity::None => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::None
                                }
                                KeyboardInteractivity::Exclusive => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::Exclusive
                                }
                                KeyboardInteractivity::OnDemand => {
                                    niri_ipc::LayerSurfaceKeyboardInteractivity::OnDemand
                                }
                            };

                        layers.push(niri_ipc::LayerSurface {
                            id: mapped.id().get(),
                            pid: crate::utils::get_credentials_for_surface(surface.wl_surface())
                                .map(|c| c.pid),
                            geometry: (
                                geometry.loc.x,
                                geometry.loc.y,
                                geometry.size.w,
                                geometry.size.h,
                            ),
                            namespace: surface.namespace().to_owned(),
                            output: name.clone(),
                            layer,
                            keyboard_interactivity,
                        });
                    }
                }

                let _ = tx.send_blocking(layers);
            });
            let result = rx.recv().await;
            let layers = result.map_err(|_| String::from("error getting layers info"))?;
            Response::Layers(layers)
        }
        Request::KeyboardLayouts => {
            let state = ctx.event_stream_state.borrow();
            let layout = state.keyboard_layouts.keyboard_layouts.clone();
            let layout = layout.expect("keyboard layouts should be set at startup");
            Response::KeyboardLayouts(layout)
        }
        Request::FocusedWindow => {
            let state = ctx.event_stream_state.borrow();
            let windows = &state.windows.windows;
            let window = windows.values().find(|win| win.is_focused).cloned();
            Response::FocusedWindow(window)
        }
        Request::PickWindow => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let pointer = state.niri.seat.get_pointer().unwrap();
                let start_data = PointerGrabStartData {
                    focus: None,
                    button: 0,
                    location: pointer.current_location(),
                };
                let grab = PickWindowGrab::new(start_data);
                // The `WindowPickGrab` ungrab handler will cancel the previous ongoing pick, if
                // any.
                pointer.set_grab(state, grab, SERIAL_COUNTER.next_serial(), Focus::Clear);
                state.niri.pick_window = Some(tx);
                state
                    .niri
                    .cursor_manager
                    .set_cursor_image(CursorImageStatus::Named(CursorIcon::Crosshair));
                // Redraw to update the cursor.
                state.niri.queue_redraw_all();
            });
            let result = rx.recv().await;
            let id = result.map_err(|_| String::from("error getting picked window info"))?;
            let window = id.and_then(|id| {
                let state = ctx.event_stream_state.borrow();
                state.windows.windows.get(&id.get()).cloned()
            });
            Response::PickedWindow(window)
        }
        Request::PickColor => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                state.handle_pick_color(tx);
            });
            let result = rx.recv().await;
            let color = result.map_err(|_| String::from("error getting picked color"))?;
            Response::PickedColor(color)
        }
        Request::Screenshot(_) => unreachable!("screenshot requests are handled separately"),
        Request::Action(action) => {
            validate_action(&action)?;

            let (tx, rx) = async_channel::bounded(1);

            let action = niri_config::Action::from(action);
            ctx.event_loop.insert_idle(move |state| {
                // Make sure some logic like workspace clean-up has a chance to run before doing
                // actions.
                state.niri.advance_animations();
                state.do_action(action, false);
                let _ = tx.send_blocking(());
            });

            // Wait until the action has been processed before returning. This is important for a
            // few actions, for instance for DoScreenTransition this wait ensures that the screen
            // contents were sampled into the texture.
            let _ = rx.recv().await;
            Response::Handled
        }
        Request::Output { output, action } => {
            action.validate()?;

            let ipc_outputs = ctx.ipc_outputs.lock().unwrap();
            let found = ipc_outputs
                .values()
                .any(|o| OutputName::from_ipc_output(o).matches(&output));
            let response = if found {
                OutputConfigChanged::Applied
            } else {
                OutputConfigChanged::OutputWasMissing
            };
            drop(ipc_outputs);

            ctx.event_loop.insert_idle(move |state| {
                state.apply_transient_output_config(&output, action);
            });

            Response::OutputConfigChanged(response)
        }
        Request::FocusedOutput => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let active_output = state
                    .niri
                    .layout
                    .active_output()
                    .map(|output| output.name());

                let output = active_output.and_then(|active_output| {
                    state
                        .backend
                        .ipc_outputs()
                        .lock()
                        .unwrap()
                        .values()
                        .find(|o| o.name == active_output)
                        .cloned()
                });

                let _ = tx.send_blocking(output);
            });
            let result = rx.recv().await;
            let output = result.map_err(|_| String::from("error getting active output info"))?;
            Response::FocusedOutput(output)
        }
        Request::EventStream | Request::AgentSession => Response::Handled,
        Request::AgentDesktopSession { session, action } => {
            let (tx, rx) = async_channel::bounded(1);
            let alive = ctx.alive.clone();
            let owner = ctx.owner;
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                let result = state.control_desktop_session(session, action, owner);
                let _ = tx.send_blocking(result);
            });
            Response::AgentDesktopSession(
                rx.recv()
                    .await
                    .map_err(|_| "error controlling desktop session".to_owned())??,
            )
        }
        Request::OverviewState => {
            let state = ctx.event_stream_state.borrow();
            let is_open = state.overview.is_open;
            Response::OverviewState(Overview { is_open })
        }
        Request::Casts => {
            let state = ctx.event_stream_state.borrow();
            let casts = state.casts.casts.values().cloned().collect();
            Response::Casts(casts)
        }
        Request::AgentCleanupCheck { workspace } => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let _ = tx.send_blocking(state.niri.agent_cleanup_needed(&workspace));
            });
            Response::AgentCleanupNeeded(
                rx.recv()
                    .await
                    .map_err(|_| "error checking agent cleanup".to_owned())?,
            )
        }
        Request::AgentFinish { workspace } => {
            let (tx, rx) = async_channel::bounded(1);
            let alive = ctx.alive.clone();
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                let _ = tx.send_blocking(state.finish_agent_session(&workspace));
            });
            Response::AgentFinished(
                rx.recv()
                    .await
                    .map_err(|_| "error finishing agent workspace".to_owned())??,
            )
        }
        Request::AgentKeepWindow {
            workspace,
            window_id,
        } => {
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let _ = tx.send_blocking(state.niri.keep_agent_window(window_id, Some(&workspace)));
            });
            rx.recv()
                .await
                .map_err(|_| "error keeping agent window".to_owned())??;
            Response::Handled
        }
        Request::AgentWorkspace { ref workspace }
        | Request::AgentWorkspaceStatus { ref workspace } => {
            let workspace = workspace.clone();
            let connect = matches!(request, Request::AgentWorkspace { .. })
                || ctx
                    .agent_workspace
                    .as_ref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(&workspace));
            let owner = ctx.owner;
            let (tx, rx) = async_channel::bounded(1);
            ctx.event_loop.insert_idle(move |state| {
                let result = if connect {
                    state.niri.agent_workspace_changed(workspace, true, owner)
                } else {
                    state.niri.agent_workspace_status(&workspace, Some(owner))
                };
                let _ = tx.send_blocking(result);
            });
            Response::AgentWorkspace(
                rx.recv()
                    .await
                    .map_err(|_| "error getting agent workspace".to_owned())??,
            )
        }
        Request::AgentHandoff { ref workspace, .. } | Request::AgentResume { ref workspace } => {
            let workspace = workspace.clone();
            let reason = if let Request::AgentHandoff { reason, .. } = request {
                Some(reason)
            } else {
                None
            };
            let (tx, rx) = async_channel::bounded(1);
            let owner = ctx.owner;
            let alive = ctx.alive.clone();
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                let _ = tx.send_blocking(state.set_agent_workspace_control(
                    &workspace,
                    reason.is_some(),
                    reason,
                    Some(owner),
                ));
            });
            Response::AgentWorkspace(
                rx.recv()
                    .await
                    .map_err(|_| "error changing workspace control".to_owned())??,
            )
        }
        Request::AgentSpawn { command, workspace } => {
            if command.first().is_none_or(|program| program.is_empty()) {
                return Err(String::from("command requires an executable"));
            }
            let (tx, rx) = async_channel::bounded(1);
            let alive = ctx.alive.clone();
            let owner = ctx.owner;
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                if let Err(error) = state.niri.check_desktop_session_access(owner) {
                    let _ = tx.send_blocking(Err(error));
                    return;
                }
                let launch = if let Some(name) = &workspace {
                    let info = match state.niri.agent_workspace_status(name, None) {
                        Ok(info) if !info.paused => info,
                        Ok(_) => {
                            let _ =
                                tx.send_blocking(Err("workspace automation is paused".to_owned()));
                            return;
                        }
                        Err(error) => {
                            let _ = tx.send_blocking(Err(error));
                            return;
                        }
                    };
                    Some(Arc::new(crate::agent_input::AgentWorkspaceLaunch {
                        name: name.to_ascii_lowercase(),
                        workspace_id: WorkspaceId::specific(info.id),
                        existing: state
                            .niri
                            .layout
                            .windows()
                            .map(|(_, w)| w.id().get())
                            .collect(),
                        created: Instant::now(),
                    }))
                } else {
                    None
                };
                let mut marked = vec![String::from("env"), String::from("NIRI_AGENT_LAUNCH=1")];
                if let Some(launch) = &launch {
                    match state.niri.agent_launch_display(launch.clone()) {
                        Ok(display) => {
                            marked.push(format!("WAYLAND_DISPLAY={}", display.to_string_lossy()));
                        }
                        Err(error) => {
                            let _ = tx.send_blocking(Err(error));
                            return;
                        }
                    }
                }
                let (token, data) = state.niri.activation_state.create_external_token(None);
                if let Some(launch) = &launch {
                    data.user_data.insert_if_missing(|| launch.clone());
                }
                data.user_data
                    .insert_if_missing(|| crate::handlers::UrgentOnlyMarker::Agent);
                marked.extend(command);
                crate::utils::spawning::spawn(marked, Some(token.clone()));
                let Some(launch) = launch else {
                    let _ = tx.send_blocking(Ok(Vec::new()));
                    return;
                };
                let error_tx = tx.clone();
                if let Err(error) = state.niri.event_loop.insert_source(
                    Timer::from_duration(Duration::ZERO),
                    move |_, _, state| {
                        if !alive.get() {
                            return TimeoutAction::Drop;
                        }
                        let result = state.niri.agent_workspace_status(&launch.name, None).and_then(|info| {
                            if info.paused {
                                return Err("workspace automation is paused".to_owned());
                            }
                            let windows: Vec<_> = state.niri.layout.windows()
                                .filter(|(_, window)| {
                                    !launch.existing.contains(&window.id().get())
                                        && state.niri.agent_launch_for_surface(window.toplevel().wl_surface())
                                            .is_some_and(|other| Arc::ptr_eq(&other, &launch))
                                        && state.niri.check_agent_workspace_access(
                                            window.id().get(), Some(&launch.name)
                                        ).is_ok()
                                })
                                .map(|(_, window)| window.id().get())
                                .collect();
                            Ok(windows)
                        });
                        if result.as_ref().is_ok_and(Vec::is_empty) {
                            if launch.created.elapsed() < crate::handlers::XDG_ACTIVATION_TOKEN_TIMEOUT {
                                return TimeoutAction::ToDuration(Duration::from_millis(25));
                            }
                            let _ = tx.send_blocking(Err(format!(
                                "No new app window was confirmed on workspace {}. The command may have failed or reused an existing window. Inspect existing agent windows before retrying.", launch.name
                            )));
                        } else {
                            let _ = tx.send_blocking(result);
                        }
                        TimeoutAction::Drop
                    },
                ) {
                    let _ = error_tx.send_blocking(Err(error.to_string()));
                }
            });
            let windows = rx
                .recv()
                .await
                .map_err(|_| "error launching application".to_owned())??;
            if windows.is_empty() {
                Response::Handled
            } else {
                Response::AgentSpawn(windows)
            }
        }
        Request::AgentInput {
            window_id,
            workspace,
            geometry,
            events,
        } => {
            let (tx, rx) = async_channel::bounded(1);
            let owner = ctx.owner;
            let alive = ctx.alive.clone();
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                let cleanup = crate::agent_input::is_agent_cleanup(&events);
                let result = if cleanup {
                    Ok(())
                } else {
                    state
                        .niri
                        .check_agent_workspace_access(window_id, workspace.as_deref())
                }
                .and_then(|()| {
                    if let Some(expected) = geometry {
                        if state.agent_geometry(window_id)? != expected {
                            return Err("window geometry changed, take another screenshot".into());
                        }
                    }
                    state.agent_input(owner, window_id, events)
                });
                if result.is_err() {
                    state.agent_release_input(owner);
                }
                let _ = tx.send_blocking(result);
            });
            let result = rx.recv().await;
            result.map_err(|_| String::from("error delivering agent input"))??;
            Response::Handled
        }
        Request::AgentScreenshot {
            window_id,
            workspace,
            path,
            show_pointer,
        } => {
            if !Path::new(&path).is_absolute() {
                return Err(format!("path must be absolute: {path}"));
            }
            let (tx, rx) = async_channel::bounded(1);
            let owner = ctx.owner;
            let alive = ctx.alive.clone();
            ctx.event_loop.insert_idle(move |state| {
                if !alive.get() {
                    return;
                }
                let result = state
                    .niri
                    .check_agent_workspace_access(window_id, workspace.as_deref())
                    .and_then(|()| {
                        state.agent_screenshot(
                            owner,
                            workspace,
                            window_id,
                            path,
                            show_pointer,
                            tx.clone(),
                        )
                    });
                if let Err(err) = result {
                    let _ = tx.send_blocking(Err(err));
                }
            });
            let result = rx.recv().await;
            let info = result.map_err(|_| String::from("error taking agent screenshot"))??;
            Response::AgentScreenshot(info)
        }
    };

    Ok(response)
}

async fn process_screenshot(
    ctx: &ClientCtx,
    request: ScreenshotRequest,
    read: &mut (impl futures_util::AsyncRead + Unpin),
) -> Option<(Reply, Option<Vec<u8>>)> {
    let (tx, rx) = async_channel::bounded(1);
    let cancel_tx = tx.clone();
    ctx.event_loop.insert_idle(move |state| {
        state.handle_ipc_screenshot(request, tx);
    });

    let mut byte = [0];
    let result = select_biased! {
        result = rx.recv().fuse() => result,
        read = read.read(&mut byte).fuse() => {
            ctx.event_loop.insert_idle(move |state| {
                state.niri.cancel_ipc_screenshot(&cancel_tx);
            });

            return match read {
                Ok(0) => None,
                Ok(_) => Some((Err(String::from(
                    "screenshot requests must be last on an IPC connection",
                )), None)),
                Err(_) => None,
            };
        },
    };
    let result = match result {
        Ok(Ok(Some(result))) => result,
        Ok(Ok(None)) => return Some((Ok(Response::Screenshot(None)), None)),
        Ok(Err(err)) => return Some((Err(err), None)),
        Err(_) => {
            return Some((Err(String::from("error receiving screenshot result")), None));
        }
    };
    let response = Screenshot {
        data_length: result.len() as u64,
    };

    Some((Ok(Response::Screenshot(Some(response))), Some(result)))
}

fn validate_action(action: &Action) -> Result<(), String> {
    if let Action::Screenshot { path, .. }
    | Action::ScreenshotScreen { path, .. }
    | Action::ScreenshotWindow { path, .. }
    | Action::LoadConfigFile { path } = action
    {
        if let Some(path) = path {
            // Relative paths are resolved against the niri compositor's working directory, which
            // is almost certainly not what you want.
            if !Path::new(path).is_absolute() {
                return Err(format!("path must be absolute: {path}"));
            }
        }
    }

    if let Action::LoadConfigFile { path: Some(path) } = action {
        let p = Path::new(path);
        if !p.is_file() {
            return Err(format!("path does not point to a file: {path}"));
        }
    }

    Ok(())
}

async fn handle_event_stream_client(client: EventStreamClient) -> anyhow::Result<()> {
    let EventStreamClient {
        events,
        disconnect,
        mut write,
    } = client;

    while let Ok(event) = events.recv().await {
        let mut buf = serde_json::to_vec(&event).context("error formatting event")?;
        buf.push(b'\n');

        let res = select_biased! {
            _ = disconnect.recv().fuse() => return Ok(()),
            res = write.write_all(&buf).fuse() => res,
        };

        match res {
            Ok(()) => (),
            // Normal client disconnection.
            Err(err) if err.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
            res @ Err(_) => res.context("error writing event")?,
        }
    }

    Ok(())
}

fn make_ipc_window(
    mapped: &Mapped,
    workspace_id: Option<WorkspaceId>,
    layout: WindowLayout,
) -> niri_ipc::Window {
    with_toplevel_role(mapped.toplevel(), |role| niri_ipc::Window {
        id: mapped.id().get(),
        title: role.title.clone(),
        app_id: role.app_id.clone(),
        pid: mapped.credentials().map(|c| c.pid),
        workspace_id: workspace_id.map(|id| id.get()),
        is_focused: mapped.is_focused(),
        is_floating: mapped.is_floating(),
        is_urgent: mapped.is_urgent(),
        layout,
        focus_timestamp: mapped.get_focus_timestamp().map(Timestamp::from),
    })
}

impl State {
    pub fn ipc_keyboard_layouts_changed(&mut self) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let keyboard_layouts = keyboard.with_xkb_state(self, |context| {
            let xkb = context.xkb().lock().unwrap();
            let layouts = xkb.layouts();
            KeyboardLayouts {
                names: layouts
                    .map(|layout| xkb.layout_name(layout).to_owned())
                    .collect(),
                current_idx: xkb.active_layout().0 as u8,
            }
        });

        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.keyboard_layouts;

        let event = Event::KeyboardLayoutsChanged { keyboard_layouts };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_keyboard_layout_index(&mut self) {
        let keyboard = self.niri.seat.get_keyboard().unwrap();
        let idx = keyboard.with_xkb_state(self, |context| {
            let xkb = context.xkb().lock().unwrap();
            xkb.active_layout().0 as u8
        });

        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.keyboard_layouts;

        if state.keyboard_layouts.as_ref().unwrap().current_idx == idx {
            return;
        }

        let event = Event::KeyboardLayoutSwitched { idx };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_layout(&mut self) {
        self.ipc_refresh_workspaces();
        self.ipc_refresh_windows();
        self.ipc_refresh_overview();
    }

    fn ipc_refresh_workspaces(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_workspaces");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.workspaces;

        let mut events = Vec::new();
        let layout = &self.niri.layout;
        let focused_ws_id = layout.active_workspace().map(|ws| ws.id().get());

        // Check for workspace changes.
        let mut seen = HashSet::new();
        let mut need_workspaces_changed = false;
        for (mon, ws_idx, ws) in layout.workspaces() {
            let id = ws.id().get();
            seen.insert(id);

            let Some(ipc_ws) = state.workspaces.get(&id) else {
                // A new workspace was added.
                need_workspaces_changed = true;
                break;
            };

            // Check for any changes that we can't signal as individual events.
            let output_name = mon.map(|mon| mon.output_name());
            if ipc_ws.idx != u8::try_from(ws_idx + 1).unwrap_or(u8::MAX)
                || ipc_ws.name.as_ref() != ws.name()
                || ipc_ws.output.as_ref() != output_name
            {
                need_workspaces_changed = true;
                break;
            }

            let active_window_id = ws.active_window().map(|win| win.id().get());
            if ipc_ws.active_window_id != active_window_id {
                events.push(Event::WorkspaceActiveWindowChanged {
                    workspace_id: id,
                    active_window_id,
                });
            }

            // Check if this workspace urgent state changed.
            let urgent = ws.is_urgent();
            if urgent != ipc_ws.is_urgent {
                events.push(Event::WorkspaceUrgencyChanged { id, urgent });
            }

            // Check if this workspace became focused.
            let is_focused = Some(id) == focused_ws_id;
            if is_focused && !ipc_ws.is_focused {
                events.push(Event::WorkspaceActivated { id, focused: true });
                continue;
            }

            // Check if this workspace became active.
            let is_active = mon.is_some_and(|mon| mon.active_workspace_idx() == ws_idx);
            if is_active && !ipc_ws.is_active {
                events.push(Event::WorkspaceActivated { id, focused: false });
            }
        }

        // Check if any workspaces were removed.
        if !need_workspaces_changed && state.workspaces.keys().any(|id| !seen.contains(id)) {
            need_workspaces_changed = true;
        }

        if need_workspaces_changed {
            events.clear();

            let workspaces = layout
                .workspaces()
                .map(|(mon, ws_idx, ws)| {
                    let id = ws.id().get();
                    Workspace {
                        id,
                        idx: u8::try_from(ws_idx + 1).unwrap_or(u8::MAX),
                        name: ws.name().cloned(),
                        output: mon.map(|mon| mon.output_name().clone()),
                        is_urgent: ws.is_urgent(),
                        is_active: mon.is_some_and(|mon| mon.active_workspace_idx() == ws_idx),
                        is_focused: Some(id) == focused_ws_id,
                        active_window_id: ws.active_window().map(|win| win.id().get()),
                    }
                })
                .collect();

            events.push(Event::WorkspacesChanged { workspaces });
        }

        for event in events {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    fn ipc_refresh_windows(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_windows");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.windows;

        let mut events = Vec::new();
        let layout = &self.niri.layout;

        let mut batch_change_layouts: Vec<(u64, WindowLayout)> = Vec::new();

        // Check for window changes.
        let mut seen = HashSet::new();
        let mut focused_id = None;
        layout.with_windows(|mapped, _, ws_id, window_layout| {
            let id = mapped.id().get();
            seen.insert(id);

            if mapped.is_focused() {
                focused_id = Some(id);
            }

            let Some(ipc_win) = state.windows.get(&id) else {
                let window = make_ipc_window(mapped, ws_id, window_layout);
                events.push(Event::WindowOpenedOrChanged { window });
                return;
            };

            let workspace_id = ws_id.map(|id| id.get());
            let mut changed =
                ipc_win.workspace_id != workspace_id || ipc_win.is_floating != mapped.is_floating();

            changed |= with_toplevel_role(mapped.toplevel(), |role| {
                ipc_win.title != role.title || ipc_win.app_id != role.app_id
            });

            if changed {
                let window = make_ipc_window(mapped, ws_id, window_layout);
                events.push(Event::WindowOpenedOrChanged { window });
                return;
            }

            if ipc_win.layout != window_layout {
                batch_change_layouts.push((id, window_layout));
            }

            if mapped.is_focused() && !ipc_win.is_focused {
                events.push(Event::WindowFocusChanged { id: Some(id) });
            }

            let focus_timestamp = mapped.get_focus_timestamp().map(Timestamp::from);
            if focus_timestamp != ipc_win.focus_timestamp {
                events.push(Event::WindowFocusTimestampChanged {
                    id,
                    focus_timestamp,
                });
            }

            let urgent = mapped.is_urgent();
            if urgent != ipc_win.is_urgent {
                events.push(Event::WindowUrgencyChanged { id, urgent })
            }
        });

        // It might make sense to push layout changes after closed windows (since windows about to
        // be closed will occupy the same column/tile positions as the window that moved into this
        // vacated space), but also we are already pushing some layout changes in
        // WindowOpenedOrChanged above, meaning that the receiving end has to handle this case
        // anyway.
        if !batch_change_layouts.is_empty() {
            events.push(Event::WindowLayoutsChanged {
                changes: batch_change_layouts,
            });
        }

        // Check for closed windows.
        let mut ipc_focused_id = None;
        for (id, ipc_win) in &state.windows {
            if !seen.contains(id) {
                events.push(Event::WindowClosed { id: *id });
            }

            if ipc_win.is_focused {
                ipc_focused_id = Some(id);
            }
        }

        // Extra check for focus becoming None, since the checks above only work for focus becoming
        // a different window.
        if focused_id.is_none() && ipc_focused_id.is_some() {
            events.push(Event::WindowFocusChanged { id: None });
        }

        for event in events {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    pub fn ipc_refresh_overview(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.overview;
        let is_open = self.niri.layout.is_overview_open();

        if state.is_open == is_open {
            return;
        }

        let event = Event::OverviewOpenedOrClosed { is_open };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_refresh_casts(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let _span = tracy_client::span!("State::ipc_refresh_casts");

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.casts;

        let mut events = Vec::new();
        let mut seen = HashSet::new();

        // Check PipeWire screencasts.
        #[cfg(feature = "xdp-gnome-screencast")]
        {
            // Check pending dynamic casts.
            for pending in &self.niri.casting.pending_dynamic_casts {
                let stream_id = pending.stream_id.get();
                seen.insert(stream_id);

                // Pending dynamic casts don't change any properties, so we only need to check if
                // it's missing from the state.
                if !state.casts.contains_key(&stream_id) {
                    let cast = niri_ipc::Cast {
                        session_id: pending.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::PipeWire,
                        target: niri_ipc::CastTarget::Nothing {},
                        is_dynamic_target: true,
                        is_active: false,
                        pid: None,
                        pw_node_id: None,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }

            // Check active casts.
            for cast in &self.niri.casting.casts {
                let stream_id = cast.stream_id.get();
                seen.insert(stream_id);

                let pw_node_id = cast.node_id();
                if state.casts.get(&stream_id).is_none_or(|existing| {
                    // Only these properties can change.
                    existing.is_active != cast.is_active()
                        || !cast.target.matches(&existing.target)
                        || existing.pw_node_id != pw_node_id
                }) {
                    let cast = niri_ipc::Cast {
                        session_id: cast.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::PipeWire,
                        target: cast.target.make_ipc(),
                        is_dynamic_target: cast.dynamic_target,
                        is_active: cast.is_active(),
                        pid: None,
                        pw_node_id,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }
        }

        // Check screencopy casts.
        //
        // First, clear expired casts. Ideally we'd have a deadline timer, but our 1 second frame
        // callback timer calls refresh regularly, so that's fine as is.
        self.niri.screencopy_state.clear_expired_casts();

        for queue in self.niri.screencopy_state.queues() {
            if let Some(cast_info) = queue.cast() {
                let stream_id = cast_info.stream_id.get();
                seen.insert(stream_id);

                if state.casts.get(&stream_id).is_none_or(|existing| {
                    // Only this property can change.
                    match &existing.target {
                        niri_ipc::CastTarget::Output { name } => *name != cast_info.output_name,
                        _ => true,
                    }
                }) {
                    let cast = niri_ipc::Cast {
                        session_id: cast_info.session_id.get(),
                        stream_id,
                        kind: niri_ipc::CastKind::WlrScreencopy,
                        target: niri_ipc::CastTarget::Output {
                            name: cast_info.output_name.clone(),
                        },
                        is_dynamic_target: false,
                        is_active: true,
                        pid: queue.credentials().map(|creds| creds.pid),
                        pw_node_id: None,
                    };
                    events.push(Event::CastStartedOrChanged { cast });
                }
            }
        }

        // Check ext-image-copy-capture casts.
        //
        // Unlike wlr-screencopy, a session is an explicit protocol object, so
        // the cast lifetime is tied to it. Dead sessions are already dropped by
        // refresh_image_copy_capture(), which runs before this.
        let output_sessions = self
            .niri
            .image_copy_sessions
            .iter()
            .map(|s| (s.session_id, s.stream_id, s.session.source(), s.credentials));
        let cursor_sessions = self
            .niri
            .image_copy_cursor_sessions
            .iter()
            .map(|s| (s.session_id, s.stream_id, s.session.source(), s.credentials));
        for (session_id, stream_id, source, credentials) in output_sessions.chain(cursor_sessions) {
            let Some(output) = image_copy_capture::source_output(&source) else {
                continue;
            };

            let stream_id = stream_id.get();
            seen.insert(stream_id);

            if !state.casts.contains_key(&stream_id) {
                let cast = niri_ipc::Cast {
                    session_id: session_id.get(),
                    stream_id,
                    kind: niri_ipc::CastKind::ExtImageCopyCapture,
                    target: niri_ipc::CastTarget::Output {
                        name: output.name(),
                    },
                    is_dynamic_target: false,
                    is_active: true,
                    pid: credentials.map(|creds| creds.pid),
                    pw_node_id: None,
                };
                events.push(Event::CastStartedOrChanged { cast });
            }
        }

        // Check for stopped casts.
        for stream_id in state.casts.keys() {
            if !seen.contains(stream_id) {
                events.push(Event::CastStopped {
                    stream_id: *stream_id,
                });
            }
        }

        for event in events {
            state.apply(event.clone());
            server.send_event(event);
        }
    }

    pub fn ipc_refresh_touchpad(&mut self) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        let mut state = server.event_stream_state.borrow_mut();
        let state = &mut state.touchpad;
        let on = !self.niri.touchpad_disabled_by_toggle;

        if state.on == on {
            return;
        }

        let event = Event::TouchpadToggled { on };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_config_loaded(&mut self, failed: bool) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };
        let mut state = server.event_stream_state.borrow_mut();

        let event = Event::ConfigLoaded { failed };
        state.apply(event.clone());
        server.send_event(event);
    }

    pub fn ipc_agent_input_activity(&mut self, window_id: u64, kind: AgentActivityKind) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };

        // Transient: there's no state to apply.
        server.send_event(Event::AgentInputActivity { window_id, kind });
    }

    pub fn ipc_screenshot_taken(&mut self, path: Option<String>) {
        let Some(server) = &self.niri.ipc_server else {
            return;
        };
        let mut state = server.event_stream_state.borrow_mut();

        let event = Event::ScreenshotCaptured { path };
        state.apply(event.clone());
        server.send_event(event);
    }
}
