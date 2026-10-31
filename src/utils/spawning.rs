use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex as StdMutex, RwLock};
use std::{io, thread};

use libc::{getrlimit, rlimit, setrlimit, RLIMIT_NOFILE};
use niri_config::Environment;
use smithay::wayland::xdg_activation::XdgActivationToken;

#[doc(hidden)]
pub use crate::spawn_helper::spawn_child;
use crate::spawn_helper::SpawnRequest;
use crate::utils::expand_home;

pub static REMOVE_ENV_RUST_BACKTRACE: AtomicBool = AtomicBool::new(false);
pub static REMOVE_ENV_RUST_LIB_BACKTRACE: AtomicBool = AtomicBool::new(false);
pub static CHILD_ENV: RwLock<Environment> = RwLock::new(Environment(Vec::new()));
pub static CHILD_DISPLAY: RwLock<Option<String>> = RwLock::new(None);

static ORIGINAL_NOFILE_RLIMIT: StdMutex<rlimit> = StdMutex::new(rlimit {
    rlim_cur: 0,
    rlim_max: 0,
});

/// Increases the nofile rlimit to the maximum and stores the original value.
pub fn store_and_increase_nofile_rlimit() {
    let mut rlim = rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { getrlimit(RLIMIT_NOFILE, &mut rlim) } != 0 {
        let err = io::Error::last_os_error();
        warn!("error getting nofile rlimit: {err:?}");
        return;
    }

    *ORIGINAL_NOFILE_RLIMIT.lock().unwrap() = rlim;

    trace!(
        "changing nofile rlimit from {} to {}",
        rlim.rlim_cur,
        rlim.rlim_max
    );
    rlim.rlim_cur = rlim.rlim_max;

    if unsafe { setrlimit(RLIMIT_NOFILE, &rlim) } != 0 {
        let err = io::Error::last_os_error();
        warn!("error setting nofile rlimit: {err:?}");
    }
}

struct SpawnCommand {
    command: Vec<OsString>,
    token: Option<XdgActivationToken>,
}

#[cfg(feature = "systemd")]
const MAX_INITIAL_WARM_HELPERS: usize = 4;

fn spawner(initial_warm: usize) -> io::Result<async_channel::Sender<SpawnCommand>> {
    static SPAWNER: StdMutex<Option<async_channel::Sender<SpawnCommand>>> = StdMutex::new(None);
    let mut spawner = SPAWNER.lock().unwrap();
    if let Some(sender) = spawner.as_ref().filter(|sender| !sender.is_closed()) {
        return Ok(sender.clone());
    }

    #[cfg(not(feature = "systemd"))]
    let _ = initial_warm;

    // Spawned apps like Bubblewrap (`--die-with-parent`) set PR_SET_PDEATHSIG, which Linux ties to
    // the thread that created the child.
    let (sender, receiver) = async_channel::unbounded();
    thread::Builder::new()
        .name("Command Spawner".to_owned())
        .spawn(move || {
            #[cfg(feature = "systemd")]
            if systemd::active() {
                systemd::run(receiver, initial_warm.clamp(1, MAX_INITIAL_WARM_HELPERS));
                return;
            }
            run_direct(receiver);
        })?;
    *spawner = Some(sender.clone());
    Ok(sender)
}

/// Spawns the command to run independently of the compositor.
pub fn spawn<T: AsRef<OsStr>>(command: Vec<T>, token: Option<XdgActivationToken>) {
    let _span = tracy_client::span!();

    if command.is_empty() {
        return;
    }

    let spawner = match spawner(1) {
        Ok(spawner) => spawner,
        Err(err) => {
            warn!("error starting the command spawner: {err:?}");
            return;
        }
    };

    let command = command.iter().map(|arg| arg.as_ref().to_owned()).collect();
    if let Err(err) = spawner.try_send(SpawnCommand { command, token }) {
        warn!("error sending command to the spawner: {err:?}");
    }
}

/// Spawns the command through the shell.
///
/// We hardcode `sh -c`, consistent with other compositors:
///
/// - https://github.com/swaywm/sway/blob/b3dcde8d69c3f1304b076968a7a64f54d0c958be/sway/commands/exec_always.c#L64
/// - https://github.com/hyprwm/Hyprland/blob/1ac1ff457ab8ef1ae6a8f2ab17ee7965adfa729f/src/managers/KeybindManager.cpp#L987
pub fn spawn_sh(command: String, token: Option<XdgActivationToken>) {
    spawn(vec![String::from("sh"), String::from("-c"), command], token);
}

fn build_spawn_request(SpawnCommand { command, token }: SpawnCommand) -> SpawnRequest {
    let _span = tracy_client::span!();

    let (program, args) = command.split_first().unwrap();
    let mut command = program.as_os_str();

    // Expand `~` at the start.
    let expanded = expand_home(Path::new(command));
    match &expanded {
        Ok(Some(expanded)) => command = expanded.as_ref(),
        Ok(None) => (),
        Err(err) => {
            warn!("error expanding ~: {err:?}");
        }
    }

    // Keep the overrides out of the helper's own environment. Forward cursor env vars too
    // because CursorManager::reload mutates them at runtime after a warm helper was started.
    let mut env = BTreeMap::from([
        (
            "DISPLAY".into(),
            CHILD_DISPLAY.read().unwrap().clone().map(OsString::from),
        ),
        ("NOTIFY_SOCKET".into(), None),
        ("XCURSOR_THEME".into(), std::env::var_os("XCURSOR_THEME")),
        ("XCURSOR_SIZE".into(), std::env::var_os("XCURSOR_SIZE")),
    ]);
    for (name, remove) in [
        ("RUST_BACKTRACE", &REMOVE_ENV_RUST_BACKTRACE),
        ("RUST_LIB_BACKTRACE", &REMOVE_ENV_RUST_LIB_BACKTRACE),
    ] {
        if remove.load(Ordering::Relaxed) {
            env.insert(name.into(), None);
        }
    }
    for var in &CHILD_ENV.read().unwrap().0 {
        env.insert(
            var.name.as_str().into(),
            var.value.as_deref().map(OsString::from),
        );
    }
    if let Some(token) = token {
        for name in ["XDG_ACTIVATION_TOKEN", "DESKTOP_STARTUP_ID"] {
            env.insert(name.into(), Some(token.as_str().into()));
        }
    }

    let rlim = *ORIGINAL_NOFILE_RLIMIT.lock().unwrap();
    SpawnRequest {
        // Preserve invalid NULs so the helper's Command rejects them.
        program: command.to_owned(),
        args: args.to_vec(),
        env: env.into_iter().collect(),
        nofile_soft: rlim.rlim_cur,
        nofile_hard: rlim.rlim_max,
    }
}

fn run_direct(receiver: async_channel::Receiver<SpawnCommand>) {
    let mut warm: Option<(async_process::Child, io::PipeWriter)> = start_spawn_helper().ok();

    while let Ok(command) = receiver.recv_blocking() {
        let request = build_spawn_request(command);
        let program = request.program.clone();
        let data = match serde_json::to_vec(&request) {
            Ok(data) => data,
            Err(err) => {
                warn!("error serializing spawn request for {program:?}: {err:?}");
                continue;
            }
        };

        let mut launched = false;
        if let Some((mut child, mut writer)) = warm.take() {
            if matches!(child.try_status(), Ok(None)) && writer.write_all(&data).is_ok() {
                drop(writer);
                drop(child);
                launched = true;
            } else {
                drop(writer);
                drop(child);
            }
        }

        if !launched {
            if let Err(err) = launch_helper_direct(&data) {
                warn!("error sending {program:?} to the spawn helper: {err:?}");
            }
        }

        warm = start_spawn_helper().ok();
    }
}

struct SpawnHelper {
    path: PathBuf,
    #[cfg(target_os = "linux")]
    _file: std::fs::File,
}

// Update this name with incompatible changes to SpawnRequest.
pub(crate) const SPAWN_HELPER_NAME: &str = "niri-spawn-v1";
static SPAWN_HELPER: LazyLock<Option<SpawnHelper>> = LazyLock::new(SpawnHelper::find);

#[doc(hidden)]
pub fn prepare_spawn_helper() {
    LazyLock::force(&SPAWN_HELPER);
}

#[doc(hidden)]
pub fn warm_spawner(initial_warm: usize) {
    let _ = spawner(initial_warm);
}

impl SpawnHelper {
    fn find() -> Option<Self> {
        let path = std::env::current_exe()
            .ok()?
            .with_file_name(SPAWN_HELPER_NAME);
        if !path.is_file() {
            return None;
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;

            // Keep using this helper if an upgrade replaces it while Niri is running.
            let file = std::fs::File::open(path).ok()?;
            let path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
            Some(Self { path, _file: file })
        }
        #[cfg(not(target_os = "linux"))]
        Some(Self { path })
    }
}

fn spawn_with_input(
    mut command: async_process::Command,
) -> io::Result<(async_process::Child, io::PipeWriter)> {
    let (input, writer) = io::pipe()?;
    // Keep setup out of pre_exec so launching the helper can use posix_spawn.
    let child = command.stdin(Stdio::from(input)).spawn()?;
    Ok((child, writer))
}

fn start_spawn_helper() -> io::Result<(async_process::Child, io::PipeWriter)> {
    if let Some(helper) = &*SPAWN_HELPER {
        use async_process::unix::CommandExt as _;

        let mut command = async_process::Command::new(&helper.path);
        // Set argv[0] so /proc/self/fd/<fd> shows up as niri-spawn-v1 in ps.
        command.arg0(SPAWN_HELPER_NAME);
        match spawn_with_input(command) {
            Ok(child) => return Ok(child),
            Err(err) => debug!("error starting the packaged spawn helper: {err}"),
        }
    }

    // A standalone Niri executable can still launch apps.
    #[cfg(target_os = "linux")]
    let executable = linux_executable()?;
    #[cfg(not(target_os = "linux"))]
    let executable = std::env::current_exe()?;
    let mut command = async_process::Command::new(executable);
    command.arg("spawn");
    spawn_with_input(command)
}

#[cfg(target_os = "linux")]
fn linux_executable() -> io::Result<PathBuf> {
    use std::ffi::CStr;
    use std::os::unix::ffi::OsStrExt;

    let proc_exe = Path::new("/proc/self/exe");
    if proc_exe.exists() {
        return Ok(proc_exe.to_owned());
    }

    // In a sandbox without procfs, use the exec path the kernel supplied at startup.
    let path = unsafe { libc::getauxval(libc::AT_EXECFN) } as *const libc::c_char;
    if path.is_null() {
        return Err(io::Error::last_os_error());
    }
    let path = unsafe { CStr::from_ptr(path) };
    std::path::absolute(OsStr::from_bytes(path.to_bytes()))
}

fn launch_helper_direct(data: &[u8]) -> anyhow::Result<()> {
    let (child, mut writer) = start_spawn_helper()?;
    writer.write_all(data)?;
    // async-process reaps children without blocking the spawning thread.
    drop(child);
    Ok(())
}

#[cfg(feature = "systemd")]
mod systemd {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::Poll;
    use std::time::{Duration, Instant};

    use anyhow::Context;
    use async_executor::LocalExecutor;
    use futures_util::future::{select, Either};
    use futures_util::io::AsyncWriteExt;
    use futures_util::lock::Mutex;
    use futures_util::StreamExt;
    use zbus::zvariant::{OwnedObjectPath, Value};

    use super::*;

    const SCOPE_TIMEOUT: Duration = Duration::from_secs(5);
    // POSIX PIPE_BUF: a single write of this size to a fresh empty pipe does not block.
    const SYNC_PIPE_WRITE_LIMIT: usize = 4096;

    struct WarmHelper {
        child: async_process::Child,
        writer: io::PipeWriter,
    }

    struct WarmState {
        ready: VecDeque<WarmHelper>,
        warming: usize,
        target: usize,
    }

    pub(super) fn active() -> bool {
        crate::utils::IS_SYSTEMD_SERVICE.load(Ordering::Relaxed)
    }

    pub(super) fn run(receiver: async_channel::Receiver<SpawnCommand>, initial_warm: usize) {
        // LocalExecutor keeps every start_spawn_helper() call on this long-lived spawner thread so
        // children using PR_SET_PDEATHSIG (e.g. Bubblewrap) do not get killed when a worker exits.
        let executor = Rc::new(LocalExecutor::new());
        let state = Rc::new(RefCell::new(WarmState {
            ready: VecDeque::new(),
            warming: 0,
            target: initial_warm,
        }));

        async_io::block_on(executor.run(async {
            ensure_warming(&executor, &state);

            while let Ok(command) = receiver.recv().await {
                launch_command(&executor, &state, command);
                // Let background warming and burst tasks make progress between queued commands.
                yield_once().await;
            }
        }));
    }

    async fn yield_once() {
        let mut first_poll = true;
        std::future::poll_fn(|cx| {
            if std::mem::take(&mut first_poll) {
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        })
        .await;
    }

    fn ensure_warming(executor: &Rc<LocalExecutor<'static>>, state: &Rc<RefCell<WarmState>>) {
        let to_spawn = {
            let mut warm = state.borrow_mut();
            let have = warm.ready.len() + warm.warming;
            let needed = warm.target.saturating_sub(have);
            warm.warming += needed;
            needed
        };

        for _ in 0..to_spawn {
            let state = Rc::clone(state);
            executor
                .spawn(async move {
                    let helper = warm_helper().await;
                    let mut warm = state.borrow_mut();
                    warm.warming = warm.warming.saturating_sub(1);
                    if let Some(helper) = helper {
                        if warm.ready.len() < warm.target {
                            warm.ready.push_back(helper);
                        }
                    }
                })
                .detach();
        }
    }

    fn take_warm_helper(state: &Rc<RefCell<WarmState>>) -> Option<WarmHelper> {
        let mut warm = state.borrow_mut();
        // After startup burst helpers are consumed, settle back to 1 idle helper.
        if warm.target > 1 {
            warm.target -= 1;
        }
        while let Some(mut helper) = warm.ready.pop_front() {
            if matches!(helper.child.try_status(), Ok(None)) {
                return Some(helper);
            }
            debug!("discarding exited pre-warmed spawn helper");
            drop(helper);
        }
        None
    }

    async fn warm_helper() -> Option<WarmHelper> {
        let (child, writer) = match start_spawn_helper() {
            Ok(pair) => pair,
            Err(err) => {
                warn!("error starting pre-warmed spawn helper: {err:?}");
                return None;
            }
        };

        // Only cache helpers that systemd confirmed in a scope; otherwise drop it so the next
        // launch retries scoping once the session bus recovers.
        if !place_in_scope(child.id()).await {
            drop(writer);
            drop(child);
            return None;
        }
        Some(WarmHelper { child, writer })
    }

    async fn place_in_scope(pid: u32) -> bool {
        let Some(manager) = manager().await else {
            return false;
        };

        // Keep the app-niri- prefix so app-.scope.d/ and app-niri-.scope.d/ drop-ins apply, while
        // avoiding a reverse-DNS app ID that xdg-desktop-portal could misattribute.
        let scope_name = format!("app-niri-spawn-{pid}.scope");
        match with_timeout(
            SCOPE_TIMEOUT,
            start_systemd_scope(&manager, &scope_name, pid),
        )
        .await
        {
            Some(Ok(())) => true,
            Some(Err(err)) => {
                warn!("error starting systemd scope {scope_name}: {err:#}");
                false
            }
            None => {
                warn!("timed out starting systemd scope {scope_name}");
                false
            }
        }
    }

    fn launch_command(
        executor: &Rc<LocalExecutor<'static>>,
        state: &Rc<RefCell<WarmState>>,
        command: SpawnCommand,
    ) {
        let request = build_spawn_request(command);
        let program = request.program.clone();
        let data = match serde_json::to_vec(&request) {
            Ok(data) => data,
            Err(err) => {
                warn!("error serializing spawn request for {program:?}: {err:?}");
                return;
            }
        };

        // Fast path: reuse a live pre-warmed helper that is already in its own scope.
        let warm = take_warm_helper(state);
        if let Some(mut warm) = warm {
            ensure_warming(executor, state);

            if data.len() <= SYNC_PIPE_WRITE_LIMIT {
                match warm.writer.write_all(&data) {
                    Ok(()) => {
                        drop(warm.writer);
                        drop(warm.child);
                        return;
                    }
                    Err(err) => {
                        debug!(
                            "pre-warmed spawn helper pipe failed ({err:?}); falling back to fresh helper"
                        );
                        drop(warm.writer);
                        drop(warm.child);
                    }
                }
            } else {
                let child = warm.child;
                let writer = warm.writer;
                executor
                    .spawn(async move {
                        if let Err(err) = write_pipe_async(writer, &data).await {
                            debug!(
                                "pre-warmed spawn helper async pipe failed ({err:?}); falling back to fresh helper"
                            );
                            drop(child);
                            // spawn_child reads stdin to EOF before exec, so a failed write cannot
                            // have launched the target yet.
                            launch_fresh_async(program, data).await;
                            return;
                        }
                        drop(child);
                    })
                    .detach();
                return;
            }
        }

        // Burst/fallback path: start a fresh helper, place it in its scope before exec so
        // immediate forks stay contained and empty-scope races cannot happen, then send the
        // request.
        ensure_warming(executor, state);
        executor
            .spawn(async move {
                launch_fresh_async(program, data).await;
            })
            .detach();
    }

    async fn launch_fresh_async(program: OsString, data: Vec<u8>) {
        let (child, writer) = match start_spawn_helper() {
            Ok(pair) => pair,
            Err(err) => {
                warn!("error starting fallback spawn helper for {program:?}: {err:?}");
                return;
            }
        };
        place_in_scope(child.id()).await;
        if let Err(err) = write_pipe_async(writer, &data).await {
            warn!("error sending {program:?} to the fallback spawn helper: {err:?}");
        }
        drop(child);
    }

    async fn write_pipe_async(writer: io::PipeWriter, data: &[u8]) -> io::Result<()> {
        let mut writer = async_io::Async::new(writer)?;
        writer.write_all(data).await?;
        writer.close().await?;
        Ok(())
    }

    enum BusState {
        Disconnected,
        // Reusing the proxy keeps zbus's watch on systemd's bus name across launches.
        Connected(zbus::Proxy<'static>),
        Backoff(Instant),
    }

    async fn with_timeout<T>(timeout: Duration, future: impl Future<Output = T>) -> Option<T> {
        match select(std::pin::pin!(future), async_io::Timer::after(timeout)).await {
            Either::Left((result, _)) => Some(result),
            Either::Right(_) => None,
        }
    }

    async fn manager() -> Option<zbus::Proxy<'static>> {
        static BUS: Mutex<BusState> = Mutex::new(BusState::Disconnected);
        const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);

        let mut bus = BUS.lock().await;
        match &*bus {
            BusState::Connected(proxy) if !proxy.connection().is_closed() => {
                return Some(proxy.clone());
            }
            BusState::Backoff(retry_at) if Instant::now() < *retry_at => return None,
            _ => (),
        }

        let connect = async {
            zbus::Proxy::new_owned(
                zbus::Connection::session().await?,
                "org.freedesktop.systemd1",
                "/org/freedesktop/systemd1",
                "org.freedesktop.systemd1.Manager",
            )
            .await
        };
        match with_timeout(CONNECT_TIMEOUT, connect).await {
            Some(Ok(proxy)) => {
                *bus = BusState::Connected(proxy.clone());
                return Some(proxy);
            }
            Some(Err(err)) => {
                warn!("could not connect to the session bus for systemd scopes: {err}");
            }
            None => warn!("timed out connecting to the session bus for systemd scopes"),
        }
        // Let the other pending launches proceed without repeating a failed handshake.
        *bus = BusState::Backoff(Instant::now() + CONNECT_TIMEOUT);
        None
    }

    async fn start_systemd_scope(
        proxy: &zbus::Proxy<'_>,
        scope_name: &str,
        pid: u32,
    ) -> anyhow::Result<()> {
        // Drain `JobRemoved` concurrently while `StartTransientUnit` is pending: zbus shares a
        // bounded broadcast queue across identical match rules and its socket reader stalls if any
        // subscriber stops polling. Filtering by `arg2` is avoided because dbus-broker 28 (RHEL 9 /
        // AlmaLinux 9) checks arg0's type for all `argN` matches and rejects `arg2` on
        // `JobRemoved`.
        let mut signals = proxy
            .receive_signal("JobRemoved")
            .await
            .context("error creating a signal iterator")?;

        let pids: &[_] = &[pid];
        let properties: &[(&str, Value)] = &[
            ("PIDs", Value::new(pids)),
            ("CollectMode", Value::new("inactive-or-failed")),
        ];
        let aux: &[(&str, &[(&str, Value)])] = &[];

        let body = (scope_name, "fail", properties, aux);
        let mut call =
            std::pin::pin!(proxy.call::<_, _, OwnedObjectPath>("StartTransientUnit", &body));
        let mut early_result: Option<(OwnedObjectPath, String)> = None;

        let job = loop {
            match select(call.as_mut(), signals.next()).await {
                Either::Left((reply, _)) => {
                    break reply.context("error calling StartTransientUnit")?;
                }
                Either::Right((Some(message), _)) => {
                    let body = message.body();
                    let body: (u32, OwnedObjectPath, &str, &str) =
                        body.deserialize().context("error parsing signal")?;
                    if body.2 == scope_name {
                        early_result = Some((body.1, body.3.to_owned()));
                    }
                }
                Either::Right((None, _)) => {
                    anyhow::bail!("lost the systemd connection before the scope started");
                }
            }
        };

        if let Some((removed_job, result)) = early_result {
            if removed_job == job {
                anyhow::ensure!(result == "done", "systemd scope job ended with {result}");
                return Ok(());
            }
        }

        trace!("waiting for JobRemoved");
        while let Some(message) = signals.next().await {
            let body = message.body();
            let body: (u32, OwnedObjectPath, &str, &str) =
                body.deserialize().context("error parsing signal")?;

            if body.1 == job {
                anyhow::ensure!(body.3 == "done", "systemd scope job ended with {}", body.3);
                return Ok(());
            }
        }

        anyhow::bail!("lost the systemd connection before the scope started")
    }
}
