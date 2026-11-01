//! Private filesystem, compositor and session bus, or explicit desktop-session discovery.

use std::env;
use std::fs::{self, DirBuilder};
use std::io::{BufRead, BufReader};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};

use crate::cli::HeadlessSize;

pub(super) fn isolate(size: HeadlessSize) -> Result<()> {
    let home = env::var_os("HOME").context("HOME is required for filesystem isolation")?;
    let home = fs::canonicalize(home).context("error resolving HOME")?;
    ensure!(
        home != Path::new("/"),
        "HOME must not be the filesystem root"
    );
    let workspace = env::current_dir()?;
    let mut command = Command::new(option_env!("NIRI_MCP_BWRAP").unwrap_or("bwrap"));
    command
        .args([
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--tmpfs",
            "/run",
            "--ro-bind-try",
            "/run/current-system",
            "/run/current-system",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
        ])
        .arg(&home);
    // Binding an ancestor of HOME would expose the real home over the private mount.
    if !home.starts_with(&workspace)
        && workspace != Path::new("/tmp")
        && !["/run", "/dev", "/proc", "/sys"]
            .iter()
            .any(|root| workspace.starts_with(root))
    {
        command.arg("--bind").arg(&workspace).arg(&workspace);
        command.arg("--chdir").arg(&workspace);
    } else {
        command.arg("--chdir").arg(&home);
    }
    // Keep locally installed binaries usable when the private home hides their original path.
    command
        .arg("--ro-bind")
        .arg(env::current_exe()?)
        .arg("/tmp/niri-mcp");
    // Stdin EOF ends the session. --die-with-parent also reacts to a spawning
    // thread exiting, which kills healthy sessions launched by a thread pool.
    command
        .args([
            "--unshare-pid",
            "--proc",
            "/proc",
            "--unshare-ipc",
            "--new-session",
            "--clearenv",
            "--setenv",
            "HOME",
        ])
        .arg(&home);
    let mut setenv = |name: &str, value: &std::ffi::OsStr| {
        command.args(["--setenv", name]).arg(value);
    };
    setenv(
        "PATH",
        &env::var_os("PATH").unwrap_or_else(|| "/usr/local/bin:/usr/bin:/bin".into()),
    );
    setenv(
        "LANG",
        &env::var_os("LANG").unwrap_or_else(|| "C.UTF-8".into()),
    );
    let mut data_dirs = Vec::new();
    if let Some(path) = option_env!("NIRI_MCP_DATA_DIR") {
        data_dirs.push(std::path::PathBuf::from(path));
    }
    data_dirs.extend(env::split_paths(
        &env::var_os("XDG_DATA_DIRS").unwrap_or_else(|| "/usr/local/share:/usr/share".into()),
    ));
    setenv("XDG_DATA_DIRS", &env::join_paths(data_dirs)?);
    for (name, value) in [
        ("GDK_BACKEND", "wayland"),
        ("QT_QPA_PLATFORM", "wayland"),
        ("ELECTRON_OZONE_PLATFORM_HINT", "wayland"),
        ("LIBGL_ALWAYS_SOFTWARE", "1"),
        ("NO_AT_BRIDGE", "0"),
        ("ACCESSIBILITY_ENABLED", "1"),
        ("ATSPI_DBUS_IMPLEMENTATION", "dbus-daemon"),
        ("DO_NOT_TRACK", "1"),
    ] {
        setenv(name, value.as_ref());
    }
    if let Some(path) = env::var_os("__EGL_VENDOR_LIBRARY_FILENAMES")
        .or_else(|| option_env!("NIRI_MCP_EGL_VENDOR").map(Into::into))
    {
        setenv("__EGL_VENDOR_LIBRARY_FILENAMES", &path);
    }
    // Vulkan selects its driver independently of LIBGL_ALWAYS_SOFTWARE.
    let vulkan_driver = option_env!("NIRI_MCP_VULKAN_DRIVERS")
        .and_then(|directory| fs::read_dir(directory).ok())
        .and_then(|entries| {
            entries.flatten().find(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("lvp_icd.") && name.ends_with(".json"))
            })
        });
    if let Some(driver) = vulkan_driver {
        setenv("VK_DRIVER_FILES", driver.path().as_os_str());
    }
    // NixOS exposes its configured trust store here. Other systems use their library defaults.
    if Path::new("/etc/ssl/certs/ca-certificates.crt").exists() {
        setenv(
            "SSL_CERT_FILE",
            "/etc/ssl/certs/ca-certificates.crt".as_ref(),
        );
    }
    let error = command
        .arg("--")
        .arg("/tmp/niri-mcp")
        .args(["mcp", "--headless"])
        .arg(format!("{}x{}", size.0, size.1))
        .exec();
    Err(error).context("error starting Bubblewrap for the isolated session")
}

pub(super) fn desktop_environment() -> Result<()> {
    let names = ["NIRI_SOCKET", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"];
    if names
        .iter()
        .all(|name| env::var_os(name).is_some_and(|v| !v.is_empty()))
    {
        return Ok(());
    }
    let output = Command::new(option_env!("NIRI_MCP_SYSTEMCTL").unwrap_or("systemctl"))
        .args(["--user", "show-environment"]).output()
        .context("error discovering desktop session, set NIRI_SOCKET, WAYLAND_DISPLAY and DBUS_SESSION_BUS_ADDRESS explicitly")?;
    ensure!(
        output.status.success(),
        "error discovering desktop session: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    for line in String::from_utf8(output.stdout)?.lines() {
        if let Some((name, value)) = line.split_once('=') {
            if names.contains(&name) && env::var_os(name).is_none_or(|v| v.is_empty()) {
                env::set_var(name, value);
            }
        }
    }
    Ok(())
}

struct Process(Child);

impl Process {
    fn spawn(command: &mut Command) -> Result<Self> {
        // The MCP process blocks termination signals before creating D-Bus worker threads.
        unsafe {
            command.pre_exec(crate::utils::signals::unblock_all);
        }
        Ok(Self(command.spawn()?))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) fn run(size: HeadlessSize, input: impl BufRead + Send + 'static) -> Result<()> {
    let directory = env::temp_dir().join(format!("niri-{:016x}", fastrand::u64(..)));
    DirBuilder::new().mode(0o700).create(&directory)?;
    let directory = super::CaptureDir(directory);
    let config = directory.0.join("config.kdl");
    fs::write(
        &config,
        r#"
layout { gaps 8; default-column-width { proportion 1.0; }; }
prefer-no-csd
hotkey-overlay { skip-at-startup; }
animations { off; }
xwayland-satellite { off; }
"#,
    )?;
    for name in [
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "WAYLAND_SOCKET",
        "NIRI_SOCKET",
        "AT_SPI_BUS_ADDRESS",
        "DBUS_STARTER_ADDRESS",
        "DBUS_STARTER_BUS_TYPE",
        "DBUS_SESSION_BUS_ADDRESS",
    ] {
        env::remove_var(name);
    }
    env::set_var("XDG_RUNTIME_DIR", &directory.0);
    env::set_var("XDG_CURRENT_DESKTOP", "niri");
    env::set_var("XDG_SESSION_TYPE", "wayland");
    let mut bus = Process::spawn(
        Command::new(option_env!("NIRI_MCP_DBUS_DAEMON").unwrap_or("dbus-daemon"))
            .args(["--session", "--nofork", "--print-address=1"])
            .env("XDG_RUNTIME_DIR", &directory.0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped()),
    )
    .context("error starting private session bus")?;
    let mut address = String::new();
    BufReader::new(bus.0.stdout.take().unwrap()).read_line(&mut address)?;
    ensure!(
        !address.trim().is_empty(),
        "private session bus did not provide an address"
    );
    env::set_var("DBUS_SESSION_BUS_ADDRESS", address.trim());
    let mut compositor = Process::spawn(
        Command::new(env::current_exe()?)
            .arg("--headless")
            .arg(format!("{}x{}", size.0, size.1))
            .arg("--config")
            .arg(config)
            .env_remove("NOTIFY_SOCKET")
            .env("NIRI_DISABLE_SYSTEM_MANAGER_NOTIFY", "1")
            .env("NIRI_AGENT_LAUNCH", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
    .context("error starting private compositor")?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = compositor.0.try_wait()? {
            bail!("private compositor exited: {status}");
        }
        let mut wayland = None;
        let mut ipc = None;
        for entry in fs::read_dir(&directory.0)? {
            let entry = entry?;
            if !entry.file_type()?.is_socket() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("wayland-") {
                wayland = Some(entry.file_name());
            }
            if name.starts_with("niri.") && name.ends_with(".sock") {
                ipc = Some(entry.path());
            }
        }
        if let (Some(wayland), Some(ipc)) = (wayland, ipc) {
            env::set_var("WAYLAND_DISPLAY", wayland);
            env::set_var("NIRI_SOCKET", ipc);
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "private compositor did not become ready"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    #[cfg(feature = "dbus")]
    super::accessibility::bounded(async {
        let connection = zbus::Connection::session().await?;
        let environment: std::collections::HashMap<_, _> = [
            "WAYLAND_DISPLAY",
            "NIRI_SOCKET",
            "XDG_RUNTIME_DIR",
            "XDG_CURRENT_DESKTOP",
        ]
        .into_iter()
        .map(|name| (name.to_owned(), env::var(name).unwrap()))
        .collect();
        let bus = zbus::fdo::DBusProxy::new(&connection).await?;
        bus.update_activation_environment(
            environment
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect(),
        )
        .await?;
        super::accessibility::enable().await
    })?;
    super::serve(input, false)
}
