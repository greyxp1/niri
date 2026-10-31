use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::AsFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

use libc::rlim_t;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct SpawnRequest {
    pub(crate) program: OsString,
    pub(crate) args: Vec<OsString>,
    pub(crate) env: Vec<(OsString, Option<OsString>)>,
    pub(crate) nofile_soft: rlim_t,
    pub(crate) nofile_hard: rlim_t,
}

pub fn spawn_child() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    unsafe {
        // Set kernel comm name so `ps -e` / `top` show niri-spawn-v1 instead of the /proc/self/fd
        // number while idle. execve() overwrites this with the target binary name.
        libc::prctl(libc::PR_SET_NAME, c"niri-spawn-v1".as_ptr(), 0, 0, 0);
    }
    unblock_all()?;

    let mut data = Vec::new();
    io::stdin().lock().read_to_end(&mut data)?;
    if data.is_empty() {
        // Niri exited or discarded a pre-warmed helper before sending a command.
        return Ok(());
    }

    let request: SpawnRequest = serde_json::from_slice(&data)?;

    let mut process = Command::new(&request.program);
    process
        .args(request.args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    for (key, value) in request.env {
        if let Some(value) = value {
            process.env(key, value);
        } else {
            process.env_remove(key);
        }
    }

    if request.nofile_soft != 0 {
        let rlim = libc::rlimit {
            rlim_cur: request.nofile_soft,
            rlim_max: request.nofile_hard,
        };
        // Command::exec opens the stdio descriptors before running this callback.
        unsafe {
            process.pre_exec(move || {
                libc::setrlimit(libc::RLIMIT_NOFILE, &rlim);
                Ok(())
            });
        }
    }

    // exec replaces stderr before attempting to start the app. Keep a CLOEXEC copy for failures.
    let mut stderr = File::from(io::stderr().as_fd().try_clone_to_owned()?);
    let err = process.exec();
    writeln!(stderr, "could not spawn {:?}: {err}", request.program)?;
    std::process::exit(1);
}

pub fn unblock_all() -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        set_sigmask(&empty_sigset()?)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(crate) fn empty_sigset() -> io::Result<libc::sigset_t> {
    // Some libc implementations only initialize the part of sigset_t they use.
    let mut set = std::mem::MaybeUninit::zeroed();
    if unsafe { libc::sigemptyset(set.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { set.assume_init() })
}

#[cfg(target_os = "linux")]
pub(crate) fn set_sigmask(set: &libc::sigset_t) -> io::Result<()> {
    let result = unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, set, std::ptr::null_mut()) };
    if result != 0 {
        return Err(io::Error::from_raw_os_error(result));
    }
    Ok(())
}
