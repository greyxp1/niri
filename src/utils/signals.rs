//! We set a signal handler with `calloop::signals::Signals::new`.
//! This does two things:
//! 1. It blocks the thread from receiving these signals normally (pthread_sigmask)
//! 2. It creates a signalfd to read them in the event loop.
//!
//! When spawning children, calloop already deals with the signalfd.
//! `Signals::new` creates it with CLOEXEC, so it will not be inherited by children.
//!
//! But, the sigmask is always inherited, so we want to clear it before spawning children.
//! That way, we don't affect their normal signal handling.
//!
//! In particular, if a child doesn't care about signals, we must not block it from receiving them.
//!
//! The spawn helper clears the inherited mask before executing an app.
//!
//! Technically, a "more correct" solution would be to remember the original sigmask and restore it
//! after the child exits, but that's painful *and* likely to cause issues, because the user almost
//! never intended to spawn niri with a nonempty sigmask. It indicates a bug in whoever spawned us,
//! so we may as well clean up after them (which is easier than not doing so).

pub use platform::*;

pub use crate::spawn_helper::unblock_all;

#[cfg(not(target_os = "linux"))]
mod platform {
    use std::io;

    // FIXME: implement for FreeBSD. But probably, that should be done in calloop::signals.
    pub fn listen(_handle: &calloop::LoopHandle<crate::niri::State>) {}

    // Without signal handling in listen(), blocking termination signals would leave no handler.
    pub fn block_early() -> io::Result<()> {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::io;

    use crate::spawn_helper::{empty_sigset, set_sigmask};

    pub fn termination_fd() -> io::Result<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        let mask = preferred_sigset()?;
        set_sigmask(&mask)?;
        let fd = unsafe { libc::signalfd(-1, &mask, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: signalfd returned a new, owned descriptor.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
    }

    pub fn listen(handle: &calloop::LoopHandle<crate::niri::State>) {
        use calloop::signals::{Signal, Signals};

        handle
            .insert_source(
                Signals::new(&[Signal::SIGINT, Signal::SIGTERM, Signal::SIGHUP]).unwrap(),
                |event, _, state| {
                    info!("quitting due to receiving signal {:?}", event.signal());
                    state.niri.stop_signal.stop();
                },
            )
            .unwrap();
    }

    // We block the signals early, so that they apply to all threads.
    // They are then blocked *again* by the `Signals` source. That's fine.
    pub fn block_early() -> io::Result<()> {
        set_sigmask(&preferred_sigset()?)
    }

    fn preferred_sigset() -> io::Result<libc::sigset_t> {
        let mut set = empty_sigset()?;
        unsafe {
            add_signal(&mut set, libc::SIGINT)?;
            add_signal(&mut set, libc::SIGTERM)?;
            add_signal(&mut set, libc::SIGHUP)?;
        }
        Ok(set)
    }

    // SAFETY: `signum` must be a valid signal number.
    unsafe fn add_signal(set: &mut libc::sigset_t, signum: libc::c_int) -> io::Result<()> {
        if unsafe { libc::sigaddset(set, signum) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}
