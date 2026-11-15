//! Virtual outputs with an optional surfaceless renderer. Tests can omit the renderer.

use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use niri_config::OutputName;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Fourcc;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::native::EGLSurfacelessDisplay;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::RenderElementStates;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::backend::renderer::{Bind, ImportDma};
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
use smithay::utils::{Physical, Size};
use smithay::wayland::dmabuf::DmabufGlobal;
use smithay::wayland::presentation::Refresh;

use super::{IpcOutputMap, OutputId, RenderResult};
use crate::niri::{Niri, RedrawState};
use crate::render_helpers::{create_texture, resources, shaders, RenderCtx, RenderTarget};
use crate::utils::{get_monotonic_time, logical_output};

pub struct Headless {
    renderer: Option<GlesRenderer>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
    frames: HashMap<Output, VirtualFrame>,
    dmabuf_global: Option<DmabufGlobal>,
}

struct VirtualFrame {
    size: Size<i32, Physical>,
    texture: GlesTexture,
    damage: OutputDamageTracker,
}

impl Headless {
    pub fn new() -> Self {
        Self {
            renderer: None,
            ipc_outputs: Default::default(),
            frames: HashMap::new(),
            dmabuf_global: None,
        }
    }

    pub fn init(&mut self, _niri: &mut Niri) {}

    pub fn start(&mut self, niri: &mut Niri, size: (u16, u16)) -> anyhow::Result<()> {
        self.add_renderer()?;
        let formats = self.renderer.as_ref().unwrap().dmabuf_formats();
        if formats.iter().next().is_some() {
            let global = niri
                .dmabuf_state
                .create_global::<crate::niri::State>(&niri.display_handle, formats);
            assert!(self.dmabuf_global.replace(global).is_none());
        }
        self.add_output(niri, 1, size);
        Ok(())
    }

    pub fn add_renderer(&mut self) -> anyhow::Result<()> {
        if self.renderer.is_some() {
            error!("add_renderer: renderer must not already exist");
            return Ok(());
        }

        let mut renderer = unsafe {
            let display =
                EGLDisplay::new(EGLSurfacelessDisplay).context("error creating EGL display")?;
            let context = EGLContext::new(&display).context("error creating EGL context")?;
            GlesRenderer::new(context).context("error creating renderer")?
        };

        resources::init(&mut renderer);
        shaders::init(&mut renderer);

        self.renderer = Some(renderer);
        Ok(())
    }

    pub fn add_output(&mut self, niri: &mut Niri, n: u8, size: (u16, u16)) {
        let connector = format!("headless-{n}");
        let make = "niri".to_string();
        let model = "headless".to_string();
        let serial = n.to_string();

        let output = Output::new(
            connector.clone(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: make.clone(),
                model: model.clone(),
                serial_number: serial.clone(),
            },
        );

        let mode = Mode {
            size: Size::from((i32::from(size.0), i32::from(size.1))),
            refresh: 60_000,
        };
        output.change_current_state(Some(mode), None, None, None);
        output.set_preferred(mode);

        output.user_data().insert_if_missing(|| OutputName {
            connector,
            make: Some(make),
            model: Some(model),
            serial: Some(serial),
        });

        let physical_properties = output.physical_properties();
        self.ipc_outputs.lock().unwrap().insert(
            OutputId::next(),
            niri_ipc::Output {
                name: output.name(),
                make: physical_properties.make,
                model: physical_properties.model,
                serial: None,
                physical_size: None,
                modes: vec![niri_ipc::Mode {
                    width: size.0,
                    height: size.1,
                    refresh_rate: 60_000,
                    is_preferred: true,
                }],
                current_mode: Some(0),
                is_custom_mode: true,
                vrr_supported: false,
                vrr_enabled: false,
                logical: Some(logical_output(&output)),
                max_bpc: None,
            },
        );

        niri.add_output(output, None, false);
    }

    pub fn seat_name(&self) -> String {
        "headless".to_owned()
    }

    pub fn with_primary_renderer<T>(
        &mut self,
        f: impl FnOnce(&mut GlesRenderer) -> T,
    ) -> Option<T> {
        self.renderer.as_mut().map(f)
    }

    pub fn primary_render_node(&mut self) -> Option<DrmNode> {
        None
    }

    pub fn render(&mut self, niri: &mut Niri, output: &Output) -> RenderResult {
        let states = if let Some(renderer) = &mut self.renderer {
            let rendered = (|| -> anyhow::Result<_> {
                let size = output
                    .current_mode()
                    .context("virtual output has no mode")?
                    .size;
                if !self
                    .frames
                    .get(output)
                    .is_some_and(|frame| frame.size == size)
                {
                    self.frames.insert(
                        output.clone(),
                        VirtualFrame {
                            size,
                            texture: create_texture(renderer, size, Fourcc::Abgr8888)?,
                            damage: OutputDamageTracker::from_output(output),
                        },
                    );
                }
                let elements = niri.render_to_vec(
                    RenderCtx {
                        renderer,
                        target: RenderTarget::Output,
                        xray: None,
                    },
                    output,
                    true,
                );
                let frame = self.frames.get_mut(output).unwrap();
                let mut target = renderer.bind(&mut frame.texture)?;
                // The same offscreen buffer retains the previous frame's undamaged pixels.
                let result =
                    frame
                        .damage
                        .render_output(renderer, &mut target, 1, &elements, [0.; 4])?;
                Ok(result.states)
            })();
            match rendered {
                Ok(states) => {
                    niri.update_primary_scanout_output(output, &states);
                    states
                }
                Err(error) => {
                    warn!("error rendering virtual output: {error:#}");
                    return RenderResult::Skipped;
                }
            }
        } else {
            RenderElementStates::default()
        };
        let mut presentation_feedbacks = niri.take_presentation_feedbacks(output, &states);
        presentation_feedbacks.presented::<_, smithay::utils::Monotonic>(
            get_monotonic_time(),
            Refresh::Unknown,
            0,
            wp_presentation_feedback::Kind::empty(),
        );

        let output_state = niri.output_state.get_mut(output).unwrap();
        match mem::replace(&mut output_state.redraw_state, RedrawState::Idle) {
            RedrawState::Idle => unreachable!(),
            RedrawState::Queued => (),
            RedrawState::WaitingForVBlank { .. } => unreachable!(),
            RedrawState::WaitingForEstimatedVBlank(_) => unreachable!(),
            RedrawState::WaitingForEstimatedVBlankAndQueued(_) => unreachable!(),
        }

        output_state.frame_callback_sequence = output_state.frame_callback_sequence.wrapping_add(1);

        if self.renderer.is_some() {
            let interval = Duration::from_nanos(1_000_000_000 / 60);
            output_state.frame_clock.presented(get_monotonic_time());
            output_state.redraw_state = RedrawState::WaitingForVBlank {
                redraw_needed: false,
            };
            let output = output.clone();
            niri.event_loop
                .insert_source(Timer::from_duration(interval), move |_, _, state| {
                    if let Some(output_state) = state.niri.output_state.get_mut(&output) {
                        if let RedrawState::WaitingForVBlank { redraw_needed } =
                            mem::replace(&mut output_state.redraw_state, RedrawState::Idle)
                        {
                            if redraw_needed || output_state.unfinished_animations_remain {
                                output_state.redraw_state = RedrawState::Queued;
                            }
                        }
                    }
                    TimeoutAction::Drop
                })
                .unwrap();
        }

        RenderResult::Submitted
    }

    pub fn import_dmabuf(&mut self, dmabuf: &Dmabuf) -> bool {
        self.renderer
            .as_mut()
            .is_some_and(|renderer| renderer.import_dmabuf(dmabuf, None).is_ok())
    }

    pub fn ipc_outputs(&self) -> Arc<Mutex<IpcOutputMap>> {
        self.ipc_outputs.clone()
    }
}

impl Default for Headless {
    fn default() -> Self {
        Self::new()
    }
}
