use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::utils::{Logical, Rectangle};
use smithay::wayland::compositor::{add_post_commit_hook, with_states, SurfaceData};

use super::raw::zgrey_blur::server::zgrey_blur_manager_v1::{self, ZgreyBlurManagerV1};
use super::raw::zgrey_blur::server::zgrey_blur_surface_v1::{self, ZgreyBlurSurfaceV1};
use crate::niri::State;

type Rectangles = Arc<Vec<Rectangle<i32, Logical>>>;
type PendingRegion = (u32, u32, Rectangle<i32, Logical>);

#[derive(Debug, Clone)]
pub struct BlurGroup {
    pub strength: u32,
    pub regions: Vec<(u32, Rectangles)>,
}

#[derive(Default)]
struct SurfaceBlur(Mutex<SurfaceBlurInner>);

#[derive(Default)]
struct SurfaceBlurInner {
    attached: bool,
    hook_registered: bool,
    pending: Option<Vec<PendingRegion>>,
    current: Arc<Vec<BlurGroup>>,
}

pub fn current(states: &SurfaceData) -> Arc<Vec<BlurGroup>> {
    states
        .data_map
        .get::<SurfaceBlur>()
        .map(|data| data.0.lock().unwrap().current.clone())
        .unwrap_or_default()
}

pub fn init(display: &DisplayHandle) {
    display.create_global::<State, ZgreyBlurManagerV1, _>(1, ());
}

impl GlobalDispatch<ZgreyBlurManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<ZgreyBlurManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZgreyBlurManagerV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &ZgreyBlurManagerV1,
        request: zgrey_blur_manager_v1::Request,
        _data: &(),
        _handle: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zgrey_blur_manager_v1::Request::Destroy => (),
            zgrey_blur_manager_v1::Request::GetSurface { id, surface } => {
                let register = with_states(&surface, |states| {
                    let data = states
                        .data_map
                        .get_or_insert_threadsafe(SurfaceBlur::default);
                    let mut inner = data.0.lock().unwrap();
                    if inner.attached {
                        return None;
                    }
                    inner.attached = true;
                    let register = !inner.hook_registered;
                    inner.hook_registered = true;
                    Some(register)
                });
                let Some(register) = register else {
                    resource.post_error(
                        zgrey_blur_manager_v1::Error::AlreadyExists,
                        "surface already has a blur object",
                    );
                    return;
                };
                if register {
                    add_post_commit_hook::<State, _>(&surface, |state, _dh, surface| {
                        let changed = with_states(surface, |states| {
                            let data = states.data_map.get::<SurfaceBlur>().unwrap();
                            let mut inner = data.0.lock().unwrap();
                            let Some(pending) = inner.pending.take() else {
                                return false;
                            };
                            let mut groups: BTreeMap<
                                u32,
                                BTreeMap<u32, Vec<Rectangle<i32, Logical>>>,
                            > = BTreeMap::new();
                            for (strength, coverage, rect) in pending {
                                groups
                                    .entry(strength)
                                    .or_default()
                                    .entry(coverage)
                                    .or_default()
                                    .push(rect);
                            }
                            inner.current = Arc::new(
                                groups
                                    .into_iter()
                                    .map(|(strength, regions)| BlurGroup {
                                        strength,
                                        regions: regions
                                            .into_iter()
                                            .map(|(coverage, rectangles)| {
                                                (coverage, Arc::new(rectangles))
                                            })
                                            .collect(),
                                    })
                                    .collect(),
                            );
                            crate::render_helpers::background_effect::damage_surface(states);
                            true
                        });
                        if changed {
                            state.niri.queue_redraw_all();
                        }
                    });
                }
                data_init.init(id, surface);
            }
        }
    }
}

impl Dispatch<ZgreyBlurSurfaceV1, WlSurface> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &ZgreyBlurSurfaceV1,
        request: zgrey_blur_surface_v1::Request,
        surface: &WlSurface,
        _handle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        if !surface.is_alive() {
            return;
        }
        let (regions, replace) = match request {
            zgrey_blur_surface_v1::Request::Destroy => return,
            zgrey_blur_surface_v1::Request::SetRegions { regions } => (regions, true),
            zgrey_blur_surface_v1::Request::AppendRegions { regions } => (regions, false),
        };
        with_states(surface, |states| {
            let data = states.data_map.get::<SurfaceBlur>().unwrap();
            let mut inner = data.0.lock().unwrap();
            if replace {
                inner.pending = Some(Vec::new());
            }
            if inner.pending.is_none() {
                inner.pending = Some(
                    inner
                        .current
                        .iter()
                        .flat_map(|group| {
                            group
                                .regions
                                .iter()
                                .flat_map(move |(coverage, rectangles)| {
                                    rectangles
                                        .iter()
                                        .map(move |rect| (group.strength, *coverage, *rect))
                                })
                        })
                        .collect(),
                );
            }
            let pending = inner.pending.as_mut().unwrap();
            if regions.len() % 24 != 0 || pending.len() + regions.len() / 24 > 65536 {
                resource.post_error(
                    zgrey_blur_surface_v1::Error::InvalidRegions,
                    "invalid region array length",
                );
                return;
            }
            for record in regions.chunks_exact(24) {
                let mut values = record
                    .chunks_exact(4)
                    .map(|v| i32::from_ne_bytes(v.try_into().unwrap()));
                let (x, y, w, h, strength, coverage) = (
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                );
                if w <= 0
                    || h <= 0
                    || x.checked_add(w).is_none()
                    || y.checked_add(h).is_none()
                    || !(1..=64).contains(&strength)
                    || !(1..=16).contains(&coverage)
                {
                    resource.post_error(
                        zgrey_blur_surface_v1::Error::InvalidRegions,
                        "invalid rectangle or strength",
                    );
                    return;
                }
                pending.push((
                    strength as u32,
                    coverage as u32,
                    Rectangle::new((x, y).into(), (w, h).into()),
                ));
            }
        });
    }

    fn destroyed(
        _state: &mut Self,
        _client: wayland_backend::server::ClientId,
        _resource: &ZgreyBlurSurfaceV1,
        surface: &WlSurface,
    ) {
        if surface.is_alive() {
            with_states(surface, |states| {
                let data = states.data_map.get::<SurfaceBlur>().unwrap();
                let mut inner = data.0.lock().unwrap();
                inner.attached = false;
                inner.pending = Some(Vec::new());
            });
        }
    }
}
