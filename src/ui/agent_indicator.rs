use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::time::Duration;

use ordered_float::NotNan;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::output::Output;
use smithay::utils::{Rectangle, Transform};

use crate::animation::{Animation, Clock, Curve};
use crate::render_helpers::primary_gpu_texture::PrimaryGpuTextureRenderElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::texture::{TextureBuffer, TextureRenderElement};
use crate::utils::output_size;

const RADIUS: f64 = 20.;
const WIDTH: f64 = 3.;
const GLOW: f64 = 3.5;
const COLOR: [u8; 3] = [0xc0, 0x84, 0xfc];
const PULSE_PERIOD: f64 = 3.;

pub struct AgentIndicator {
    sessions: usize,
    animation: Option<Animation>,
    started_at: Duration,
    clock: Clock,
    buffers: RefCell<HashMap<NotNan<f64>, Option<Buffers>>>,
}

struct Buffers {
    corners: [TextureBuffer<GlesTexture>; 4],
    masks: [TextureBuffer<GlesTexture>; 4],
    edges: [TextureBuffer<GlesTexture>; 4],
}

impl AgentIndicator {
    pub fn new(clock: Clock) -> Self {
        Self {
            sessions: 0,
            animation: None,
            started_at: Duration::ZERO,
            clock,
            buffers: RefCell::new(HashMap::new()),
        }
    }

    fn opacity(&self) -> f64 {
        self.animation
            .as_ref()
            .map_or(f64::from(self.sessions > 0), Animation::value)
    }

    pub fn session_changed(&mut self, connected: bool) -> bool {
        let was_active = self.sessions > 0;
        let from = self.opacity();
        if connected {
            self.sessions += 1;
        } else {
            self.sessions -= 1;
        }
        let active = self.sessions > 0;
        if was_active == active {
            return false;
        }
        // Preserve the pulse phase when reversing an unfinished fade-out.
        if active && from == 0. {
            self.started_at = self.clock.now();
        }
        let to = f64::from(active);
        self.animation = (from != to).then(|| {
            Animation::ease(
                self.clock.clone(),
                from,
                to,
                0.,
                if active { 150 } else { 200 },
                Curve::EaseOutCubic,
            )
        });
        true
    }

    pub fn advance_animations(&mut self) {
        if self.animation.as_ref().is_some_and(Animation::is_done) {
            self.animation = None;
        }
    }

    pub fn are_animations_ongoing(&self) -> bool {
        self.animation.is_some() || (self.sessions > 0 && !self.clock.should_complete_instantly())
    }

    pub fn render<R: NiriRenderer>(
        &self,
        renderer: &mut R,
        output: &Output,
        push: &mut dyn FnMut(PrimaryGpuTextureRenderElement),
    ) {
        let opacity = self.opacity() as f32;
        if opacity == 0. {
            return;
        }
        let pulse = if self.clock.should_complete_instantly() {
            1.
        } else {
            let elapsed = self
                .clock
                .now()
                .saturating_sub(self.started_at)
                .as_secs_f64();
            (0.88 + 0.12 * (elapsed * TAU / PULSE_PERIOD).cos()) as f32
        };
        let scale = output.current_scale().fractional_scale();
        let mut buffers = self.buffers.borrow_mut();
        let buffers = buffers
            .entry(NotNan::new(scale).unwrap())
            .or_insert_with(|| {
                Buffers::new(renderer.as_gles_renderer(), scale)
                    .inspect_err(|err| warn!("error rendering agent indicator: {err:?}"))
                    .ok()
            });
        let Some(buffers) = buffers else { return };
        let size = output_size(output);
        let r = buffers.corners[0]
            .logical_size()
            .w
            .min(size.w / 2.)
            .min(size.h / 2.);
        let corners = [
            (0., 0.),
            (size.w - r, 0.),
            (size.w - r, size.h - r),
            (0., size.h - r),
        ];
        let edges = [
            Rectangle::new((r, 0.).into(), (size.w - 2. * r, r).into()),
            Rectangle::new((size.w - r, r).into(), (r, size.h - 2. * r).into()),
            Rectangle::new((r, size.h - r).into(), (size.w - 2. * r, r).into()),
            Rectangle::new((0., r).into(), (r, size.h - 2. * r).into()),
        ];
        let mut draw = |buffer: &TextureBuffer<GlesTexture>, rect: Rectangle<f64, _>, alpha| {
            if rect.size.w > 0. && rect.size.h > 0. {
                push(PrimaryGpuTextureRenderElement(
                    TextureRenderElement::from_texture_buffer(
                        buffer.clone(),
                        rect.loc,
                        alpha,
                        None,
                        Some(rect.size),
                        Kind::Unspecified,
                    ),
                ));
            }
        };
        for (buffer, location) in buffers.corners.iter().zip(corners) {
            draw(
                buffer,
                Rectangle::new(location.into(), (r, r).into()),
                opacity * pulse,
            );
        }
        for (buffer, rect) in buffers.edges.iter().zip(edges) {
            draw(buffer, rect, opacity * pulse);
        }
        // The corner masks fade with visibility, but never breathe with the violet light.
        for (buffer, location) in buffers.masks.iter().zip(corners) {
            draw(
                buffer,
                Rectangle::new(location.into(), (r, r).into()),
                opacity,
            );
        }
    }
}

impl Buffers {
    fn new(renderer: &mut GlesRenderer, scale: f64) -> anyhow::Result<Self> {
        let pixels = (RADIUS * scale).round().max(1.) as i32;
        let radius = f64::from(pixels) / scale;
        let width = (WIDTH * scale).round().max(1.) / scale;
        let mut corner = Vec::new();
        let mut mask = Vec::new();
        let mut edge = Vec::new();
        let coverage = |distance: f64| (distance * scale + 0.5).clamp(0., 1.);
        let light = |distance: f64| {
            let stroke = coverage(width - distance);
            let glow = 0.28 * (-0.5 * ((distance - width).max(0.) / GLOW).powi(2)).exp();
            coverage(distance) * (stroke + (1. - stroke) * glow)
        };
        let rgba = |alpha: f64| {
            let [r, g, b] = COLOR.map(|c| (f64::from(c) * alpha).round() as u8);
            [r, g, b, (255. * alpha).round() as u8]
        };
        for y in 0..pixels {
            let y = (f64::from(y) + 0.5) / scale;
            edge.extend_from_slice(&rgba(light(y)));
            for x in 0..pixels {
                let x = (f64::from(x) + 0.5) / scale;
                let distance = radius - (radius - x).hypot(radius - y);
                corner.extend_from_slice(&rgba(light(distance)));
                mask.extend_from_slice(&[
                    0,
                    0,
                    0,
                    (255. * (1. - coverage(distance))).round() as u8,
                ]);
            }
        }
        let mut import = |data: &[u8], size, transforms: [Transform; 4]| {
            let texture = TextureBuffer::from_memory(
                renderer,
                data,
                Fourcc::Abgr8888,
                size,
                false,
                scale,
                Transform::Normal,
                Vec::new(),
            )?;
            // Distinct IDs let the damage tracker follow each piece independently.
            Ok::<_, anyhow::Error>(transforms.map(|transform| {
                TextureBuffer::from_texture(
                    renderer,
                    texture.texture().clone(),
                    scale,
                    transform,
                    Vec::new(),
                )
            }))
        };
        let transforms = [
            Transform::Normal,
            Transform::Flipped,
            Transform::_180,
            Transform::Flipped180,
        ];
        Ok(Self {
            corners: import(&corner, (pixels, pixels), transforms)?,
            masks: import(&mask, (pixels, pixels), transforms)?,
            edges: import(
                &edge,
                (1, pixels),
                [
                    Transform::Normal,
                    Transform::_90,
                    Transform::_180,
                    Transform::_270,
                ],
            )?,
        })
    }
}
