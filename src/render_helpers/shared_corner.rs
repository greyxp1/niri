use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use glam::{Mat3, Vec2};
use niri_config::CornerRadius;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{
    build_texture_mat, ffi, GlesError, GlesFrame, GlesRenderer, GlesTexture, Uniform, UniformName,
    UniformType, UniformValue,
};
use smithay::backend::renderer::utils::{CommitCounter, DamageBag, DamageSet};
use smithay::backend::renderer::{Bind, Frame, FrameContext, Offscreen, Renderer, Texture};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Rectangle, Scale, Size, Transform};

use super::renderer::AsGlesFrame;
use super::shader_element::{ShaderProgram, ShaderRenderElement};
use super::shaders::shared_corner_portable::{
    ShaderKey, ATLAS, BACKGROUND, EFFECT, EXTERNAL, FORCE_ALPHA, MAX_SOURCES, NATIVE_ONLY,
    NO_NATIVE, RESIZE, SOLID,
};
use super::shaders::ProgramType;
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};
use crate::utils::region::TransformedRegion;

#[derive(Debug)]
pub struct CornerCache {
    id: Id,
    history: RefCell<History>,
    template: RefCell<Option<(f64, ShaderRenderElement)>>,
    spare: RefCell<Scratch>,
}

#[derive(Debug, Default)]
struct Scratch {
    sources: Vec<Source>,
    textures: Vec<Option<GlesTexture>>,
}

#[derive(Debug, Clone, Copy)]
struct SurfaceState {
    commit: CommitCounter,
    geometry: Rectangle<f64, Logical>,
    alpha: f32,
    epoch: u64,
    order: usize,
}

#[derive(Debug, Default)]
struct History {
    layout: Option<LayoutState>,
    sources: HashMap<Id, SurfaceState>,
    damage: DamageBag<f64, Logical>,
    epoch: u64,
}

#[derive(Debug, PartialEq)]
struct LayoutState {
    geometry: Rectangle<f64, Logical>,
    window: Rectangle<f64, Logical>,
    radius: CornerRadius,
    decorations: Vec<Decoration>,
}

impl Default for CornerCache {
    fn default() -> Self {
        Self {
            id: Id::new(),
            history: Default::default(),
            template: Default::default(),
            spare: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Source {
    texture_units: usize,
    flags: u8,
    alpha: f32,
    rect: [f32; 4],
    tex_x: [f32; 4],
    tex_y: [f32; 4],
    color: [f32; 4],
    postprocess: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decoration {
    pub rect: [f32; 4],
    pub radius: [f32; 4],
    pub color_from: [f32; 4],
    pub color_to: [f32; 4],
    pub grad: [f32; 4],
    pub options: [f32; 4],
    pub behavior: [f32; 4],
}

#[derive(Debug)]
struct NativeDraw {
    source: usize,
    texture: Option<GlesTexture>,
    damage: Vec<Rectangle<i32, Physical>>,
}

/// The pixel regions share a boundary, so their appearance must be composed before
/// applying coverage. Only the existing corner draws resolve that shared coverage.
#[derive(Debug)]
pub struct SharedCorner {
    pub geometry: Rectangle<f64, Logical>,
    pub window: Rectangle<f64, Logical>,
    pub radius: CornerRadius,
    scale: f64,
    pub corners: [Rectangle<f64, Logical>; 4],
    regions: Vec<Rectangle<f64, Logical>>,
    cache: Rc<CornerCache>,
    sources: RefCell<Vec<Source>>,
    textures: RefCell<Vec<Option<GlesTexture>>>,
    decorations: Vec<Decoration>,
    epoch: u64,
    next_surface: Cell<usize>,
    native: RefCell<Option<NativeDraw>>,
    damage: RefCell<Vec<Rectangle<f64, Logical>>>,
    resize: RefCell<Option<ResizeSource>>,
}

#[derive(Debug)]
struct ResizeSource {
    textures: [GlesTexture; 2],
    uniforms: Rc<[Uniform<'static>]>,
    size: [f32; 2],
    alpha: f32,
}

#[derive(Debug)]
pub struct SharedCornerRenderElement {
    inner: ShaderRenderElement,
    shared: Rc<SharedCorner>,
}

impl SharedCorner {
    pub fn new(
        geometry: Rectangle<f64, Logical>,
        window: Rectangle<f64, Logical>,
        radius: CornerRadius,
        corners: [Rectangle<f64, Logical>; 4],
        decorations: Vec<Decoration>,
        cache: Rc<CornerCache>,
        scale: f64,
    ) -> Rc<Self> {
        let radius = radius.fit_to(window.size.w as f32, window.size.h as f32);
        let relative_window = Rectangle::new(window.loc - geometry.loc, window.size);
        let epoch = {
            let mut history = cache.history.borrow_mut();
            let unchanged = history.layout.as_ref().is_some_and(|previous| {
                previous.geometry.size == geometry.size
                    && previous.window == relative_window
                    && previous.radius == radius
                    && previous.decorations == decorations
            });
            if !unchanged {
                history.layout = Some(LayoutState {
                    geometry: Rectangle::from_size(geometry.size),
                    window: relative_window,
                    radius,
                    decorations: decorations.clone(),
                });
                history.damage.add([Rectangle::from_size(geometry.size)]);
                cache.template.borrow_mut().take();
            }
            history.epoch = history.epoch.wrapping_add(1);
            history.epoch
        };
        let scratch = std::mem::take(&mut *cache.spare.borrow_mut());
        let corners = corners.map(|corner| {
            let width = corner.size.w.max(64. / scale).min(geometry.size.w);
            let height = corner.size.h.max(64. / scale).min(geometry.size.h);
            let right = corner.loc.x > geometry.loc.x;
            let bottom = corner.loc.y > geometry.loc.y;
            Rectangle::new(
                (
                    if right {
                        geometry.loc.x + geometry.size.w - width
                    } else {
                        geometry.loc.x
                    },
                    if bottom {
                        geometry.loc.y + geometry.size.h - height
                    } else {
                        geometry.loc.y
                    },
                )
                    .into(),
                (width, height).into(),
            )
        });
        let mut regions = Rectangle::subtract_rects_many([geometry], [window]);
        regions.extend(
            corners
                .iter()
                .filter_map(|corner| corner.intersection(window)),
        );
        Rc::new(Self {
            geometry,
            window,
            radius,
            scale,
            corners,
            regions,
            decorations,
            cache,
            sources: RefCell::new(scratch.sources),
            textures: RefCell::new(scratch.textures),
            epoch,
            next_surface: Cell::new(0),
            native: RefCell::default(),
            resize: RefCell::default(),
            damage: RefCell::default(),
        })
    }

    pub fn background_damage(
        &self,
        geometry: Rectangle<f64, Logical>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
    ) -> Vec<Rectangle<i32, Physical>> {
        let geometry = geometry
            .to_physical_precise_round::<_, i32>(self.scale)
            .to_f64()
            .to_logical(self.scale);
        let excluded = map_damage(&self.regions, geometry, src, dst, damage);
        let interior = map_damage(&[self.window], geometry, src, dst, damage);
        Rectangle::subtract_rects_many(interior, excluded)
    }

    pub fn note_surface(&self, surface: &impl Element, scale: Scale<f64>) -> bool {
        let order = self.next_surface.get();
        self.next_surface.set(order + 1);
        let id = surface.id().clone();
        let geometry = surface.geometry(scale).to_f64().to_logical(scale);
        let geometry = Rectangle::new(geometry.loc - self.geometry.loc, geometry.size);
        let state = SurfaceState {
            commit: surface.current_commit(),
            geometry,
            alpha: surface.alpha(),
            epoch: self.epoch,
            order,
        };
        let mut history = self.cache.history.borrow_mut();
        let previous = history.sources.get(&id).copied();
        let damage = if previous.is_some_and(|previous| {
            previous.geometry == geometry
                && previous.alpha == surface.alpha()
                && previous.order == order
        }) {
            surface
                .damage_since(scale, previous.map(|state| state.commit))
                .into_iter()
                .map(|rect| {
                    let rect = rect.to_f64().to_logical(scale);
                    Rectangle::new(rect.loc + geometry.loc, rect.size)
                })
                .collect::<Vec<_>>()
        } else {
            let mut damage = vec![geometry];
            if let Some(previous) = previous {
                damage.push(previous.geometry);
            }
            damage
        };
        history.sources.insert(id, state);
        if !damage.is_empty() {
            self.damage.borrow_mut().extend(damage);
        }
        order == 0
    }

    fn finish_draw(&self) {
        self.sources.borrow_mut().clear();
        self.textures.borrow_mut().clear();
        self.native.borrow_mut().take();
        self.resize.borrow_mut().take();
    }

    pub fn set_native_damage(
        &self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
    ) {
        let sources = self.sources.borrow();
        let source = sources.len() - 1;
        *self.native.borrow_mut() = Some(NativeDraw {
            source,
            texture: self.textures.borrow()[source].clone(),
            damage: damage
                .iter()
                .map(|rect| Rectangle::new(rect.loc + dst.loc, rect.size))
                .collect(),
        });
    }

    pub fn finish(&self) {
        let mut history = self.cache.history.borrow_mut();
        let mut damage = self.damage.borrow_mut();
        history.sources.retain(|_, state| {
            if state.epoch == self.epoch {
                return true;
            }
            damage.push(state.geometry);
            false
        });
        if !damage.is_empty() {
            history.damage.add(std::mem::take(&mut *damage));
        }
    }

    pub fn add_resize(&self, element: &ShaderRenderElement, scale: Scale<f64>) {
        let geometry = element.geometry(scale).to_f64().to_logical(scale);
        let local = geometry.loc - self.geometry.loc;
        self.sources.borrow_mut().push(Source {
            texture_units: 2,
            flags: RESIZE,
            alpha: element.alpha(),
            rect: [
                local.x as f32,
                local.y as f32,
                geometry.size.w as f32,
                geometry.size.h as f32,
            ],
            tex_x: [
                1. / geometry.size.w as f32,
                0.,
                -local.x as f32 / geometry.size.w as f32,
                0.,
            ],
            tex_y: [
                0.,
                1. / geometry.size.h as f32,
                -local.y as f32 / geometry.size.h as f32,
                0.,
            ],
            color: [0.; 4],
            postprocess: [0.; 4],
        });
        self.textures.borrow_mut().push(None);
        *self.resize.borrow_mut() = Some(ResizeSource {
            textures: ["niri_tex_prev", "niri_tex_next"]
                .map(|name| element.texture(name).unwrap().clone()),
            uniforms: element.uniforms(),
            size: [
                element.geometry(scale).size.w as f32,
                element.geometry(scale).size.h as f32,
            ],
            alpha: element.alpha(),
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_source(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        texture: Option<GlesTexture>,
        color: [f32; 4],
        geometry: Rectangle<f64, Logical>,
        source: Rectangle<f64, Buffer>,
        transform: Transform,
        alpha: f32,
        scale: Scale<f64>,
    ) -> Result<(), GlesError> {
        let geometry = geometry
            .to_physical_precise_round::<_, i32>(scale)
            .to_f64()
            .to_logical(scale);
        let local = geometry.loc - self.geometry.loc;
        let mut descriptor = Source {
            texture_units: 0,
            flags: SOLID,
            alpha,
            rect: [
                local.x as f32,
                local.y as f32,
                geometry.size.w as f32,
                geometry.size.h as f32,
            ],
            tex_x: [0.; 4],
            tex_y: [0.; 4],
            color,
            postprocess: [0., 1., 0., 0.],
        };
        if let Some(texture) = &texture {
            descriptor.flags = u8::from(!texture.has_alpha()) * FORCE_ALPHA
                + u8::from(texture.is_external()) * EXTERNAL;
            descriptor.texture_units = if texture.is_external() {
                frame.with_context(|gl| unsafe {
                    let mut previous = 0;
                    let mut required = 0;
                    gl.GetIntegerv(ffi::TEXTURE_BINDING_EXTERNAL_OES, &mut previous);
                    gl.BindTexture(ffi::TEXTURE_EXTERNAL_OES, texture.tex_id());
                    gl.GetTexParameteriv(ffi::TEXTURE_EXTERNAL_OES, 0x8d68, &mut required);
                    gl.BindTexture(ffi::TEXTURE_EXTERNAL_OES, previous as u32);
                    required.max(1) as usize
                })?
            } else {
                1
            };
            let physical = geometry.to_physical_precise_round(scale);
            let mut matrix = build_texture_mat(source, physical, texture.size(), transform);
            if texture.is_y_inverted() {
                matrix = Mat3::from_translation(Vec2::new(0., 1.))
                    * Mat3::from_scale(Vec2::new(1., -1.))
                    * matrix;
            }
            let offset = self.geometry.loc.to_physical(scale) - physical.loc.to_f64();
            matrix *= Mat3::from_translation(Vec2::new(offset.x as f32, offset.y as f32))
                * Mat3::from_scale(Vec2::new(scale.x as f32, scale.y as f32));
            let array = matrix.to_cols_array_2d();
            descriptor.tex_x = [array[0][0], array[1][0], array[2][0], 0.];
            descriptor.tex_y = [array[0][1], array[1][1], array[2][1], 0.];
        }
        self.textures.borrow_mut().push(texture);
        self.sources.borrow_mut().push(descriptor);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_effect(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        texture: GlesTexture,
        geometry: Rectangle<f64, Logical>,
        source: Rectangle<f64, Buffer>,
        transform: Transform,
        noise: f32,
        saturation: f32,
        background: [f32; 4],
        scale: Scale<f64>,
        subregion: Option<&TransformedRegion>,
    ) -> Result<(), GlesError> {
        let physical = geometry.to_physical_precise_round(scale);
        let mut regions = Vec::new();
        if let Some(subregion) = subregion {
            subregion.filter_damage(
                geometry,
                physical,
                &[Rectangle::from_size(physical.size)],
                &mut regions,
            );
        } else {
            regions.push(Rectangle::from_size(physical.size));
        }
        for region in regions {
            self.add_source(
                frame,
                Some(texture.clone()),
                [0.; 4],
                geometry,
                source,
                transform,
                1.,
                scale,
            )?;
            let mut sources = self.sources.borrow_mut();
            let source = sources.last_mut().unwrap();
            let local = (region.loc + physical.loc).to_f64().to_logical(scale) - self.geometry.loc;
            let size = region.size.to_f64().to_logical(scale);
            source.rect = [local.x as f32, local.y as f32, size.w as f32, size.h as f32];
            source.flags |= EFFECT;
            source.color = background;
            source.postprocess = [noise, saturation, 0., 0.];
        }
        Ok(())
    }

    pub fn element(self: &Rc<Self>, scale: f64) -> SharedCornerRenderElement {
        if let Some((cached_scale, inner)) = &*self.cache.template.borrow() {
            if *cached_scale == scale {
                return SharedCornerRenderElement {
                    inner: inner.clone().with_location(self.geometry.loc),
                    shared: self.clone(),
                };
            }
        }
        let offset = self.window.loc - self.geometry.loc;
        let input = Mat3::from_scale(Vec2::new(
            self.geometry.size.w as f32,
            self.geometry.size.h as f32,
        ));
        let mut columns = Vec::new();
        for index in 0..2 {
            let decoration = self
                .decorations
                .get(index)
                .or_else(|| self.decorations.last())
                .copied()
                .unwrap_or(Decoration {
                    rect: [0.; 4],
                    radius: [0.; 4],
                    color_from: [0.; 4],
                    color_to: [0.; 4],
                    grad: [0.; 4],
                    options: [0.; 4],
                    behavior: [0.; 4],
                });
            columns.extend([
                decoration.rect,
                decoration.radius,
                decoration.color_from,
                decoration.color_to,
                decoration.grad,
                decoration.options,
            ]);
        }
        let matrices = columns
            .chunks_exact(4)
            .map(|columns| std::array::from_fn(|index| columns[index / 4][index % 4]))
            .collect();
        let inner = ShaderRenderElement::new(
            ProgramType::SharedCorner,
            self.geometry.size,
            None,
            scale as f32,
            1.,
            Rc::new([
                super::shaders::mat3_uniform("niri_matrix", input),
                Uniform::new(
                    "niri_window",
                    [
                        offset.x as f32,
                        offset.y as f32,
                        self.window.size.w as f32,
                        self.window.size.h as f32,
                    ],
                ),
                Uniform::new("niri_radius", <[f32; 4]>::from(self.radius)),
                Uniform::new(
                    "niri_decorations",
                    UniformValue::Matrix4x4 {
                        matrices,
                        transpose: false,
                    },
                ),
            ]),
            HashMap::new(),
            Kind::Unspecified,
        )
        .with_location(self.geometry.loc);
        *self.cache.template.borrow_mut() = Some((scale, inner.clone()));
        SharedCornerRenderElement {
            inner,
            shared: self.clone(),
        }
    }
}

impl Drop for SharedCorner {
    fn drop(&mut self) {
        let sources = self.sources.get_mut();
        let textures = self.textures.get_mut();
        sources.clear();
        textures.clear();
        let mut spare = self.cache.spare.borrow_mut();
        if sources.capacity() > spare.sources.capacity() {
            std::mem::swap(sources, &mut spare.sources);
        }
        if textures.capacity() > spare.textures.capacity() {
            std::mem::swap(textures, &mut spare.textures);
        }
    }
}

pub(crate) fn default_key() -> ShaderKey {
    ShaderKey {
        flags: [0; MAX_SOURCES],
        count: 1,
        native: NO_NATIVE,
        native_flags: 0,
        background_mask: 0,
        appearance: false,
    }
}

fn uniform_names(count: usize, resize: bool) -> Vec<UniformName<'static>> {
    let mut names = vec![
        UniformName::new("niri_matrix", UniformType::Matrix3x3),
        UniformName::new("niri_window", UniformType::_4f),
        UniformName::new("niri_radius", UniformType::_4f),
        UniformName::new("niri_decorations", UniformType::Matrix4x4),
        UniformName::new("niri_atlas_mapping", UniformType::_4f),
        UniformName::new("niri_output_matrix", UniformType::Matrix3x3),
        UniformName::new("niri_native_matrix", UniformType::Matrix3x3),
        UniformName::new("niri_native_texture", UniformType::_1i),
    ];
    for index in 0..count {
        for (prefix, kind) in [
            ("niri_rect", UniformType::_4f),
            ("niri_source_matrix", UniformType::Matrix3x3),
            ("niri_color", UniformType::_4f),
            ("niri_post", UniformType::_4f),
            ("niri_source_texture", UniformType::_1i),
        ] {
            names.push(UniformName::new(format!("{prefix}_{index}"), kind));
        }
    }
    if resize {
        names.extend(super::shaders::resize_uniforms());
        names.extend([
            UniformName::new("niri_shared_resize_size", UniformType::_2f),
            UniformName::new("niri_shared_resize_alpha", UniformType::_1f),
            UniformName::new("niri_tex_prev", UniformType::_1i),
            UniformName::new("niri_tex_next", UniformType::_1i),
        ]);
    }
    names
}

pub fn compile(renderer: &mut GlesRenderer) -> Option<ShaderProgram> {
    let (vertex, fragment) = super::shaders::shared_corner_portable::generate(default_key(), None);
    ShaderProgram::compile_with_vertex(renderer, &vertex, &fragment, &uniform_names(1, false))
        .map_err(|err| warn!("error compiling shared coverage shader: {err:?}"))
        .ok()
}

#[derive(Debug, Clone, Copy)]
struct Limits {
    fragment: usize,
    varying: usize,
    textures: usize,
    max_texture: i32,
}

impl Limits {
    fn get(frame: &mut GlesFrame<'_, '_>) -> Result<Self, GlesError> {
        if let Some(limits) = frame.local_user_data().get::<Self>() {
            return Ok(*limits);
        }
        let limits = frame.with_context(|gl| unsafe {
            let mut fragment = 0;
            let mut varying = 0;
            let mut textures = 0;
            let mut max_texture = 0;
            gl.GetIntegerv(ffi::MAX_FRAGMENT_UNIFORM_VECTORS, &mut fragment);
            gl.GetIntegerv(ffi::MAX_VARYING_VECTORS, &mut varying);
            gl.GetIntegerv(ffi::MAX_TEXTURE_IMAGE_UNITS, &mut textures);
            gl.GetIntegerv(ffi::MAX_TEXTURE_SIZE, &mut max_texture);
            Self {
                fragment: fragment as usize,
                varying: varying as usize,
                textures: textures as usize,
                max_texture,
            }
        })?;
        frame.local_user_data().insert_if_missing(|| limits);
        Ok(limits)
    }
    fn direct(self, count: usize, effects: bool) -> bool {
        count <= MAX_SOURCES
            && 14 + count * 2 + usize::from(effects) <= self.fragment
            && count * 4 + 3 + usize::from(effects) * 2 <= self.varying * 4
    }
}

fn source_matrix(source: &Source) -> Mat3 {
    Mat3::from_cols_array(&[
        source.tex_x[0],
        source.tex_y[0],
        0.,
        source.tex_x[1],
        source.tex_y[1],
        0.,
        source.tex_x[2],
        source.tex_y[2],
        1.,
    ])
}

impl Element for SharedCornerRenderElement {
    fn id(&self) -> &Id {
        &self.shared.cache.id
    }
    fn current_commit(&self) -> CommitCounter {
        self.shared.cache.history.borrow().damage.current_commit()
    }
    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }
    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
    }
    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        let history = self.shared.cache.history.borrow();
        let Some(damage) = history.damage.damage_since(commit) else {
            return self
                .shared
                .regions
                .iter()
                .map(|rect| {
                    let mut rect = rect.to_physical_precise_up(scale);
                    rect.loc -= self.geometry(scale).loc;
                    rect
                })
                .collect();
        };
        damage
            .into_iter()
            .flat_map(|rect| {
                self.shared.regions.iter().filter_map(move |corner| {
                    let rect = Rectangle::new(rect.loc + self.shared.geometry.loc, rect.size)
                        .intersection(*corner)?;
                    let mut physical = rect.to_physical_precise_up(scale);
                    physical.loc -= self.geometry(scale).loc;
                    Some(physical)
                })
            })
            .collect()
    }
}

impl SharedCornerRenderElement {
    fn draw_damage(
        &self,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
    ) -> (Vec<Rectangle<i32, Physical>>, usize) {
        let boundary = map_damage(&self.shared.regions, self.shared.geometry, src, dst, damage);
        let mut combined = Vec::new();
        if let Some(native) = &*self.shared.native.borrow() {
            for rect in &native.damage {
                if let Some(mut rect) = rect.intersection(dst) {
                    rect.loc -= dst.loc;
                    for damage in damage {
                        if let Some(rect) = rect.intersection(*damage) {
                            let rects = Rectangle::subtract_rects_many(
                                [rect],
                                boundary.iter().chain(combined.iter()).copied(),
                            );
                            combined.extend(rects);
                        }
                    }
                }
            }
        }
        let native_count = combined.len();
        combined.extend(boundary);
        (combined, native_count)
    }

    fn key(&self, sources: &[Source], native: usize, appearance: bool) -> ShaderKey {
        let mut flags = [0; MAX_SOURCES];
        for (slot, source) in flags.iter_mut().zip(sources) {
            *slot = source.flags;
        }
        let background_mask = self
            .shared
            .decorations
            .iter()
            .enumerate()
            .fold(0, |mask, (index, decoration)| {
                mask | (u8::from(decoration.behavior[0] == 1.) << index)
            });
        ShaderKey {
            flags,
            count: sources.len() as u8,
            native: native as u8,
            native_flags: 0,
            background_mask,
            appearance,
        }
    }

    fn program(frame: &mut GlesFrame<'_, '_>, key: ShaderKey) -> Result<ShaderProgram, GlesError> {
        if let Some(program) = super::shaders::Shaders::get_from_frame(frame)
            .shared_programs
            .borrow()
            .get(&key)
            .cloned()
        {
            return program.ok_or(GlesError::UnsupportedPixelLayout);
        }
        let resize = key.uses_resize();
        let resize_source = resize.then(|| {
            super::shaders::Shaders::get_from_frame(frame)
                .shared_resize_source
                .borrow()
                .clone()
                .unwrap_or_else(|| include_str!("shaders/resize.frag").to_owned())
        });
        let (vertex, fragment) =
            super::shaders::shared_corner_portable::generate(key, resize_source.as_deref());
        let program = ShaderProgram::compile_for_frame(
            frame,
            &vertex,
            &fragment,
            &uniform_names(usize::from(key.count), resize),
        );
        super::shaders::Shaders::get_from_frame(frame)
            .shared_programs
            .borrow_mut()
            .insert(key, program.as_ref().ok().cloned());
        program
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_program(
        &self,
        base: &ShaderRenderElement,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        native_count: usize,
        sources: &[Source],
        textures: &[Option<GlesTexture>],
        key: ShaderKey,
        extra: Vec<Uniform<'static>>,
        native_source: Option<Source>,
    ) -> Result<(), GlesError> {
        if damage.is_empty() {
            return Ok(());
        }
        let program = Self::program(frame, key)?;
        let resize = self.shared.resize.borrow();
        let needs_resize = key.uses_resize();
        let mut uniforms = Vec::new();
        for (index, source) in sources.iter().enumerate() {
            uniforms.extend([
                Uniform::new(format!("niri_rect_{index}"), source.rect),
                super::shaders::mat3_uniform(
                    &format!("niri_source_matrix_{index}"),
                    source_matrix(source),
                )
                .to_owned(),
                Uniform::new(format!("niri_color_{index}"), source.color),
                Uniform::new(
                    format!("niri_post_{index}"),
                    [
                        source.postprocess[0],
                        source.postprocess[1],
                        self.shared.scale as f32,
                        source.alpha,
                    ],
                ),
            ]);
        }
        uniforms.extend(extra);
        let mut bindings: Vec<(Cow<'static, str>, &GlesTexture, u32, bool)> = Vec::new();
        if needs_resize {
            let resize = resize.as_ref().unwrap();
            uniforms.extend(resize.uniforms.iter().cloned());
            uniforms.extend([
                Uniform::new("niri_shared_resize_size", resize.size),
                Uniform::new("niri_shared_resize_alpha", resize.alpha),
            ]);
            for (name, texture) in ["niri_tex_prev", "niri_tex_next"]
                .into_iter()
                .zip(&resize.textures)
            {
                bindings.push((Cow::Borrowed(name), texture, ffi::CLAMP_TO_BORDER, false));
            }
        }
        for (index, texture) in textures.iter().enumerate() {
            if let Some(texture) = texture {
                bindings.push((
                    Cow::Owned(format!("niri_source_texture_{index}")),
                    texture,
                    ffi::CLAMP_TO_EDGE,
                    sources[index].flags & ATLAS != 0,
                ));
            }
        }
        let native = self.shared.native.borrow();
        if let Some(source) = native_source {
            uniforms.push(super::shaders::mat3_uniform(
                "niri_native_matrix",
                source_matrix(&source),
            ));
            if let Some(texture) = native.as_ref().and_then(|native| native.texture.as_ref()) {
                bindings.push((
                    Cow::Borrowed("niri_native_texture"),
                    texture,
                    ffi::CLAMP_TO_EDGE,
                    false,
                ));
            }
        }
        for (unit, (name, _, _, _)) in bindings.iter().enumerate() {
            uniforms.push(Uniform::new(name.clone(), unit as i32));
        }
        frame.with_context(|gl| unsafe {
            for (unit, (_, texture, wrap, nearest)) in bindings.iter().enumerate() {
                let target = if texture.is_external() {
                    ffi::TEXTURE_EXTERNAL_OES
                } else {
                    ffi::TEXTURE_2D
                };
                gl.ActiveTexture(ffi::TEXTURE0 + unit as u32);
                gl.BindTexture(target, texture.tex_id());
                let filter = if *nearest { ffi::NEAREST } else { ffi::LINEAR };
                gl.TexParameteri(target, ffi::TEXTURE_MIN_FILTER, filter as i32);
                gl.TexParameteri(target, ffi::TEXTURE_MAG_FILTER, filter as i32);
                gl.TexParameteri(target, ffi::TEXTURE_WRAP_S, *wrap as i32);
                gl.TexParameteri(target, ffi::TEXTURE_WRAP_T, *wrap as i32);
            }
        })?;
        let inner = base
            .clone()
            .with_program_and_uniforms(ProgramType::SharedCorner, uniforms)
            .with_shader(program, native_count);
        let result = <ShaderRenderElement as RenderElement<GlesRenderer>>::draw(
            &inner,
            frame,
            src,
            dst,
            damage,
            &[],
            None,
        );
        frame.with_context(|gl| unsafe {
            for (unit, (_, texture, _, _)) in bindings.iter().enumerate() {
                gl.ActiveTexture(ffi::TEXTURE0 + unit as u32);
                gl.BindTexture(
                    if texture.is_external() {
                        ffi::TEXTURE_EXTERNAL_OES
                    } else {
                        ffi::TEXTURE_2D
                    },
                    0,
                );
            }
            gl.ActiveTexture(ffi::TEXTURE0);
        })?;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_gles(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        native_count: usize,
        _opaque: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        if damage.is_empty() {
            return Ok(());
        }
        let sources = self.shared.sources.borrow();
        let textures = self.shared.textures.borrow();
        // Resize snapshots are produced and sampled on this renderer's command stream.
        let reads = textures.iter().flatten().cloned().collect::<Vec<_>>();
        frame.with_textures(&reads, |frame| {
            let limits = Limits::get(frame)?;
            let empty = Source {
                texture_units: 0,
                flags: SOLID,
                alpha: 0.,
                rect: [0.; 4],
                tex_x: [0.; 4],
                tex_y: [0.; 4],
                color: [0.; 4],
                postprocess: [0., 1., 0., 0.],
            };
            let sources = if sources.is_empty() {
                std::slice::from_ref(&empty)
            } else {
                sources.as_slice()
            };
            let native = self
                .shared
                .native
                .borrow()
                .as_ref()
                .map_or(0, |native| native.source);
            if native_count == damage.len() {
                let key = self.key(&sources[native..=native], NATIVE_ONLY as usize, false);
                return self.draw_program(
                    &self.inner,
                    frame,
                    src,
                    dst,
                    damage,
                    native_count,
                    &sources[native..=native],
                    &textures[native..=native],
                    key,
                    Vec::new(),
                    None,
                );
            }
            let direct_key = self.key(
                sources,
                if native_count == 0 {
                    NO_NATIVE as usize
                } else {
                    native
                },
                false,
            );
            let direct_fits = limits.direct(
                sources.len(),
                sources.iter().any(|source| source.flags & EFFECT != 0),
            ) && sources
                .iter()
                .map(|source| source.texture_units)
                .sum::<usize>()
                <= limits.textures;
            let direct_program = direct_fits.then(|| Self::program(frame, direct_key));
            if !matches!(direct_program, Some(Ok(_))) {
                let effective_scale = (
                    f64::from(dst.size.w) / (src.size.w * self.shared.geometry.size.w),
                    f64::from(dst.size.h) / (src.size.h * self.shared.geometry.size.h),
                );
                let max_corner = self
                    .shared
                    .corners
                    .iter()
                    .map(|corner| {
                        (corner.size.w.max(corner.size.h) * self.shared.scale).ceil() as i32
                    })
                    .max()
                    .unwrap_or(0);
                let max_window = (self.shared.window.size.w.max(self.shared.window.size.h)
                    * self.shared.scale)
                    .round() as i32;
                if (effective_scale.0 - self.shared.scale).abs() > 1e-6
                    || (effective_scale.1 - self.shared.scale).abs() > 1e-6
                    || max_corner * 2 > limits.max_texture
                    || max_window > limits.max_texture
                {
                    return self.draw_pages(
                        frame,
                        src,
                        dst,
                        damage,
                        native_count,
                        sources,
                        &textures,
                        limits,
                    );
                }
                let Some(atlas) = self.appearance_atlas(
                    frame,
                    src,
                    dst,
                    &damage[native_count..],
                    sources,
                    &textures,
                    limits,
                )?
                else {
                    if native_count == 0 {
                        let key = self.key(std::slice::from_ref(&empty), NO_NATIVE as usize, false);
                        return self.draw_program(
                            &self.inner,
                            frame,
                            src,
                            dst,
                            damage,
                            0,
                            std::slice::from_ref(&empty),
                            &[None],
                            key,
                            Vec::new(),
                            None,
                        );
                    }
                    let key = self.key(&sources[native..=native], 0, false);
                    return self.draw_program(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        damage,
                        native_count,
                        &sources[native..=native],
                        &textures[native..=native],
                        key,
                        Vec::new(),
                        None,
                    );
                };
                let window = self.shared.window;
                let offset = window.loc - self.shared.geometry.loc;
                let native_source = sources[native];
                let atlas_source = Source {
                    texture_units: 1,
                    flags: ATLAS,
                    alpha: native_source.alpha,
                    rect: [
                        offset.x as f32,
                        offset.y as f32,
                        window.size.w as f32,
                        window.size.h as f32,
                    ],
                    tex_x: [0.; 4],
                    tex_y: [0.; 4],
                    color: native_source.color,
                    postprocess: [0., 1., 0., 0.],
                };
                let mut key = self.key(
                    &[atlas_source],
                    if native_count == 0 {
                        NO_NATIVE as usize
                    } else {
                        0
                    },
                    false,
                );
                key.native_flags = if native_count == 0 {
                    0
                } else {
                    native_source.flags
                };
                let mut final_native_count = native_count;
                if native_count > 0 && Self::program(frame, key).is_err() {
                    let native_key =
                        self.key(&sources[native..=native], NATIVE_ONLY as usize, false);
                    self.draw_program(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        &damage[..native_count],
                        native_count,
                        &sources[native..=native],
                        &textures[native..=native],
                        native_key,
                        Vec::new(),
                        None,
                    )?;
                    key.native = NO_NATIVE;
                    key.native_flags = 0;
                    final_native_count = 0;
                }
                // The atlas is private to this renderer. Its reads and writes share the command
                // stream.
                let extra = vec![Uniform::new(
                    "niri_post_0",
                    [
                        if atlas.full {
                            0.
                        } else {
                            atlas.slot.w as f32 / self.shared.scale as f32
                        },
                        if atlas.full {
                            0.
                        } else {
                            atlas.slot.h as f32 / self.shared.scale as f32
                        },
                        self.shared.scale as f32,
                        native_source.alpha,
                    ],
                )];
                return self.draw_program(
                    &self.inner,
                    frame,
                    src,
                    dst,
                    &damage[if final_native_count == 0 {
                        native_count
                    } else {
                        0
                    }..],
                    final_native_count,
                    &[atlas_source],
                    &[Some(atlas.texture)],
                    key,
                    extra,
                    (key.native != NO_NATIVE).then_some(native_source),
                );
            }
            self.draw_program(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                native_count,
                sources,
                &textures,
                direct_key,
                Vec::new(),
                None,
            )
        })
    }
}

#[derive(Default)]
struct AtlasCache(Rc<RefCell<AtlasStorage>>);

#[derive(Default)]
struct AtlasStorage {
    texture: Option<GlesTexture>,
    format: Option<Fourcc>,
}

impl AtlasStorage {
    fn texture(
        &mut self,
        renderer: &mut GlesRenderer,
        size: Size<i32, Buffer>,
    ) -> Result<GlesTexture, GlesError> {
        if let Some(texture) = &self.texture {
            if texture.size() == size {
                return Ok(texture.clone());
            }
        }

        if self.format != Some(Fourcc::Abgr8888) {
            // Keep fractional alpha across batches and preserve signed effect colors.
            if let Ok(mut texture) =
                Offscreen::<GlesTexture>::create_buffer(renderer, Fourcc::Abgr16161616f, size)
            {
                if renderer.bind(&mut texture).is_ok() {
                    self.format = Some(Fourcc::Abgr16161616f);
                    self.texture = Some(texture.clone());
                    return Ok(texture);
                }
            }
        }

        let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, size)?;
        self.format = Some(Fourcc::Abgr8888);
        self.texture = Some(texture.clone());
        Ok(texture)
    }
}

struct AtlasDraw {
    texture: GlesTexture,
    slot: Size<i32, Physical>,
    full: bool,
}

impl SharedCornerRenderElement {
    #[allow(clippy::too_many_arguments)]
    fn appearance_atlas(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        sources: &[Source],
        textures: &[Option<GlesTexture>],
        limits: Limits,
    ) -> Result<Option<AtlasDraw>, GlesError> {
        let scale = self.shared.scale;
        let window = self.shared.window;
        let size = window.size.to_physical_precise_round::<_, i32>(scale);
        let slot = Size::from((
            self.shared
                .corners
                .iter()
                .map(|corner| (corner.size.w * scale).ceil() as i32)
                .max()
                .unwrap_or(1)
                .min(size.w),
            self.shared
                .corners
                .iter()
                .map(|corner| (corner.size.h * scale).ceil() as i32)
                .max()
                .unwrap_or(1)
                .min(size.h),
        ));
        let full = slot.w * 2 >= size.w || slot.h * 2 >= size.h;
        let slot = if full { size } else { slot };
        let atlas_size = if full {
            size
        } else {
            Size::from((slot.w * 2, slot.h * 2))
        };
        let offset = window.loc - self.shared.geometry.loc;
        let mut mapped = Vec::new();
        for quadrant in 0..if full { 1 } else { 4 } {
            let quad = smithay::utils::Point::<i32, Physical>::from((quadrant % 2, quadrant / 2));
            let origin = smithay::utils::Point::from((
                quad.x * (size.w - slot.w),
                quad.y * (size.h - slot.h),
            ));
            let corner = Rectangle::new(origin, slot);
            let mut bounds: Option<Rectangle<i32, Physical>> = None;
            for rect in damage {
                let relative = Rectangle::<f64, Logical>::new(
                    (
                        (src.loc.x + f64::from(rect.loc.x) / f64::from(dst.size.w) * src.size.w)
                            * self.shared.geometry.size.w
                            - offset.x,
                        (src.loc.y + f64::from(rect.loc.y) / f64::from(dst.size.h) * src.size.h)
                            * self.shared.geometry.size.h
                            - offset.y,
                    )
                        .into(),
                    (
                        f64::from(rect.size.w) / f64::from(dst.size.w)
                            * src.size.w
                            * self.shared.geometry.size.w,
                        f64::from(rect.size.h) / f64::from(dst.size.h)
                            * src.size.h
                            * self.shared.geometry.size.h,
                    )
                        .into(),
                )
                .to_physical_precise_up(scale);
                if let Some(rect) = relative.intersection(corner) {
                    bounds = Some(bounds.map_or(rect, |previous| previous.merge(rect)));
                }
            }
            if let Some(mut rect) = bounds {
                rect.loc = rect.loc - origin
                    + smithay::utils::Point::from((quad.x * slot.w, quad.y * slot.h));
                mapped.push(rect);
            }
        }
        if mapped.is_empty() {
            return Ok(None);
        }
        let viewport = frame.with_context(|gl| unsafe {
            let mut viewport = [0; 4];
            gl.GetIntegerv(ffi::VIEWPORT, viewport.as_mut_ptr());
            viewport
        })?;
        let from_shared = Mat3::from_translation(Vec2::new(dst.loc.x as f32, dst.loc.y as f32))
            * Mat3::from_scale(Vec2::new(
                dst.size.w as f32 / src.size.w as f32,
                dst.size.h as f32 / src.size.h as f32,
            ))
            * Mat3::from_translation(Vec2::new(-src.loc.x as f32, -src.loc.y as f32))
            * Mat3::from_scale(Vec2::new(
                1. / self.shared.geometry.size.w as f32,
                1. / self.shared.geometry.size.h as f32,
            ));
        let output_matrix =
            Mat3::from_translation(Vec2::new(viewport[0] as f32, viewport[1] as f32))
                * Mat3::from_scale(Vec2::new(
                    viewport[2] as f32 * 0.5,
                    viewport[3] as f32 * 0.5,
                ))
                * Mat3::from_translation(Vec2::ONE)
                * Mat3::from_cols_array(frame.projection())
                * from_shared;
        let texture = self.render_appearance(
            frame,
            atlas_size,
            &mapped,
            sources,
            textures,
            limits,
            [slot.w as f32, slot.h as f32, size.w as f32, size.h as f32],
            output_matrix,
            None,
        )?;
        Ok(Some(AtlasDraw {
            texture,
            slot,
            full,
        }))
    }
    #[allow(clippy::too_many_arguments)]
    fn render_appearance(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        atlas_size: Size<i32, Physical>,
        mapped: &[Rectangle<i32, Physical>],
        sources: &[Source],
        textures: &[Option<GlesTexture>],
        limits: Limits,
        mapping: [f32; 4],
        output_matrix: Mat3,
        page_matrix: Option<Mat3>,
    ) -> Result<GlesTexture, GlesError> {
        let scale = self.shared.scale;
        let window = self.shared.window;
        let offset = window.loc - self.shared.geometry.loc;
        let cache = frame
            .local_user_data()
            .get_or_insert::<AtlasCache, _>(AtlasCache::default)
            .0
            .clone();
        let mut guard = frame.renderer();
        let renderer = guard.as_mut();
        let texture = cache
            .borrow_mut()
            .texture(renderer, (atlas_size.w, atlas_size.h).into())?;
        let mut target = texture.clone();
        let mut framebuffer = renderer.bind(&mut target)?;
        let mut atlas_frame = renderer.render(&mut framebuffer, atlas_size, Transform::Normal)?;
        atlas_frame.clear([0f32; 4].into(), mapped)?;
        let base = ShaderRenderElement::new(
            ProgramType::SharedCorner,
            atlas_size.to_f64().to_logical(scale),
            None,
            scale as f32,
            1.,
            self.inner.uniforms(),
            HashMap::new(),
            Kind::Unspecified,
        );
        let mut extra = vec![
            Uniform::new("niri_atlas_mapping", mapping),
            super::shaders::mat3_uniform("niri_output_matrix", output_matrix),
        ];
        if let Some(matrix) = page_matrix {
            extra.push(super::shaders::mat3_uniform("niri_matrix", matrix));
        }
        let max_sources = MAX_SOURCES
            .min(limits.fragment / 2)
            .min((limits.varying * 4 - 5) / 4)
            .min(limits.textures)
            .max(1);
        let render_group =
            |frame: &mut GlesFrame<'_, '_>, group: &[usize]| -> Result<(), GlesError> {
                let mut start = 0;
                while start < group.len() {
                    let mut end = start;
                    let mut units = 0;
                    let mut vectors = 0;
                    while end < group.len() && end - start < max_sources {
                        let source = sources[group[end]];
                        let needed = if source.flags & EFFECT != 0 { 3 } else { 2 };
                        if units + source.texture_units > limits.textures
                            || vectors + needed > limits.fragment
                        {
                            break;
                        }
                        units += source.texture_units;
                        vectors += needed;
                        end += 1;
                    }
                    if end == start {
                        return Err(GlesError::UnsupportedPixelLayout);
                    }
                    let chunk = &group[start..end];
                    start = end;
                    let batch = chunk
                        .iter()
                        .map(|index| sources[*index])
                        .collect::<Vec<_>>();
                    let batch_textures = chunk
                        .iter()
                        .map(|index| textures[*index].clone())
                        .collect::<Vec<_>>();
                    let key = self.key(&batch, 0, true);
                    self.draw_program(
                        &base,
                        frame,
                        Rectangle::from_size((1., 1.).into()),
                        Rectangle::from_size(atlas_size),
                        mapped,
                        0,
                        &batch,
                        &batch_textures,
                        key,
                        extra.clone(),
                        None,
                    )?;
                }
                Ok(())
            };
        let effects = sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| (source.flags & EFFECT != 0).then_some(index))
            .collect::<Vec<_>>();
        render_group(&mut atlas_frame, &effects)?;
        if self
            .shared
            .decorations
            .iter()
            .any(|decoration| decoration.behavior[0] == 1.)
        {
            let background = Source {
                texture_units: 0,
                flags: BACKGROUND,
                alpha: 1.,
                rect: [
                    offset.x as f32,
                    offset.y as f32,
                    window.size.w as f32,
                    window.size.h as f32,
                ],
                tex_x: [0.; 4],
                tex_y: [0.; 4],
                color: [0.; 4],
                postprocess: [0., 1., 0., 0.],
            };
            let key = self.key(&[background], 0, true);
            self.draw_program(
                &base,
                &mut atlas_frame,
                Rectangle::from_size((1., 1.).into()),
                Rectangle::from_size(atlas_size),
                mapped,
                0,
                &[background],
                &[None],
                key,
                extra.clone(),
                None,
            )?;
        }
        let content = sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| (source.flags & EFFECT == 0).then_some(index))
            .collect::<Vec<_>>();
        render_group(&mut atlas_frame, &content)?;
        let _sync = atlas_frame.finish()?;
        Ok(texture)
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_pages(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        native_count: usize,
        sources: &[Source],
        textures: &[Option<GlesTexture>],
        limits: Limits,
    ) -> Result<(), GlesError> {
        let native = self
            .shared
            .native
            .borrow()
            .as_ref()
            .map_or(0, |native| native.source);
        if native_count > 0 {
            let key = self.key(&sources[native..=native], NATIVE_ONLY as usize, false);
            self.draw_program(
                &self.inner,
                frame,
                src,
                dst,
                &damage[..native_count],
                native_count,
                &sources[native..=native],
                &textures[native..=native],
                key,
                Vec::new(),
                None,
            )?;
        }
        let viewport = frame.with_context(|gl| unsafe {
            let mut viewport = [0; 4];
            gl.GetIntegerv(ffi::VIEWPORT, viewport.as_mut_ptr());
            viewport
        })?;
        let root_size = self.shared.geometry.size;
        let from_shared = Mat3::from_translation(Vec2::new(dst.loc.x as f32, dst.loc.y as f32))
            * Mat3::from_scale(Vec2::new(
                dst.size.w as f32 / src.size.w as f32,
                dst.size.h as f32 / src.size.h as f32,
            ))
            * Mat3::from_translation(Vec2::new(-src.loc.x as f32, -src.loc.y as f32))
            * Mat3::from_scale(Vec2::new(1. / root_size.w as f32, 1. / root_size.h as f32));
        let output_matrix =
            Mat3::from_translation(Vec2::new(viewport[0] as f32, viewport[1] as f32))
                * Mat3::from_scale(Vec2::new(
                    viewport[2] as f32 * 0.5,
                    viewport[3] as f32 * 0.5,
                ))
                * Mat3::from_translation(Vec2::ONE)
                * Mat3::from_cols_array(frame.projection())
                * from_shared;
        for rect in &damage[native_count..] {
            for y in (0..rect.size.h).step_by(limits.max_texture as usize) {
                for x in (0..rect.size.w).step_by(limits.max_texture as usize) {
                    let page = Rectangle::new(
                        rect.loc + smithay::utils::Point::from((x, y)),
                        Size::from((
                            (rect.size.w - x).min(limits.max_texture),
                            (rect.size.h - y).min(limits.max_texture),
                        )),
                    );
                    let origin = Vec2::new(
                        (src.loc.x + f64::from(page.loc.x) / f64::from(dst.size.w) * src.size.w)
                            as f32
                            * root_size.w as f32,
                        (src.loc.y + f64::from(page.loc.y) / f64::from(dst.size.h) * src.size.h)
                            as f32
                            * root_size.h as f32,
                    );
                    let pixel = Vec2::new(
                        (src.size.w * root_size.w / f64::from(dst.size.w)) as f32,
                        (src.size.h * root_size.h / f64::from(dst.size.h)) as f32,
                    );
                    let logical_size = pixel * Vec2::new(page.size.w as f32, page.size.h as f32);
                    let matrix = Mat3::from_translation(origin) * Mat3::from_scale(pixel);
                    let atlas = self.render_appearance(
                        frame,
                        page.size,
                        &[Rectangle::from_size(page.size)],
                        sources,
                        textures,
                        limits,
                        [-1., 0., 0., 0.],
                        output_matrix,
                        Some(matrix),
                    )?;
                    let uv = Mat3::from_scale(Vec2::ONE / logical_size)
                        * Mat3::from_translation(-origin);
                    let columns = uv.to_cols_array_2d();
                    let source = Source {
                        texture_units: 1,
                        flags: ATLAS,
                        alpha: 1.,
                        rect: [origin.x, origin.y, logical_size.x, logical_size.y],
                        tex_x: [columns[0][0], columns[1][0], columns[2][0], 0.],
                        tex_y: [columns[0][1], columns[1][1], columns[2][1], 0.],
                        color: [0.; 4],
                        postprocess: [0., 1., 0., 0.],
                    };
                    let key = self.key(&[source], NO_NATIVE as usize, false);
                    let extra = vec![Uniform::new(
                        "niri_post_0",
                        [-1., 0., self.shared.scale as f32, 1.],
                    )];
                    self.draw_program(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        &[page],
                        0,
                        &[source],
                        &[Some(atlas)],
                        key,
                        extra,
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl RenderElement<GlesRenderer> for SharedCornerRenderElement {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let (damage, native_count) = self.draw_damage(src, dst, damage);
        let result = self.draw_gles(frame, src, dst, &damage, native_count, opaque, cache);
        self.shared.finish_draw();
        result
    }
    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

impl<'render> RenderElement<TtyRenderer<'render>> for SharedCornerRenderElement {
    fn draw(
        &self,
        frame: &mut TtyFrame<'_, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let (damage, native_count) = self.draw_damage(src, dst, damage);
        frame.add_damage(
            damage
                .iter()
                .map(|rect| Rectangle::new(rect.loc + dst.loc, rect.size)),
        );
        let result = self.draw_gles(
            frame.as_gles_frame(),
            src,
            dst,
            &damage,
            native_count,
            opaque,
            cache,
        );
        self.shared.finish_draw();
        result?;
        Ok(())
    }
    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

fn map_damage(
    regions: &[Rectangle<f64, Logical>],
    geometry: Rectangle<f64, Logical>,
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
) -> Vec<Rectangle<i32, Physical>> {
    let mut mapped = Vec::new();
    for region in regions {
        let offset = region.loc - geometry.loc;
        let normalized = Rectangle::<f64, Buffer>::new(
            (offset.x / geometry.size.w, offset.y / geometry.size.h).into(),
            (
                region.size.w / geometry.size.w,
                region.size.h / geometry.size.h,
            )
                .into(),
        );
        let Some(clipped) = normalized.intersection(src) else {
            continue;
        };
        let relative = clipped.loc - src.loc;
        let rect = Rectangle::<f64, Physical>::new(
            (
                relative.x / src.size.w * f64::from(dst.size.w),
                relative.y / src.size.h * f64::from(dst.size.h),
            )
                .into(),
            (
                clipped.size.w / src.size.w * f64::from(dst.size.w),
                clipped.size.h / src.size.h * f64::from(dst.size.h),
            )
                .into(),
        )
        .to_i32_round();
        for damage in damage {
            if let Some(rect) = rect.intersection(*damage) {
                let remaining = Rectangle::subtract_rects_many([rect], mapped.iter().copied());
                mapped.extend(remaining);
            }
        }
    }
    mapped
}
