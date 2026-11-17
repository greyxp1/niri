use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::multigpu::gbm::GbmGlesBackend;
use smithay::backend::renderer::multigpu::MultiTexture;
use smithay::backend::renderer::{
    Bind, ContextId, ExportMem, ImportAll, ImportMem, Offscreen, Renderer, RendererSuper, Texture,
};

use crate::backend::tty::{TtyFrame, TtyRenderer};

/// Trait with our main renderer requirements to save on the typing.
pub trait NiriRenderer:
    ImportAll
    + ImportMem
    + ExportMem
    + Bind<Dmabuf>
    + Offscreen<GlesTexture>
    + Renderer<TextureId = Self::NiriTextureId, Error = Self::NiriError>
    + AsGlesRenderer
{
    // Associated types to work around the instability of associated type bounds.
    type NiriTextureId: Texture + Clone + Send + AsGlesTexture + 'static;
    type NiriError: std::error::Error
        + Send
        + Sync
        + From<<GlesRenderer as RendererSuper>::Error>
        + 'static;
}

impl<R> NiriRenderer for R
where
    R: ImportAll + ImportMem + ExportMem + Bind<Dmabuf> + Offscreen<GlesTexture> + AsGlesRenderer,
    R::TextureId: Texture + Clone + Send + AsGlesTexture + 'static,
    R::Error:
        std::error::Error + Send + Sync + From<<GlesRenderer as RendererSuper>::Error> + 'static,
{
    type NiriTextureId = R::TextureId;
    type NiriError = R::Error;
}

/// Resolve the native texture after it has been imported and synchronized.
pub trait AsGlesTexture {
    fn as_gles_texture(&self, context: &ContextId<GlesTexture>) -> Option<GlesTexture>;
}

impl AsGlesTexture for GlesTexture {
    fn as_gles_texture(&self, _context: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        Some(self.clone())
    }
}

impl AsGlesTexture for MultiTexture {
    fn as_gles_texture(&self, context: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        self.get::<GbmGlesBackend<GlesRenderer, smithay::utils::DeviceFd>>(context)
    }
}

/// Trait for getting the underlying `GlesRenderer`.
pub trait AsGlesRenderer {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer;
}

impl AsGlesRenderer for GlesRenderer {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer {
        self
    }
}

impl AsGlesRenderer for TtyRenderer<'_> {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer {
        self.as_mut()
    }
}

/// Trait for getting the underlying `GlesFrame`.
pub trait AsGlesFrame<'frame, 'buffer>
where
    Self: 'frame,
{
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer>;
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for GlesFrame<'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer> {
        self
    }
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for TtyFrame<'_, 'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer> {
        self.as_mut()
    }
}
