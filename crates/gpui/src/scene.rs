// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, Rgba, ScaledPixels, Size, bounds_tree::BoundsTree, point, size,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    time::Duration,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// Backdrop-blur regions — deliberately OUTSIDE the primitive batch
    /// stream: the renderer breaks its render pass at each blur's order to
    /// snapshot the framebuffer (macOS Metal; other renderers ignore them).
    pub backdrop_blurs: Vec<BackdropBlur>,
    /// Animations resolved at presentation time, indexed by
    /// `MonochromeSprite::pad - 1` or `SubpixelSprite::pad - 1`. The id
    /// travels inside the sprite, so it survives `finish`'s sorting; `replay`
    /// re-registers it.
    pub(crate) sprite_animations: Vec<SpriteAnimation>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.backdrop_blurs.clear();
        self.sprite_animations.clear();
    }

    /// Whether there are no drawable operations, ignoring empty layer markers.
    pub fn is_empty(&self) -> bool {
        self.paint_operations
            .iter()
            .all(|op| matches!(op, PaintOperation::StartLayer(_) | PaintOperation::EndLayer))
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_backdrop_blur(&mut self, mut blur: BackdropBlur) {
        let clipped_bounds = blur.bounds.intersect(&blur.content_mask.bounds);
        if clipped_bounds.is_empty() {
            return;
        }
        blur.order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        self.backdrop_blurs.push(blur);
        self.paint_operations
            .push(PaintOperation::BackdropBlur(blur));
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Insert a monochrome sprite whose transformation is recomputed every
    /// time the frame is presented, without repainting.
    pub(crate) fn insert_animated_monochrome_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        animation: SpriteAnimation,
    ) {
        self.sprite_animations.push(animation);
        sprite.pad = self.sprite_animations.len() as u32;
        let sprite_count = self.monochrome_sprites.len();
        self.insert_primitive(sprite);
        if self.monochrome_sprites.len() == sprite_count {
            // Fully clipped: nothing on screen, nothing to keep presenting.
            self.sprite_animations.pop();
        }
    }

    /// Insert a subpixel (glyph) sprite whose color is recomputed every time
    /// the frame is presented, without repainting. Shares the animation
    /// table, and the meaning of `pad`, with monochrome sprites.
    pub(crate) fn insert_animated_subpixel_sprite(
        &mut self,
        mut sprite: SubpixelSprite,
        animation: SpriteAnimation,
    ) {
        self.sprite_animations.push(animation);
        sprite.pad = self.sprite_animations.len() as u32;
        let sprite_count = self.subpixel_sprites.len();
        self.insert_primitive(sprite);
        if self.subpixel_sprites.len() == sprite_count {
            // Fully clipped: nothing on screen, nothing to keep presenting.
            self.sprite_animations.pop();
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                // An animated sprite's id points into the previous scene's
                // table; carry the animation over and give the sprite an id in
                // this scene.
                PaintOperation::Primitive(Primitive::MonochromeSprite(sprite))
                    if sprite.pad != 0 =>
                {
                    let animation = prev_scene.sprite_animations[sprite.pad as usize - 1];
                    self.insert_animated_monochrome_sprite(*sprite, animation);
                }
                PaintOperation::Primitive(Primitive::SubpixelSprite(sprite)) if sprite.pad != 0 => {
                    let animation = prev_scene.sprite_animations[sprite.pad as usize - 1];
                    self.insert_animated_subpixel_sprite(*sprite, animation);
                }
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::BackdropBlur(blur) => self.insert_backdrop_blur(*blur),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.backdrop_blurs.sort_by_key(|blur| blur.order);
    }

    /// Whether any sprite in this scene animates at presentation time.
    pub(crate) fn has_present_animations(&self) -> bool {
        !self.sprite_animations.is_empty()
    }

    /// The shortest frame interval requested by the scene's animations.
    pub(crate) fn present_animation_interval(&self) -> Option<Duration> {
        self.sprite_animations
            .iter()
            .map(|animation| animation.animation.frame_interval)
            .min()
    }

    /// Rewrite the transformation of every animated sprite for the given
    /// wall-clock time (time since the Unix epoch).
    ///
    /// Both the sorted sprites (what renderers draw) and the recorded paint
    /// operations are updated, because a platform may split the scene by
    /// replaying those operations (the macOS overlay plane does).
    pub(crate) fn resolve_present_animations(&mut self, since_epoch: Duration) {
        if self.sprite_animations.is_empty() {
            return;
        }
        let animations = &self.sprite_animations;
        let resolve = |pad: u32, transformation: &mut TransformationMatrix, color: &mut Hsla| {
            if pad != 0 {
                let animation = &animations[pad as usize - 1];
                *transformation = animation.transformation_at(since_epoch);
                *color = animation.color_at(since_epoch);
            }
        };
        for sprite in &mut self.monochrome_sprites {
            resolve(sprite.pad, &mut sprite.transformation, &mut sprite.color);
        }
        for sprite in &mut self.subpixel_sprites {
            resolve(sprite.pad, &mut sprite.transformation, &mut sprite.color);
        }
        for operation in &mut self.paint_operations {
            match operation {
                PaintOperation::Primitive(Primitive::MonochromeSprite(sprite)) => {
                    resolve(sprite.pad, &mut sprite.transformation, &mut sprite.color);
                }
                PaintOperation::Primitive(Primitive::SubpixelSprite(sprite)) => {
                    resolve(sprite.pad, &mut sprite.transformation, &mut sprite.color);
                }
                _ => {}
            }
        }
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    BackdropBlur(BackdropBlur),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

/// Per-primitive scoped edge fade (see `Window::with_edge_fade`): the
/// fragment shader multiplies alpha by a squared ramp measured from these
/// window-space edges (device pixels) — a TRUE per-pixel fade, so large
/// fills and images dissolve across the band instead of popping at their
/// bounding-box edge. A zero band disables that edge; zeroed = no fade.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct EdgeFadeParams {
    pub top_y: f32,
    pub bottom_y: f32,
    pub band_top: f32,
    pub band_bottom: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub band_left: f32,
    pub band_right: f32,
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub fade: EdgeFadeParams,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

/// A within-window backdrop blur region: the renderer snapshots everything
/// painted below this order and paints it back gaussian-blurred inside the
/// rounded bounds (frosted-glass popovers). macOS Metal only — see
/// [`crate::Window::paint_backdrop_blur`].
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BackdropBlur {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

/// Motion that the window resolves when it presents a frame, so a
/// decorative animation never re-renders, lays out or repaints any view.
#[derive(Clone, Copy, Debug)]
pub enum PresentMotion {
    /// Continuous, linear, clockwise rotation around the sprite's center,
    /// one full turn per `period`. The phase comes from the wall clock, so
    /// every spinner on screen with the same period turns in sync.
    Spin {
        /// Duration of one full turn.
        period: Duration,
    },
    /// Continuous opacity pulse: the sprite's alpha follows
    /// `0.5 - 0.5 * cos(phase)`, from invisible to the alpha it was painted
    /// with and back, once per `period`. The phase comes from the wall
    /// clock, so every pulse with the same period breathes in sync.
    Pulse {
        /// Duration of one full pulse.
        period: Duration,
    },
    /// Continuous highlight sweeping across a text: each glyph's color is
    /// `mix(painted, to, weight(phase, x))`, where `phase` is where the wall
    /// clock is inside `period` (`0..1`) and `x` is where the glyph's center
    /// sits along the text (`0..1`, see [`crate::StyledText::sweep`]). The
    /// mix interpolates in sRGB with premultiplied alpha. Every sweep with the
    /// same period moves in sync.
    Sweep {
        /// Duration of one full cycle of `weight`.
        period: Duration,
        /// The color a glyph reaches at weight 1. Scaled by the opacity the
        /// glyph is painted with, like the glyph's own color.
        to: Hsla,
        /// How far each glyph is toward `to`, in `0..1`, as a function of
        /// `(phase, x)`. A plain function, so the motion stays `Copy` and
        /// cheap to evaluate for every glyph on every presented frame.
        weight: SweepWeight,
    },
    /// Continuous movement of a sprite along a path: at every presented
    /// frame the sprite is moved by `path(phase).offset` and scaled by
    /// `path(phase).scale` around its own center, where `phase` is where the
    /// wall clock is inside `period` (`0..1`). Offsets are fractions of the
    /// sprite's painted size, so one path serves every size. Every travel with
    /// the same period moves in sync.
    Travel {
        /// Duration of one full trip along `path`.
        period: Duration,
        /// Where the sprite is at each phase. A plain function, so the motion
        /// stays `Copy` and cheap to evaluate on every presented frame.
        path: TravelPath,
    },
}

/// Where a [`PresentMotion::Travel`] puts its sprite at one phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TravelPose {
    /// Displacement from where the sprite was painted, as a fraction of its
    /// painted width (`x`) and height (`y`).
    pub offset: Point<f32>,
    /// Uniform scale around the sprite's center; `1.0` keeps its size.
    pub scale: f32,
}

/// The curve of a [`PresentMotion::Travel`]: the sprite's pose at `phase`,
/// in `0..1`.
pub type TravelPath = fn(phase: f32) -> TravelPose;

/// The curve of a [`PresentMotion::Sweep`]: `weight(phase, x)`, both
/// arguments in `0..1`, returning how far toward the highlight color the
/// glyph at `x` is at `phase` (clamped to `0..1`).
pub type SweepWeight = fn(phase: f32, x: f32) -> f32;

/// Equality compares a sweep's weight function (and a travel's path) by address, which is only a
/// best effort: the same function may have two addresses (or two functions
/// one), so equal motions may compare unequal. Nothing in the engine relies
/// on it.
impl PartialEq for PresentMotion {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Spin { period: a }, Self::Spin { period: b })
            | (Self::Pulse { period: a }, Self::Pulse { period: b }) => a == b,
            (
                Self::Sweep {
                    period: a_period,
                    to: a_to,
                    weight: a_weight,
                },
                Self::Sweep {
                    period: b_period,
                    to: b_to,
                    weight: b_weight,
                },
            ) => a_period == b_period && a_to == b_to && std::ptr::fn_addr_eq(*a_weight, *b_weight),
            (
                Self::Travel {
                    period: a_period,
                    path: a_path,
                },
                Self::Travel {
                    period: b_period,
                    path: b_path,
                },
            ) => a_period == b_period && std::ptr::fn_addr_eq(*a_path, *b_path),
            _ => false,
        }
    }
}

/// A [`PresentMotion`] plus how often it needs a new frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresentAnimation {
    /// What moves.
    pub motion: PresentMotion,
    /// Minimum time between two presented frames for this animation.
    pub frame_interval: Duration,
}

impl PresentAnimation {
    /// A continuous spin: one turn per `period`, at most one frame per
    /// `frame_interval`.
    pub fn spin(period: Duration, frame_interval: Duration) -> Self {
        Self {
            motion: PresentMotion::Spin { period },
            frame_interval,
        }
    }

    /// A continuous opacity pulse: one breath per `period`, at most one
    /// frame per `frame_interval`.
    pub fn pulse(period: Duration, frame_interval: Duration) -> Self {
        Self {
            motion: PresentMotion::Pulse { period },
            frame_interval,
        }
    }

    /// A highlight sweeping across a text toward `to`, following `weight`
    /// once per `period`, at most one frame per `frame_interval`. See
    /// [`PresentMotion::Sweep`].
    pub fn sweep(
        period: Duration,
        to: Hsla,
        weight: SweepWeight,
        frame_interval: Duration,
    ) -> Self {
        Self {
            motion: PresentMotion::Sweep { period, to, weight },
            frame_interval,
        }
    }

    /// A sprite moving along `path` once per `period`, at most one frame per
    /// `frame_interval`. See [`PresentMotion::Travel`].
    pub fn travel(period: Duration, path: TravelPath, frame_interval: Duration) -> Self {
        Self {
            motion: PresentMotion::Travel { period, path },
            frame_interval,
        }
    }

    /// Whether this animation can never change what is on screen.
    pub fn is_static(&self) -> bool {
        match self.motion {
            PresentMotion::Spin { period }
            | PresentMotion::Pulse { period }
            | PresentMotion::Sweep { period, .. }
            | PresentMotion::Travel { period, .. } => period.is_zero(),
        }
    }

    /// This animation for a sprite painted at `opacity`: colors the motion
    /// introduces get the same opacity as the sprite's own color.
    pub(crate) fn with_opacity(mut self, opacity: f32) -> Self {
        if let PresentMotion::Sweep { to, .. } = &mut self.motion {
            *to = to.opacity(opacity);
        }
        self
    }
}

/// A [`PresentAnimation`] bound to one sprite: the transformation it was
/// painted with and the point it moves around, in scaled pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpriteAnimation {
    pub(crate) animation: PresentAnimation,
    pub(crate) base: TransformationMatrix,
    pub(crate) pivot: Point<ScaledPixels>,
    /// The color the sprite was painted with.
    pub(crate) color: Hsla,
    /// Where the sprite sits along its text, in `0..1`; only
    /// [`PresentMotion::Sweep`] reads it.
    pub(crate) position: f32,
    /// The sprite's painted size, in scaled pixels; only
    /// [`PresentMotion::Travel`] reads it.
    pub(crate) extent: Size<ScaledPixels>,
}

impl SpriteAnimation {
    /// The sprite's transformation at `since_epoch` (wall-clock time since
    /// the Unix epoch). The motion is applied in the sprite's own space,
    /// before the static transformation it was painted with.
    pub(crate) fn transformation_at(&self, since_epoch: Duration) -> TransformationMatrix {
        match self.animation.motion {
            PresentMotion::Spin { period } => {
                let Some(phase) = wall_clock_phase(since_epoch, period) else {
                    return self.base;
                };
                let angle = Radians(phase * std::f32::consts::TAU);
                self.base.compose(
                    TransformationMatrix::unit()
                        .translate(self.pivot)
                        .rotate(angle)
                        .translate(point(
                            ScaledPixels(-self.pivot.x.0),
                            ScaledPixels(-self.pivot.y.0),
                        )),
                )
            }
            PresentMotion::Travel { period, path } => {
                let Some(phase) = wall_clock_phase(since_epoch, period) else {
                    return self.base;
                };
                let pose = path(phase);
                let finite = |v: f32| if v.is_finite() { v } else { 0. };
                let scale = if pose.scale.is_finite() { pose.scale.max(0.) } else { 1. };
                let at = point(
                    ScaledPixels(self.pivot.x.0 + finite(pose.offset.x) * self.extent.width.0),
                    ScaledPixels(self.pivot.y.0 + finite(pose.offset.y) * self.extent.height.0),
                );
                self.base.compose(
                    TransformationMatrix::unit()
                        .translate(at)
                        .scale(size(scale, scale))
                        .translate(point(
                            ScaledPixels(-self.pivot.x.0),
                            ScaledPixels(-self.pivot.y.0),
                        )),
                )
            }
            PresentMotion::Pulse { .. } | PresentMotion::Sweep { .. } => self.base,
        }
    }

    /// The sprite's color at `since_epoch`.
    pub(crate) fn color_at(&self, since_epoch: Duration) -> Hsla {
        match self.animation.motion {
            PresentMotion::Spin { .. } | PresentMotion::Travel { .. } => self.color,
            PresentMotion::Pulse { period } => {
                let Some(phase) = wall_clock_phase(since_epoch, period) else {
                    return self.color;
                };
                let wave = 0.5 - 0.5 * (phase * std::f32::consts::TAU).cos();
                self.color.opacity(wave)
            }
            PresentMotion::Sweep { period, to, weight } => {
                let Some(phase) = wall_clock_phase(since_epoch, period) else {
                    return self.color;
                };
                let weight = weight(phase, self.position);
                mix_premultiplied(self.color, to, if weight.is_nan() { 0. } else { weight })
            }
        }
    }
}

/// Interpolate from `from` to `to` in sRGB with premultiplied alpha, so a
/// translucent end does not drag the other end's hue toward black.
/// `t <= 0` returns `from` and `t >= 1` returns `to` exactly.
fn mix_premultiplied(from: Hsla, to: Hsla, t: f32) -> Hsla {
    if t <= 0. {
        return from;
    }
    if t >= 1. {
        return to;
    }
    let lerp = |a: f32, b: f32| a + (b - a) * t;
    let (f, g) = (Rgba::from(from), Rgba::from(to));
    let a = lerp(f.a, g.a);
    if a <= f32::EPSILON {
        return Hsla::from(Rgba { a: 0., ..g });
    }
    Hsla::from(Rgba {
        r: lerp(f.r * f.a, g.r * g.a) / a,
        g: lerp(f.g * f.a, g.g * g.a) / a,
        b: lerp(f.b * f.a, g.b * g.a) / a,
        a,
    })
}

/// Where the wall clock is inside a cycle of `period`, in `0..1`; `None` for
/// a zero period.
fn wall_clock_phase(since_epoch: Duration, period: Duration) -> Option<f32> {
    let period_ms = period.as_millis();
    (period_ms != 0).then(|| (since_epoch.as_millis() % period_ms) as f32 / period_ms as f32)
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    /// Zero, or `1 +` the index of this sprite's presentation-time animation
    /// in its [`Scene`]. Alignment padding for the GPU; no shader reads it.
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    /// Zero, or `1 +` the index of this sprite's presentation-time animation
    /// in its [`Scene`], like [`MonochromeSprite::pad`]. Alignment padding
    /// for the GPU (align to 8 bytes); no shader reads it.
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub fade: EdgeFadeParams,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasTextureId, AtlasTextureKind, DevicePixels, TileId, px};

    fn sprite(tile_id: u32) -> MonochromeSprite {
        let bounds = Bounds {
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
            size: Size {
                width: ScaledPixels(20.),
                height: ScaledPixels(20.),
            },
        };
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: ContentMask { bounds },
            color: Hsla::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(tile_id),
                padding: 0,
                bounds: Bounds {
                    origin: point(DevicePixels(0), DevicePixels(0)),
                    size: Size {
                        width: DevicePixels(20),
                        height: DevicePixels(20),
                    },
                },
            },
            transformation: TransformationMatrix::unit(),
        }
    }

    fn spin_animation(base: TransformationMatrix) -> SpriteAnimation {
        SpriteAnimation {
            animation: PresentAnimation::spin(
                Duration::from_millis(1000),
                Duration::from_millis(33),
            ),
            base,
            pivot: point(ScaledPixels(10.), ScaledPixels(10.)),
            color: Hsla::default(),
            position: 0.,
            extent: Size::default(),
        }
    }

    fn assert_close(actual: Point<Pixels>, expected: Point<Pixels>) {
        assert!(
            (actual.x.0 - expected.x.0).abs() < 1e-3 && (actual.y.0 - expected.y.0).abs() < 1e-3,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn spin_matrix_follows_the_wall_clock() {
        let animation = spin_animation(TransformationMatrix::unit());
        // A whole number of periods since the epoch: no rotation.
        assert_close(
            animation
                .transformation_at(Duration::from_millis(1_700_000_000_000))
                .apply(point(px(20.), px(10.))),
            point(px(20.), px(10.)),
        );
        // A quarter period later: a quarter turn clockwise around the pivot
        // (y grows downward), so the point right of the pivot moves below it.
        let quarter = animation.transformation_at(Duration::from_millis(1_700_000_000_250));
        assert_close(
            quarter.apply(point(px(20.), px(10.))),
            point(px(10.), px(20.)),
        );
        assert_close(
            quarter.apply(point(px(10.), px(10.))),
            point(px(10.), px(10.)),
        );
        // Half a period: opposite side.
        assert_close(
            animation
                .transformation_at(Duration::from_millis(1_700_000_000_500))
                .apply(point(px(20.), px(10.))),
            point(px(0.), px(10.)),
        );

        // The static transformation it was painted with still applies, after
        // the spin.
        let shifted = spin_animation(
            TransformationMatrix::unit().translate(point(ScaledPixels(5.), ScaledPixels(0.))),
        );
        assert_close(
            shifted
                .transformation_at(Duration::from_millis(1_700_000_000_250))
                .apply(point(px(20.), px(10.))),
            point(px(15.), px(20.)),
        );

        // A zero period never moves and never divides by zero.
        let still = SpriteAnimation {
            animation: PresentAnimation::spin(Duration::ZERO, Duration::from_millis(33)),
            ..animation
        };
        assert!(still.animation.is_static());
        assert_eq!(
            still.transformation_at(Duration::from_millis(1_700_000_000_250)),
            TransformationMatrix::unit()
        );
    }

    #[test]
    fn travel_moves_and_scales_around_the_sprite_center() {
        fn path(phase: f32) -> TravelPose {
            TravelPose {
                offset: point(phase, -phase),
                scale: 1. + phase,
            }
        }
        let travel = SpriteAnimation {
            animation: PresentAnimation::travel(
                Duration::from_millis(1000),
                path,
                Duration::from_millis(33),
            ),
            pivot: point(ScaledPixels(10.), ScaledPixels(5.)),
            extent: size(ScaledPixels(20.), ScaledPixels(10.)),
            ..spin_animation(TransformationMatrix::unit())
        };
        // Half way: moved by half its width right and half its height up,
        // and 1.5 times its size around the moved center.
        let at = travel.transformation_at(Duration::from_millis(1_700_000_000_500));
        assert_eq!(at.apply(point(px(10.), px(5.))), point(px(20.), px(0.)));
        assert_eq!(at.apply(point(px(20.), px(5.))), point(px(35.), px(0.)));
        // Start of the trip: where it was painted.
        assert_eq!(
            travel.transformation_at(Duration::from_millis(1_700_000_000_000)),
            TransformationMatrix::unit()
                .translate(point(ScaledPixels(10.), ScaledPixels(5.)))
                .scale(size(1., 1.))
                .translate(point(ScaledPixels(-10.), ScaledPixels(-5.)))
        );
        // A travel never touches the color.
        assert_eq!(
            travel.color_at(Duration::from_millis(1_700_000_000_500)),
            travel.color
        );
    }

    #[test]
    fn pulse_alpha_follows_the_wall_clock() {
        let painted = Hsla {
            h: 0.,
            s: 0.,
            l: 1.,
            a: 0.5,
        };
        let pulse = SpriteAnimation {
            animation: PresentAnimation::pulse(
                Duration::from_millis(1000),
                Duration::from_millis(33),
            ),
            color: painted,
            ..spin_animation(TransformationMatrix::unit())
        };
        let alpha = |ms: u64| pulse.color_at(Duration::from_millis(ms)).a;
        // Start of a cycle: invisible. Half way: the painted alpha. Quarter: half of it.
        assert!(alpha(1_700_000_000_000).abs() < 1e-4);
        assert!((alpha(1_700_000_000_500) - 0.5).abs() < 1e-4);
        assert!((alpha(1_700_000_000_250) - 0.25).abs() < 1e-4);
        // A pulse never moves the sprite, and a spin never recolors it.
        assert_eq!(
            pulse.transformation_at(Duration::from_millis(1_700_000_000_250)),
            TransformationMatrix::unit()
        );
        let mut spin = spin_animation(TransformationMatrix::unit());
        spin.color = painted;
        assert_eq!(
            spin.color_at(Duration::from_millis(1_700_000_000_250)),
            painted
        );

        // Resolving a scene applies it to the sprite the renderer draws.
        let mut scene = Scene::default();
        let mut pulsing = sprite(1);
        pulsing.color = painted;
        scene.insert_animated_monochrome_sprite(pulsing, pulse);
        scene.finish();
        scene.resolve_present_animations(Duration::from_millis(1_700_000_000_250));
        assert!((scene.monochrome_sprites[0].color.a - 0.25).abs() < 1e-4);
    }

    /// Resolve at a quarter turn and return (tile id, transformed probe point)
    /// for every monochrome sprite.
    fn resolved(scene: &mut Scene) -> Vec<(u32, Point<Pixels>)> {
        scene.resolve_present_animations(Duration::from_millis(1_700_000_000_250));
        scene
            .monochrome_sprites
            .iter()
            .map(|sprite| {
                (
                    sprite.tile.tile_id.0,
                    sprite.transformation.apply(point(px(20.), px(10.))),
                )
            })
            .collect()
    }

    fn assert_only_tile_7_spins(resolved: &[(u32, Point<Pixels>)]) {
        assert_eq!(resolved.len(), 2);
        for (tile_id, probe) in resolved {
            if *tile_id == 7 {
                assert_close(*probe, point(px(10.), px(20.)));
            } else {
                assert_close(*probe, point(px(20.), px(10.)));
            }
        }
    }

    #[test]
    fn spin_survives_sorting_and_replay() {
        let full = Bounds {
            origin: point(ScaledPixels(0.), ScaledPixels(0.)),
            size: Size {
                width: ScaledPixels(100.),
                height: ScaledPixels(100.),
            },
        };
        let mut scene = Scene::default();
        scene.push_layer(full);
        // Inserted first, but sorts after the static sprite (higher tile id).
        scene.insert_animated_monochrome_sprite(
            sprite(7),
            spin_animation(TransformationMatrix::unit()),
        );
        scene.insert_primitive(sprite(1));
        scene.pop_layer();
        scene.finish();
        assert_eq!(scene.monochrome_sprites[0].tile.tile_id, TileId(1));
        assert!(scene.has_present_animations());
        assert_eq!(
            scene.present_animation_interval(),
            Some(Duration::from_millis(33))
        );
        assert_only_tile_7_spins(&resolved(&mut scene));

        // A cached view's paint range is replayed into the next scene.
        let mut next = Scene::default();
        next.replay(0..scene.paint_operations.len(), &scene);
        next.finish();
        assert!(next.has_present_animations());
        assert_only_tile_7_spins(&resolved(&mut next));

        // A platform that splits the presented scene by replaying it (the
        // macOS overlay plane) sees the resolved rotation too.
        let mut split = Scene::default();
        split.replay(0..scene.paint_operations.len(), &scene);
        split.finish();
        let split_probes = split
            .monochrome_sprites
            .iter()
            .map(|sprite| {
                (
                    sprite.tile.tile_id.0,
                    sprite.transformation.apply(point(px(20.), px(10.))),
                )
            })
            .collect::<Vec<_>>();
        assert_only_tile_7_spins(&split_probes);

        // Replaying only the static part carries no animation.
        let mut static_only = Scene::default();
        static_only.replay(2..3, &scene);
        static_only.finish();
        assert!(!static_only.has_present_animations());

        // A fully clipped animated sprite leaves nothing behind.
        let mut clipped = Scene::default();
        let mut hidden = sprite(9);
        hidden.content_mask.bounds.size = Size::default();
        clipped.insert_animated_monochrome_sprite(
            hidden,
            spin_animation(TransformationMatrix::unit()),
        );
        assert!(!clipped.has_present_animations());

        scene.clear();
        assert!(!scene.has_present_animations());
    }

    /// A highlight band a quarter of the text wide on each side, centered on
    /// `phase`: full at the center, fading linearly to nothing.
    fn band(phase: f32, x: f32) -> f32 {
        (1. - (x - phase).abs() * 4.).max(0.)
    }

    fn opaque(r: f32, g: f32, b: f32) -> Hsla {
        Hsla::from(Rgba { r, g, b, a: 1. })
    }

    fn assert_rgba_close(actual: Hsla, expected: Rgba) {
        let actual = Rgba::from(actual);
        assert!(
            (actual.r - expected.r).abs() < 1e-3
                && (actual.g - expected.g).abs() < 1e-3
                && (actual.b - expected.b).abs() < 1e-3
                && (actual.a - expected.a).abs() < 1e-3,
            "{actual:?} != {expected:?}"
        );
    }

    fn sweep_animation(position: f32) -> SpriteAnimation {
        SpriteAnimation {
            animation: PresentAnimation::sweep(
                Duration::from_millis(1000),
                opaque(1., 1., 1.),
                band,
                Duration::from_millis(33),
            ),
            color: opaque(0., 0., 0.),
            position,
            ..spin_animation(TransformationMatrix::unit())
        }
    }

    fn subpixel_sprite(tile_id: u32) -> SubpixelSprite {
        let sprite = sprite(tile_id);
        SubpixelSprite {
            order: 0,
            pad: 0,
            bounds: sprite.bounds,
            content_mask: sprite.content_mask,
            color: sprite.color,
            tile: sprite.tile,
            transformation: sprite.transformation,
        }
    }

    #[test]
    fn sweep_color_follows_the_wall_clock_and_the_glyph_position() {
        let gray = |v: f32| Rgba {
            r: v,
            g: v,
            b: v,
            a: 1.,
        };
        // Half way through the period the band is centered on the text.
        let half = Duration::from_millis(1_700_000_000_500);
        assert_eq!(sweep_animation(0.5).color_at(half), opaque(1., 1., 1.));
        assert_rgba_close(sweep_animation(0.375).color_at(half), gray(0.5));
        assert_eq!(sweep_animation(0.).color_at(half), opaque(0., 0., 0.));
        // A quarter later the band has moved on to the end of the text.
        let three_quarters = Duration::from_millis(1_700_000_000_750);
        assert_eq!(
            sweep_animation(0.5).color_at(three_quarters),
            opaque(0., 0., 0.)
        );
        assert_eq!(
            sweep_animation(0.75).color_at(three_quarters),
            opaque(1., 1., 1.)
        );
        // A sweep recolors, it never moves the glyph.
        assert_eq!(
            sweep_animation(0.5).transformation_at(half),
            TransformationMatrix::unit()
        );

        // A zero period is static and keeps the painted color.
        let mut still = sweep_animation(0.5);
        still.animation =
            PresentAnimation::sweep(Duration::ZERO, opaque(1., 1., 1.), band, Duration::ZERO);
        assert!(still.animation.is_static());
        assert_eq!(still.color_at(half), opaque(0., 0., 0.));

        // The highlight takes the opacity the glyph was painted with.
        let mut faded = sweep_animation(0.5);
        faded.animation = faded.animation.with_opacity(0.5);
        faded.color = faded.color.opacity(0.5);
        assert_rgba_close(
            faded.color_at(half),
            Rgba {
                r: 1.,
                g: 1.,
                b: 1.,
                a: 0.5,
            },
        );
        let mut faded_edge = sweep_animation(0.375);
        faded_edge.animation = faded_edge.animation.with_opacity(0.5);
        faded_edge.color = faded_edge.color.opacity(0.5);
        assert_rgba_close(
            faded_edge.color_at(half),
            Rgba {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 0.5,
            },
        );
    }

    #[test]
    fn sweep_mix_interpolates_with_premultiplied_alpha() {
        let red = opaque(1., 0., 0.);
        let clear_white = Hsla::from(Rgba {
            r: 1.,
            g: 1.,
            b: 1.,
            a: 0.,
        });
        // From transparent, the hue is the opaque end's: red at half alpha,
        // not pink.
        assert_rgba_close(
            mix_premultiplied(clear_white, red, 0.5),
            Rgba {
                r: 1.,
                g: 0.,
                b: 0.,
                a: 0.5,
            },
        );
        assert_eq!(mix_premultiplied(clear_white, red, 0.), clear_white);
        assert_eq!(mix_premultiplied(clear_white, red, -1.), clear_white);
        assert_eq!(mix_premultiplied(clear_white, red, 1.), red);
        assert_eq!(mix_premultiplied(clear_white, red, 2.), red);
        // Both ends transparent: transparent, with the target's hue.
        let clear_red = red.opacity(0.);
        assert_eq!(mix_premultiplied(clear_white, clear_red, 0.5).a, 0.);
    }

    #[test]
    fn sweep_glyphs_resolve_and_survive_replay_in_both_sprite_kinds() {
        let half = Duration::from_millis(1_700_000_000_500);
        let mut scene = Scene::default();
        // Grayscale glyph at the start of the text, subpixel glyph in the
        // middle, and a static subpixel glyph.
        scene.insert_animated_monochrome_sprite(sprite(1), sweep_animation(0.));
        scene.insert_animated_subpixel_sprite(subpixel_sprite(2), sweep_animation(0.5));
        scene.insert_primitive(subpixel_sprite(3));
        scene.finish();
        assert!(scene.has_present_animations());
        assert_eq!(scene.subpixel_sprites.len(), 2);

        let colors = |scene: &mut Scene| {
            scene.resolve_present_animations(half);
            let mut colors = scene
                .monochrome_sprites
                .iter()
                .map(|sprite| (sprite.tile.tile_id.0, sprite.color))
                .chain(
                    scene
                        .subpixel_sprites
                        .iter()
                        .map(|sprite| (sprite.tile.tile_id.0, sprite.color)),
                )
                .collect::<Vec<_>>();
            colors.sort_by_key(|(tile_id, _)| *tile_id);
            colors
        };
        let expected = vec![
            (1, opaque(0., 0., 0.)),
            (2, opaque(1., 1., 1.)),
            (3, Hsla::default()),
        ];
        assert_eq!(colors(&mut scene), expected);

        // The recorded operations are resolved too (the macOS overlay plane
        // replays them).
        let subpixel_ops = scene
            .paint_operations
            .iter()
            .filter_map(|operation| match operation {
                PaintOperation::Primitive(Primitive::SubpixelSprite(sprite)) => {
                    Some((sprite.tile.tile_id.0, sprite.color))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            subpixel_ops,
            vec![(2, opaque(1., 1., 1.)), (3, Hsla::default())]
        );

        // A cached view replayed into the next scene keeps every glyph's
        // animation, subpixel included.
        let mut next = Scene::default();
        next.replay(0..scene.paint_operations.len(), &scene);
        next.finish();
        assert_eq!(next.sprite_animations.len(), 2);
        assert_eq!(colors(&mut next), expected);
        // And it keeps following the clock: a quarter later the band has
        // left the middle glyph.
        next.resolve_present_animations(Duration::from_millis(1_700_000_000_750));
        let middle = next
            .subpixel_sprites
            .iter()
            .find(|sprite| sprite.tile.tile_id.0 == 2)
            .unwrap();
        assert_eq!(middle.color, opaque(0., 0., 0.));

        // Replaying only the static glyph carries no animation.
        let mut static_only = Scene::default();
        static_only.replay(2..3, &scene);
        static_only.finish();
        assert!(!static_only.has_present_animations());

        // A fully clipped animated subpixel glyph leaves nothing behind.
        let mut clipped = Scene::default();
        let mut hidden = subpixel_sprite(9);
        hidden.content_mask.bounds.size = Size::default();
        clipped.insert_animated_subpixel_sprite(hidden, sweep_animation(0.5));
        assert!(!clipped.has_present_animations());
    }
}
