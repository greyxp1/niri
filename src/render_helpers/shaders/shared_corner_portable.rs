use std::fmt::Write;

pub const MAX_SOURCES: usize = 7;
pub const NO_NATIVE: u8 = u8::MAX;
pub const NATIVE_ONLY: u8 = u8::MAX - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderKey {
    pub flags: [u8; MAX_SOURCES],
    pub count: u8,
    pub native: u8,
    pub native_flags: u8,
    pub background_mask: u8,
    pub appearance: bool,
}

pub(crate) const SOLID: u8 = 1;
pub(crate) const FORCE_ALPHA: u8 = 4;
pub(crate) const EFFECT: u8 = 8;
pub(crate) const RESIZE: u8 = 16;
pub(crate) const EXTERNAL: u8 = 32;
pub(crate) const BACKGROUND: u8 = 64;
pub(crate) const ATLAS: u8 = 128;

impl ShaderKey {
    pub(crate) fn uses_resize(self) -> bool {
        self.flags.iter().any(|flags| flags & RESIZE != 0) || self.native_flags & RESIZE != 0
    }
}

fn sampler(flags: u8) -> &'static str {
    if flags & EXTERNAL != 0 {
        "samplerExternalOES"
    } else {
        "sampler2D"
    }
}

fn sample(index: usize, flags: u8) -> String {
    if flags & ATLAS != 0 {
        "texture2D(niri_source_texture_0, niri_atlas_uv())".into()
    } else if flags & RESIZE != 0 {
        format!("niri_shared_resize(niri_uv_{index})")
    } else if flags & BACKGROUND != 0 {
        "niri_background(niri_p)".into()
    } else if flags & SOLID != 0 {
        format!("niri_color_{index}")
    } else {
        format!("texture2D(niri_source_texture_{index}, niri_uv_{index})")
    }
}

pub fn generate(key: ShaderKey, resize: Option<&str>) -> (String, String) {
    let count = usize::from(key.count);
    assert!((1..=MAX_SOURCES).contains(&count));
    assert!(
        usize::from(key.native) < count || key.native == NO_NATIVE || key.native == NATIVE_ONLY
    );
    let atlas = key.flags[0] & ATLAS != 0;
    let external = key.flags[..count].iter().any(|f| f & EXTERNAL != 0)
        || (atlas && !key.appearance && key.native_flags & EXTERNAL != 0);
    let mut vertex = String::from(
        "precision highp float;\n\
         attribute vec2 vert;\nattribute vec4 vert_position;\n\
         uniform mat3 matrix;\nuniform mat3 tex_matrix;\nuniform float niri_scale;\n\
         uniform vec4 niri_window;\nvarying vec2 niri_p;\nvarying float niri_part;\n\
         #if defined(DEBUG_FLAGS)\nuniform float niri_tint;\nvarying float niri_shared_tint;\n#endif\n",
    );
    vertex.push_str("uniform mat3 niri_matrix;\n");
    if key.appearance {
        vertex.push_str("uniform vec4 niri_atlas_mapping;\nuniform mat3 niri_output_matrix;\nvarying vec2 niri_output_position;\n");
    }
    let mut fragment = String::new();
    if external {
        fragment.push_str("#extension GL_OES_EGL_image_external : require\n");
    }
    fragment.push_str(
        "precision highp float;\n\
         varying vec2 niri_p;\nvarying float niri_part;\n\
         uniform vec4 niri_window;\nuniform vec4 niri_radius;\n\
         uniform mat4 niri_decorations[3];\n\
         #if defined(DEBUG_FLAGS)\nvarying float niri_shared_tint;\n#define niri_tint niri_shared_tint\n#endif\n\
         #define niri_scale niri_post_0.z\n",
    );
    if key.appearance {
        fragment.push_str("varying vec2 niri_output_position;\n");
    }
    for index in 0..count {
        writeln!(vertex, "uniform vec4 niri_rect_{index};\nuniform mat3 niri_source_matrix_{index};\nvarying vec2 niri_uv_{index};\nvarying vec2 niri_local_{index};").unwrap();
        writeln!(fragment, "varying vec2 niri_uv_{index};\nvarying vec2 niri_local_{index};\nuniform vec4 niri_color_{index};\nuniform vec4 niri_post_{index};").unwrap();
        if key.flags[index] & (SOLID | RESIZE | BACKGROUND) == 0 {
            writeln!(
                fragment,
                "uniform highp {} niri_source_texture_{index};",
                sampler(key.flags[index])
            )
            .unwrap();
        }
    }
    if atlas && !key.appearance && key.native != NO_NATIVE {
        vertex.push_str("uniform mat3 niri_native_matrix;\nvarying vec2 niri_native_uv;\n");
        fragment.push_str("varying vec2 niri_native_uv;\n");
        if key.native_flags & (SOLID | RESIZE) == 0 {
            writeln!(
                fragment,
                "uniform {} niri_native_texture;",
                sampler(key.native_flags)
            )
            .unwrap();
        }
    }
    if atlas && !key.appearance {
        fragment.push_str("vec2 niri_atlas_uv() {\nvec2 local = niri_p - niri_window.xy;\nvec2 quadrant = step(niri_window.zw * .5, local);\nvec2 slot = niri_post_0.xy;\nif (slot.x < 0.) return niri_uv_0;\nif (slot.x == 0.) return local / niri_window.zw;\nreturn (local - quadrant * (niri_window.zw - slot) + quadrant * slot) / (2. * slot);\n}\n");
    }
    vertex.push_str("void main() {\nvec3 position = vec3(vert * vec2(abs(vert_position.z), vert_position.w) + vert_position.xy, 1.);\nniri_part = vert_position.z < 0. ? 1. : 0.;\n#if defined(DEBUG_FLAGS)\nniri_shared_tint = niri_tint;\n#endif\n");
    if key.appearance {
        vertex.push_str("vec2 slot = max(abs(niri_atlas_mapping.xy), vec2(1.));\nvec2 quadrant = floor((vert_position.xy + vec2(.5)) / slot);\nvec2 p = niri_atlas_mapping.x < 0. ? (niri_matrix * position).xy : niri_window.xy + (position.xy - quadrant * slot + quadrant * (niri_atlas_mapping.zw - slot)) / niri_scale;\nniri_output_position = (niri_output_matrix * vec3(p, 1.)).xy;\n");
    } else {
        vertex.push_str("vec2 p = (niri_matrix * vec3((tex_matrix * position).xy, 1.)).xy;\n");
    }
    vertex.push_str("niri_p = p;\n");
    for index in 0..count {
        writeln!(vertex, "niri_uv_{index} = (niri_source_matrix_{index} * vec3(p, 1.)).xy;\nniri_local_{index} = (p - niri_rect_{index}.xy) / niri_rect_{index}.zw;").unwrap();
    }
    if atlas && !key.appearance && key.native != NO_NATIVE {
        vertex.push_str("niri_native_uv = (niri_native_matrix * vec3(p, 1.)).xy;\n");
    }
    vertex.push_str("gl_Position = vec4(matrix * position, 1.);\n}\n");

    fragment.push_str(include_str!("border_color.frag"));
    fragment.push_str(include_str!("rounding_alpha.frag"));
    fragment.push_str(include_str!("postprocess_common.frag"));
    // Constant matrix indices also work on implementations that cannot index
    // uniform arrays dynamically in fragment shaders.
    for decoration in 0..2 {
        for (field, column) in ["rect", "radius", "from", "to", "grad", "options"]
            .iter()
            .enumerate()
            .map(|(column, field)| (field, decoration * 6 + column))
        {
            writeln!(
                fragment,
                "#define niri_d{decoration}_{field} niri_decorations[{}][{}]",
                column / 4,
                column % 4
            )
            .unwrap();
        }
        writeln!(fragment, "vec4 niri_gradient_{decoration}(vec2 p) {{ return niri_shared_gradient_color(p - niri_d{decoration}_rect.xy, niri_d{decoration}_grad.xy, niri_d{decoration}_grad.z, niri_d{decoration}_options.xy, niri_d{decoration}_from, niri_d{decoration}_to, niri_d{decoration}_options.z, niri_d{decoration}_options.w) * niri_d{decoration}_grad.w; }}").unwrap();
    }
    let background = if key.background_mask & 2 != 0 { 1 } else { 0 };
    writeln!(
        fragment,
        "vec4 niri_background(vec2 p) {{ return niri_gradient_{background}(p); }}"
    )
    .unwrap();
    fragment.push_str("float niri_coverage(vec2 p, vec4 rect, vec4 radius) {\nvec2 local = p - rect.xy;\nif (any(lessThan(local, vec2(0.))) || any(greaterThan(local, rect.zw))) return 0.;\nreturn niri_rounding_alpha(local, rect.zw, radius);\n}\n");
    if let Some(resize) = resize {
        fragment.push_str("uniform vec2 niri_shared_resize_size;\nuniform float niri_shared_resize_alpha;\nvec2 niri_shared_resize_coords;\n");
        let prelude = include_str!("resize_prelude.frag")
            .replace("precision highp float;", "")
            .replace("varying vec2 niri_v_coords;", "")
            .replace("uniform vec2 niri_size;", "")
            .replace("uniform float niri_alpha;", "")
            .replace("uniform float niri_scale;", "")
            .replace(
                "#if defined(DEBUG_FLAGS)\nuniform float niri_tint;\n#endif",
                "",
            );
        fragment.push_str(&prelude);
        fragment.push_str("\n#define niri_v_coords niri_shared_resize_coords\n#define niri_size niri_shared_resize_size\n#define niri_alpha niri_shared_resize_alpha\n");
        if key.appearance {
            fragment.push_str("\n#define gl_FragCoord vec4(niri_output_position, .5, 1.)\n");
        }
        fragment.push_str(resize);
        if key.appearance {
            fragment.push_str("\n#undef gl_FragCoord\n");
        }
        fragment.push_str("\n#undef niri_v_coords\n#undef niri_size\n#undef niri_alpha\nvec4 niri_shared_resize(vec2 uv) { niri_shared_resize_coords = uv; return resize_color(niri_input_to_curr_geo * vec3(uv, 1.), vec3(niri_curr_geo_size, 1.)); }\n");
    }
    fragment.push_str("void main() {\n");
    if !key.appearance && key.native != NO_NATIVE {
        let native = if key.native == NATIVE_ONLY {
            0
        } else {
            usize::from(key.native)
        };
        let flags = if atlas {
            key.native_flags
        } else {
            key.flags[native]
        };
        let expression = if atlas {
            if flags & RESIZE != 0 {
                "niri_shared_resize(niri_native_uv)".into()
            } else if flags & SOLID != 0 {
                "niri_color_0".into()
            } else {
                "texture2D(niri_native_texture, niri_native_uv)".into()
            }
        } else {
            sample(native, flags)
        };
        if key.native == NATIVE_ONLY {
            writeln!(fragment, "vec4 color = {expression};").unwrap();
        } else {
            writeln!(
                fragment,
                "if (niri_part > .5) {{\nvec4 color = {expression};"
            )
            .unwrap();
        }
        if flags & FORCE_ALPHA != 0 {
            fragment.push_str("color.a = 1.;\n");
        }
        writeln!(
            fragment,
            "color *= niri_post_{}.w;",
            if atlas { 0 } else { native }
        )
        .unwrap();
        fragment.push_str("#if defined(DEBUG_FLAGS)\nif (niri_tint == 1.) color = vec4(0., .2, 0., .2) + color * .8;\n#endif\ngl_FragColor = color;\nreturn;\n}\n");
        if key.native == NATIVE_ONLY {
            return (vertex, fragment);
        }
    }
    fragment.push_str("vec4 content = vec4(0.);\nvec4 effects = vec4(0.);\n");
    if !key.appearance {
        fragment.push_str("float inner = niri_coverage(niri_p, niri_window, niri_radius);\n");
    }
    for index in (0..count).rev() {
        let flags = key.flags[index];
        writeln!(fragment, "if ({}content.a < 1. && all(greaterThanEqual(niri_local_{index}, vec2(0.))) && all(lessThan(niri_local_{index}, vec2(1.)))) {{\nvec4 color = {};", if key.appearance { "" } else { "inner > 0. && " }, sample(index, flags)).unwrap();
        if flags & FORCE_ALPHA != 0 {
            fragment.push_str("color.a = 1.;\n");
        }
        if flags & ATLAS == 0 {
            writeln!(fragment, "color *= niri_post_{index}.w;").unwrap();
        }
        if flags & EFFECT != 0 {
            writeln!(fragment, "color.rgb = niri_shared_saturate(color.rgb, niri_post_{index}.y);\ncolor.rgb += (niri_shared_hash12({}) - .5) * niri_post_{index}.x;\ncolor += niri_color_{index} * (1. - color.a);", if key.appearance { "niri_output_position" } else { "gl_FragCoord.xy" }).unwrap();
        }
        let target = if flags & EFFECT != 0 && !key.appearance {
            "effects"
        } else {
            "content"
        };
        writeln!(fragment, "{target} += color * (1. - {target}.a);\n}}").unwrap();
    }
    if key.appearance {
        fragment.push_str("gl_FragColor = content;\n}\n");
    } else {
        if key.background_mask != 0 && !atlas {
            fragment.push_str("vec4 background = niri_background(niri_p);\neffects = background + effects * (1. - background.a);\n");
        }
        fragment.push_str("vec4 resolved = (content + effects * (1. - content.a)) * inner;\n");
        for decoration in 0..2 {
            writeln!(fragment, "if (inner < 1.) {{\nfloat outer = niri_coverage(niri_p, niri_d{decoration}_rect, niri_d{decoration}_radius);\nif (outer > inner) {{\nvec4 color = niri_gradient_{decoration}(niri_p);").unwrap();
            if decoration == 0 && key.background_mask & 2 != 0 {
                fragment.push_str(
                    "vec4 behind = niri_gradient_1(niri_p);\ncolor += behind * (1. - color.a);\n",
                );
            }
            fragment.push_str(
                "resolved += color * max(outer - inner, 0.);\ninner = max(inner, outer);\n}\n}\n",
            );
        }
        fragment.push_str("#if defined(DEBUG_FLAGS)\nif (niri_tint == 1.) resolved = vec4(0., .2, 0., .2) + resolved * .8;\n#endif\ngl_FragColor = resolved;\n}\n");
    }
    (vertex, fragment)
}
