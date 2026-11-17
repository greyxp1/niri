precision highp float;

#if defined(DEBUG_FLAGS)
uniform float niri_tint;
#endif

uniform float niri_alpha;
uniform float niri_scale;

uniform vec2 niri_size;
varying vec2 niri_v_coords;

uniform float colorspace;
uniform float hue_interpolation;
uniform vec4 color_from;
uniform vec4 color_to;
uniform vec2 grad_offset;
uniform float grad_width;
uniform vec2 grad_vec;

uniform mat3 input_to_geo;
uniform vec2 geo_size;
uniform vec4 outer_radius;
uniform float border_width;

vec4 niri_shared_gradient_color(vec2 coords, vec2 grad_offset, float grad_width, vec2 grad_vec, vec4 color_from, vec4 color_to, float colorspace, float hue_interpolation);

float niri_rounding_alpha(vec2 coords, vec2 size, vec4 corner_radius);

void main() {
    vec3 coords_geo = input_to_geo * vec3(niri_v_coords, 1.0);
    vec4 color = niri_shared_gradient_color(coords_geo.xy, grad_offset, grad_width, grad_vec, color_from, color_to, colorspace, hue_interpolation);
    color = color * niri_rounding_alpha(coords_geo.xy, geo_size, outer_radius);

    if (border_width > 0.0) {
        coords_geo -= vec3(border_width);
        vec2 inner_geo_size = geo_size - vec2(border_width * 2.0);
        if (0.0 <= coords_geo.x && coords_geo.x <= inner_geo_size.x
                && 0.0 <= coords_geo.y && coords_geo.y <= inner_geo_size.y)
        {
            vec4 inner_radius = max(outer_radius - vec4(border_width), 0.0);
            color = color * (1.0 - niri_rounding_alpha(coords_geo.xy, inner_geo_size, inner_radius));
        }
    }

    color = color * niri_alpha;

#if defined(DEBUG_FLAGS)
    if (niri_tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
