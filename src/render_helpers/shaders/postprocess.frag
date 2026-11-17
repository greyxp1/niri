uniform float noise;
uniform float saturation;
uniform vec4 bg_color;

vec4 postprocess(vec4 color) {
    if (saturation != 1.0) {
        color.rgb = niri_shared_saturate(color.rgb, saturation);
    }

    if (noise > 0.0) {
        vec2 uv = gl_FragCoord.xy;
        color.rgb += (niri_shared_hash12(uv) - 0.5) * noise;
    }

    // Mix bg_color behind the texture (both premultiplied alpha).
    color = color + bg_color * (1.0 - color.a);

    return color;
}
