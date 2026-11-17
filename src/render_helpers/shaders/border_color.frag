vec4 niri_shared_premul_rect(vec4 color) {
    color.rgb *= color.a;
    return color;
}

vec4 niri_shared_premul_lch(vec4 color) {
    color.xy *= color.a;
    return color;
}

vec4 niri_shared_unpremul_rect(vec4 color) {
    if (color.a == 0.0)
        return color;

    color.rgb /= color.a;
    return color;
}

vec4 niri_shared_unpremul_lch(vec4 color) {
    if (color.a == 0.0)
        return color;

    color.xy /= color.a;
    return color;
}

vec4 niri_shared_premul_mix_unpremul_rect(vec4 color1, vec4 color2, float ratio) {
    vec4 mixed = mix(niri_shared_premul_rect(color1), niri_shared_premul_rect(color2), ratio);
    return niri_shared_unpremul_rect(mixed);
}

vec4 niri_shared_premul_mix_unpremul_lch(vec4 color1, vec4 color2, float ratio) {
    vec4 mixed = mix(niri_shared_premul_lch(color1), niri_shared_premul_lch(color2), ratio);
    return niri_shared_unpremul_lch(mixed);
}

vec3 niri_shared_srgb_to_linear(vec3 color) {
    return pow(color, vec3(2.2));
}

vec3 niri_shared_linear_to_srgb(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

vec3 niri_shared_lab_to_lch(vec3 color) {
    float c = sqrt(pow(color.y, 2.0) + pow(color.z, 2.0));
    float h = degrees(atan(color.z, color.y)) ;
    h += h <= 0.0 ?
        360.0 :
        0.0 ;
    return vec3(
        color.x,
        c,
        h
    );
}

vec3 niri_shared_lch_to_lab(vec3 color) {
    float a = color.y * clamp(cos(radians(color.z)), -1.0, 1.0);
    float b = color.y * clamp(sin(radians(color.z)), -1.0, 1.0);
    return vec3(
        color.x,
        a,
        b
    );
}

vec3 niri_shared_linear_to_oklab(vec3 color){
    mat3 rgb_to_lms = mat3(
        vec3(0.4122214708, 0.5363325363, 0.0514459929),
        vec3(0.2119034982, 0.6806995451, 0.1073969566),
        vec3(0.0883024619, 0.2817188376, 0.6299787005)
    );
    mat3 lms_to_oklab = mat3(
        vec3(0.2104542553, 0.7936177850, -0.0040720468),
        vec3(1.9779984951, -2.4285922050, 0.4505937099),
        vec3(0.0259040371, 0.7827717662, -0.8086757660)
    );
    vec3 lms = color * rgb_to_lms;
    lms = pow(lms, vec3(1.0 / 3.0));
    return lms * lms_to_oklab;
}

vec3 niri_shared_oklab_to_linear(vec3 color){
    mat3 oklab_to_lms = mat3(
        vec3(1.0, 0.3963377774, 0.2158037573),
        vec3(1.0, -0.1055613458, -0.0638541728),
        vec3(1.0, -0.0894841775, -1.2914855480)
    );
    mat3 lms_to_rgb = mat3(
        vec3(4.0767416621, -3.3077115913, 0.2309699292),
        vec3(-1.2684380046, 2.6097574011, -0.3413193965),
        vec3(-0.0041960863, -0.7034186147, 1.7076147010)
    );
    vec3 lms = color * oklab_to_lms;
    lms = pow(lms, vec3(3.0));
    return lms * lms_to_rgb;
}

vec4 niri_shared_color_mix(vec4 color1, vec4 color2, float color_ratio, float colorspace, float hue_interpolation) {
    vec4 color_out;

    // srgb
    if (colorspace == 0.0) {
        return mix(niri_shared_premul_rect(color1), niri_shared_premul_rect(color2), color_ratio);
    }

    color1.rgb = niri_shared_srgb_to_linear(color1.rgb);
    color2.rgb = niri_shared_srgb_to_linear(color2.rgb);

    // srgb-linear
    if (colorspace == 1.0) {
        color_out = niri_shared_premul_mix_unpremul_rect(color1, color2, color_ratio);
    // oklab
    } else if (colorspace == 2.0) {
        color1.xyz = niri_shared_linear_to_oklab(color1.rgb);
        color2.xyz = niri_shared_linear_to_oklab(color2.rgb);
        color_out = niri_shared_premul_mix_unpremul_rect(color1, color2, color_ratio);
        color_out.rgb = niri_shared_oklab_to_linear(color_out.xyz);
    // oklch
    } else if (colorspace == 3.0) {
        color1.xyz = niri_shared_lab_to_lch(niri_shared_linear_to_oklab(color1.rgb));
        color2.xyz = niri_shared_lab_to_lch(niri_shared_linear_to_oklab(color2.rgb));
        color_out = niri_shared_premul_mix_unpremul_lch(color1, color2, color_ratio);

        float min_hue = min(color1.z, color2.z);
        float max_hue = max(color1.z, color2.z);
        float path_direct_distance = (max_hue - min_hue) * color_ratio;
        float path_mod_distance = (360.0 - max_hue + min_hue) * color_ratio;

        float path_mod =
            color1.z == min_hue ?
                mod(color1.z - path_mod_distance, 360.0) :
                mod(color1.z + path_mod_distance, 360.0) ;
        float path_direct =
            color1.z == min_hue ?
                color1.z + path_direct_distance :
                color1.z - path_direct_distance ;

        // shorter
        if (hue_interpolation == 0.0) {
            color_out.z =
                max_hue - min_hue > 360.0 - max_hue + min_hue ?
                    path_mod :
                    path_direct ;
        // longer
        } else if (hue_interpolation == 1.0) {
            color_out.z =
                max_hue - min_hue <= 360.0 - max_hue + min_hue ?
                    path_mod :
                    path_direct ;
        // increasing
        } else if (hue_interpolation == 2.0) {
            color_out.z =
                color1.z > color2.z ?
                    path_mod :
                    path_direct ;
        // decreasing
        } else if (hue_interpolation == 3.0) {
            color_out.z =
                color1.z <= color2.z ?
                    path_mod :
                    path_direct ;
        }
        color_out.rgb = clamp(niri_shared_oklab_to_linear(niri_shared_lch_to_lab(color_out.xyz)), 0.0, 1.0);
    }

    return niri_shared_premul_rect(vec4(niri_shared_linear_to_srgb(color_out.rgb), color_out.a));
}

vec4 niri_shared_gradient_color(vec2 coords, vec2 grad_offset, float grad_width, vec2 grad_vec, vec4 color_from, vec4 color_to, float colorspace, float hue_interpolation) {
    coords = coords + grad_offset;

    if ((grad_vec.x < 0.0 && 0.0 <= grad_vec.y) || (0.0 <= grad_vec.x && grad_vec.y < 0.0))
        coords.x -= grad_width;

    float frac = dot(coords, grad_vec) / dot(grad_vec, grad_vec);

    if (grad_vec.y < 0.0)
        frac += 1.0;

    frac = clamp(frac, 0.0, 1.0);
    return niri_shared_color_mix(color_from, color_to, frac, colorspace, hue_interpolation);
}
