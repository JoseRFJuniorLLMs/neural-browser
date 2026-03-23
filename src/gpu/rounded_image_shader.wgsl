// Rounded image shader — textured quad with SDF pill masking.
// Renders an image clipped to a rounded rectangle (pill shape).

struct VertexInput {
    @location(0) ndc_pos: vec2<f32>,     // Clip-space position
    @location(1) uv: vec2<f32>,          // Texture UV coordinates
    @location(2) pixel_pos: vec2<f32>,   // Position in pixel space
    @location(3) rect_center: vec2<f32>, // Rect center in pixels
    @location(4) rect_half: vec2<f32>,   // Rect half-size in pixels
    @location(5) radius: f32,            // Corner radius in pixels
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) pixel_pos: vec2<f32>,
    @location(2) rect_center: vec2<f32>,
    @location(3) rect_half: vec2<f32>,
    @location(4) radius: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.ndc_pos, 0.0, 1.0);
    out.uv = in.uv;
    out.pixel_pos = in.pixel_pos;
    out.rect_center = in.rect_center;
    out.rect_half = in.rect_half;
    out.radius = in.radius;
    return out;
}

@group(0) @binding(0)
var t_image: texture_2d<f32>;
@group(0) @binding(1)
var s_image: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // SDF for rounded rectangle (same formula as rounded_rect_shader)
    let p = in.pixel_pos - in.rect_center;
    let q = abs(p) - in.rect_half + vec2<f32>(in.radius, in.radius);
    let dist = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - in.radius;

    // Anti-aliased edge (smooth 1.5px feather)
    let alpha = 1.0 - smoothstep(-1.0, 1.0, dist);

    if (alpha < 0.01) {
        discard;
    }

    // Sample texture and apply SDF mask
    let tex_color = textureSample(t_image, s_image, in.uv);
    return vec4<f32>(tex_color.rgb, tex_color.a * alpha);
}
