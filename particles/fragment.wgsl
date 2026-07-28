struct Params {
  viewport: vec4f,
  clock: vec4f,
  motion: vec4f,
  lifecycle: vec4f,
  source: vec4f,
  system: vec4u,
  color: vec4f,
}

struct FragmentInput {
  @location(0) uv: vec2f,
  @location(1) alpha: f32,
}

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  if (dot(input.uv, input.uv) > 1.0) {
    discard;
  }
  return vec4f(params.color.rgb, input.alpha);
}
