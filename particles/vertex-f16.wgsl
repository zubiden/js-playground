enable f16;

struct Params {
  viewport: vec4f,
  clock: vec4f,
  motion: vec4f,
  lifecycle: vec4f,
  source: vec4f,
  system: vec4u,
  color: vec4f,
}

struct Particle {
  position: vec2f,
  velocity: vec2f,
  life: vec4h,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) alpha: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

@vertex
fn vertex_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32
) -> VertexOutput {
  let corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, 1.0)
  );
  let corner = corners[vertex_index];
  let particle = particles[instance_index];
  let position_px = particle.position + corner * params.viewport.z * 0.5;
  let clip = position_px / params.viewport.xy * 2.0 - vec2f(1.0);

  var output: VertexOutput;
  output.position = vec4f(clip, 0.0, 1.0);
  output.uv = corner;
  output.alpha = f32(particle.life.w);
  return output;
}
