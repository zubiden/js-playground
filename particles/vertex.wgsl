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
  pos_vel: vec4f,
  life: vec4f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) alpha: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

fn rand(n: vec2f) -> f32 {
  return fract(sin(dot(n, vec2f(12.9898, 4.1414 - params.clock.z * 0.42))) * 43758.5453);
}

@vertex
fn vertex_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32
) -> VertexOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0)
  );
  let corner = corners[vertex_index];
  let particle = particles[instance_index];
  let position_px = particle.pos_vel.xy + corner * params.viewport.z * 0.5;
  let clip = position_px / params.viewport.xy * 2.0 - vec2f(1.0);

  var output: VertexOutput;
  output.position = vec4f(clip, 0.0, 1.0);
  output.uv = corner;
  output.alpha = sin(particle.life.x * 3.14)
    * (1.0 - params.lifecycle.w)
    * particle.life.z
    * (0.6 + 0.4 * rand(vec2f(f32(instance_index))));
  return output;
}
