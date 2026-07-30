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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var text_texture: texture_2d<f32>;

const MASK_SPAWN_ATTEMPTS: u32 = 12u;

fn hash_u32(value: u32) -> u32 {
  var x = value + 0x9e3779b9u;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn random01(value: u32) -> f32 {
  return f32(hash_u32(value) >> 8u) * (1.0 / 16777216.0);
}

fn rand(n: vec2f) -> f32 {
  let key = bitcast<u32>(n.x)
    ^ hash_u32(bitcast<u32>(n.y))
    ^ hash_u32(bitcast<u32>(params.clock.z));
  return random01(key);
}

fn loop4(p: vec4f) -> vec4f {
  let scale = params.clock.w;
  return vec4f(
    fract(p.xy / scale) * scale,
    fract(p.zw / scale) * scale
  );
}

fn loop3(p: vec3f) -> vec3f {
  let scale = params.clock.w;
  return vec3f(fract(p.xy / scale) * scale, p.z);
}

fn mod289v(x: vec4f) -> vec4f {
  return x - floor(x * (1.0 / (289.0 + params.clock.z))) * (289.0 + params.clock.z);
}

fn perm(x: vec4f) -> vec4f {
  return mod289v(((x * 34.0) + 1.0) * x);
}

fn noise(p: vec3f) -> f32 {
  let a = floor(p);
  var d = p - a;
  d = d * d * (3.0 - 2.0 * d);

  let b = a.xxyy + vec4f(0.0, 1.0, 0.0, 1.0);
  let k1 = perm(loop4(b.xyxy));
  let k2 = perm(loop4(k1.xyxy + b.zzww));

  let c = k2 + a.zzzz;
  let k3 = perm(c);
  let k4 = perm(c + 1.0);

  let o3 = fract(k4 / 41.0) * d.z + fract(k3 / 41.0) * (1.0 - d.z);
  let o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);
  return o4.y * d.y + o4.x * (1.0 - d.y);
}

fn grad(p: vec3f) -> vec3f {
  let e = vec2f(0.1, 0.0);
  return vec3f(
    noise(loop3(p + e.xyy)) - noise(loop3(p - e.xyy)),
    noise(loop3(p + e.yxy)) - noise(loop3(p - e.yxy)),
    noise(loop3(p + e.yyx)) - noise(loop3(p - e.yyx))
  ) / (2.0 * e.x);
}

fn curl_noise(p: vec3f) -> vec3f {
  var uv = p.xy / params.viewport.xy;
  uv = vec2f(uv.x * (params.viewport.x / params.viewport.y), uv.y);
  uv = fract(uv) * params.clock.w;
  let q = vec3f(uv, p.z);
  let e = vec2f(0.01, 0.0);
  return grad(loop3(q)).yzx - vec3f(
    grad(loop3(q + e.yxy)).z,
    grad(loop3(q + e.yyx)).x,
    grad(loop3(q + e.xyy)).y
  );
}

fn text_alpha(pos: vec2f) -> f32 {
  if (any(pos < vec2f(0.0)) || any(pos >= params.viewport.xy)) {
    return 0.0;
  }

  let dimensions = textureDimensions(text_texture);
  let size = vec2f(dimensions);
  let uv = vec2f(pos.x, params.viewport.y - pos.y) / params.viewport.xy;
  let coordinate = clamp(
    vec2i(uv * size),
    vec2i(0),
    vec2i(dimensions) - vec2i(1)
  );
  return textureLoad(text_texture, coordinate, 0).a;
}

fn spawn_candidate(index: u32, attempt: u32) -> vec2f {
  let count = max(params.system.x, 1u);
  let aspect = params.viewport.x / max(params.viewport.y, 1.0);
  let columns = max(1u, u32(ceil(sqrt(f32(count) * aspect))));
  let rows = (count + columns - 1u) / columns;
  let cell_count = columns * rows;

  let key = hash_u32(
    index
      ^ hash_u32(attempt + 1u)
      ^ hash_u32(params.system.z)
      ^ bitcast<u32>(params.clock.z)
  );
  let cell_index = select(
    hash_u32(key ^ 0xa511e9b3u) % cell_count,
    index % cell_count,
    attempt == 0u
  );
  let cell = vec2u(cell_index % columns, cell_index / columns);
  let jitter = vec2f(
    random01(key),
    random01(key ^ 0x68bc21ebu)
  );

  return (vec2f(cell) + jitter) / vec2f(f32(columns), f32(rows));
}

fn genpos(index: u32) -> vec2f {
  if (params.source.z <= 0.0) {
    return spawn_candidate(index, 0u) * params.viewport.xy;
  }

  for (var attempt = 0u; attempt < MASK_SPAWN_ATTEMPTS; attempt = attempt + 1u) {
    let position = spawn_candidate(index, attempt) * params.viewport.xy;
    if (text_alpha(position) >= 0.3) {
      return position;
    }
  }

  return vec2f(-1.0);
}

@compute @workgroup_size(128)
fn simulate(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x;
  if (index >= params.system.x) {
    return;
  }

  let current = particles[index];
  var position = current.position;
  var velocity = current.velocity;
  var particle_time = f32(current.life.x)
    + params.clock.y * f32(current.life.y) / max(params.lifecycle.z, 0.00001);
  var particle_duration = f32(current.life.y);
  var particle_alpha = f32(current.life.z);

  if (params.system.y == 1u) {
    let n = f32(index);
    particle_time = rand(vec2f(-94.3, 83.9) * vec2f(n));
    particle_duration = 0.5 + 2.0 * rand(vec2f(n) + params.clock.z * 32.4);
    position = genpos(index);
    velocity = vec2f(0.0);
    if (position.x < 0.0) {
      particle_time = 0.0;
      particle_alpha = -1.0;
    } else if (params.source.z > 0.5) {
      particle_alpha = text_alpha(position);
    } else {
      particle_alpha = 1.0;
    }
  } else if (particle_alpha < 0.0) {
    position = genpos(index);
    velocity = vec2f(0.0);
    if (position.x >= 0.0) {
      particle_time = 0.0;
      particle_duration = 0.5 + 2.0 * rand(vec2f(f32(index)) + position);
      particle_alpha = text_alpha(position);
    }
  } else if (particle_time >= 1.0) {
    particle_time = 0.0;
    if (params.source.z > 0.5) {
      position = genpos(index);
      if (position.x < 0.0) {
        particle_alpha = -1.0;
      } else {
        particle_duration = 0.5 + 2.0 * rand(vec2f(f32(index)) + position);
        particle_alpha = text_alpha(position);
      }
    } else {
      position = genpos(index);
      particle_duration = 0.5 + 2.0 * rand(vec2f(f32(index)) + position);
      particle_alpha = 1.0;
    }
    velocity = vec2f(0.0);
  }

  if (particle_alpha < 0.0) {
    particles[index].position = vec2f(-1.0);
    particles[index].velocity = vec2f(0.0);
    particles[index].life = vec4h(
      f16(0.0),
      f16(particle_duration),
      f16(-1.0),
      f16(0.0)
    );
    return;
  }

  var text_velocity_mult = 1.0;
  let min_size = min(params.viewport.x, params.viewport.y);
  if (params.source.z > 0.5) {
    let inside_text = text_alpha(position);
    text_velocity_mult = inside_text;
    particle_alpha = clamp(
      particle_alpha + (inside_text - 0.75) * params.clock.y * 3.0,
      0.0,
      1.0
    );
    if (params.lifecycle.w > 0.0) {
      particle_alpha = particle_alpha * mix(1.0, inside_text, params.lifecycle.w);
    }
  }

  let curl = curl_noise(vec3f(
    position + params.clock.x * (params.motion.y / 100.0 * min_size),
    params.clock.x * params.motion.x + rand(position) * 2.5
  )).xy;
  let force = curl / max(length(curl), 0.00001);

  velocity = velocity + force * params.motion.z * params.clock.y * min_size * 0.1;
  velocity = velocity * params.motion.w;
  let velocity_length = length(velocity);
  let max_velocity_px = params.lifecycle.y / 100.0 * min_size;
  if (velocity_length > max_velocity_px) {
    velocity = velocity / velocity_length * max_velocity_px;
  }

  if (params.lifecycle.w > 0.0) {
    let vector = position - params.source.xy;
    let distance = length(vector);
    let direction = vector / max(distance, 0.00001);
    let push = 0.9 * max(
      0.0,
      1.0 - max(
        0.0,
        distance / max(params.viewport.x, params.viewport.y) / params.lifecycle.w - 1.0
      )
    );
    position = position + direction * params.clock.y * 1000.0 * push;
  }

  position = position + velocity * params.lifecycle.x * text_velocity_mult * params.clock.y;

  let outside = position.x < 0.0 || position.y < 0.0
    || position.x > params.viewport.x || position.y > params.viewport.y;
  if (outside && params.lifecycle.w < 0.1) {
    particle_time = 0.0;
    position = genpos(index);
    velocity = vec2f(0.0);
    if (position.x < 0.0) {
      particle_alpha = -1.0;
    } else {
      particle_duration = 0.5 + 2.0 * rand(vec2f(f32(index)) + position);
      particle_alpha = select(1.0, text_alpha(position), params.source.z > 0.5);
    }
  }

  if (particle_alpha < 0.0) {
    particles[index].position = vec2f(-1.0);
    particles[index].velocity = vec2f(0.0);
    particles[index].life = vec4h(
      f16(0.0),
      f16(particle_duration),
      f16(-1.0),
      f16(0.0)
    );
    return;
  }

  let render_alpha = sin(particle_time * 3.14)
    * (1.0 - params.lifecycle.w)
    * particle_alpha
    * (0.6 + 0.4 * rand(vec2f(f32(index))));

  particles[index].position = position;
  particles[index].velocity = velocity;
  particles[index].life = vec4h(
    f16(particle_time),
    f16(particle_duration),
    f16(particle_alpha),
    f16(render_alpha)
  );
}
