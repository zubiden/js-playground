const stats = new Stats();

const canvasParent = document.querySelector("#preview");
const canvas = document.querySelector("canvas.gpu");
const texcanvas = document.querySelector("canvas.tex");

let W;
let H;
let device;
let context;
let format;
let maskTexture;
let uniformBuffer;
let computePipeline;
let renderPipeline;
let particleBuffers = [];
let computeBindGroups = [];
let renderBindGroups = [];
let bufferIndex = 0;
let bufferParticlesCount = 0;
let reset = true;
let running = true;
let time = 0;
let fadeOutTime = 0;
let frameIndex = 0;
let lastDrawTime = window.performance.now();

const resize = () => {
  const sz = Math.min(
    700,
    Math.min(canvasParent.clientWidth, canvasParent.clientHeight) * 0.6,
  );
  texcanvas.style.width =
    texcanvas.style.height =
    canvas.style.width =
    canvas.style.height =
      `${sz}px`;
  canvas.width = texcanvas.width = W = Math.floor(sz * window.devicePixelRatio);
  canvas.height = texcanvas.height = H = Math.floor(sz * window.devicePixelRatio);

  const ctx = texcanvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const scale = W / 500;
  ctx.save();
  ctx.scale(scale, scale);
  ctx.fillStyle = "rgba(255, 0, 0, 1)";
  ctx.fillRect(10, 10, 100, 40);
  ctx.fillRect(120, 10, 80, 40);
  ctx.fillRect(210, 10, 40, 40);
  ctx.fillRect(260, 10, 80, 40);
  ctx.fillRect(10, 60, 80, 40);
  ctx.fillRect(100, 60, 120, 40);
  ctx.fillRect(10, 110, 120, 40);
  ctx.fillRect(140, 110, 30, 40);
  ctx.fillRect(180, 110, 90, 40);
  ctx.fillRect(280, 110, 80, 40);
  ctx.fillRect(370, 110, 120, 40);
  ctx.fillRect(10, 160, 90, 40);
  ctx.fillRect(10, 260, 30, 40);
  ctx.fillRect(50, 260, 90, 40);
  ctx.fillRect(150, 260, 190, 40);
  ctx.fillRect(350, 260, 50, 40);
  ctx.fillStyle = "rgba(255, 0, 0, .5)";
  ctx.fillRect(0, 0, 350, 60);
  ctx.fillRect(0, 60, 230, 40);
  ctx.fillRect(0, 100, 500, 60);
  ctx.fillRect(0, 160, 110, 50);
  ctx.fillRect(0, 250, 410, 60);
  ctx.restore();

  if (device) {
    recreateMaskTexture();
  }
};

window.onresize = resize;
resize();

const GUI = {
  particlesCount: 7000,
  radius: window.devicePixelRatio * 1.6,
  reset: () => {
    reset = true;
  },
  destroy: () => {
    die();
  },
  seed: Math.random() * 10,
  noiseScale: 22,
  noiseSpeed: 0.6,
  forceMult: 0.6,
  velocityMult: 1,
  dampingMult: 0.9999,
  maxVelocity: 6,
  longevity: 1.4,
  noiseMovement: 4,
  timeScale: 1,
  color: 0xffffff,
  fadeOut: false,
  fadeOutXY: [0, 0],
  text: true,
  showTextTexture: false,
};

const createParticleResources = (count) => {
  for (const buffer of particleBuffers) {
    buffer.destroy();
  }

  bufferParticlesCount = Math.max(1, Math.ceil(count));
  const byteSize = bufferParticlesCount * 32;
  particleBuffers = [0, 1].map((index) =>
    device.createBuffer({
      label: `particle-state-${index}`,
      size: byteSize,
      usage: GPUBufferUsage.STORAGE,
    }),
  );

  computeBindGroups = [
    device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particleBuffers[0] } },
        { binding: 2, resource: { buffer: particleBuffers[1] } },
        { binding: 3, resource: maskTexture.createView() },
      ],
    }),
    device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particleBuffers[1] } },
        { binding: 2, resource: { buffer: particleBuffers[0] } },
        { binding: 3, resource: maskTexture.createView() },
      ],
    }),
  ];

  renderBindGroups = particleBuffers.map((buffer) =>
    device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer } },
      ],
    }),
  );

  bufferIndex = 0;
  reset = true;
};

const colorToRgb = (color) => {
  let value = color;
  if (typeof value === "string") {
    value = Number.parseInt(value.replace("#", ""), 16);
  }
  return [
    ((value >> 16) & 0xff) / 0xff,
    ((value >> 8) & 0xff) / 0xff,
    (value & 0xff) / 0xff,
  ];
};

function recreateMaskTexture() {
  if (!device) {
    return;
  }

  maskTexture?.destroy();
  maskTexture = device.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source: texcanvas },
    { texture: maskTexture },
    [W, H],
  );

  if (computePipeline && renderPipeline && particleBuffers.length) {
    createParticleResources(bufferParticlesCount);
  } else {
    reset = true;
  }
}

const writeUniforms = (deltaTime, fadeOutT) => {
  const data = new ArrayBuffer(112);
  const floats = new Float32Array(data);
  const uints = new Uint32Array(data);
  const particleCount = Math.max(0, Math.ceil(GUI.particlesCount));
  const [red, green, blue] = colorToRgb(GUI.color);

  floats.set([W, H, GUI.radius, 0], 0);
  floats.set([time, deltaTime, GUI.seed, GUI.noiseScale], 4);
  floats.set([GUI.noiseSpeed, GUI.noiseMovement, GUI.forceMult, GUI.dampingMult], 8);
  floats.set([GUI.velocityMult, GUI.maxVelocity, GUI.longevity, fadeOutT], 12);
  floats.set([GUI.fadeOutXY[0], GUI.fadeOutXY[1], GUI.text ? 1 : 0, 0], 16);
  uints.set([particleCount, reset ? 1 : 0, frameIndex, 0], 20);
  floats.set([red, green, blue, 1], 24);
  device.queue.writeBuffer(uniformBuffer, 0, data);
};

const loadShader = async (path) => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Unable to load shader: ${path}`);
  }
  return response.text();
};

const init = async () => {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported by this browser");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter is available");
  }

  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  uniformBuffer = device.createBuffer({
    size: 112,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  recreateMaskTexture();

  const [computeCode, vertexCode, fragmentCode] = await Promise.all([
    loadShader("./compute.wgsl"),
    loadShader("./vertex.wgsl"),
    loadShader("./fragment.wgsl"),
  ]);
  const computeModule = device.createShaderModule({ code: computeCode });
  const vertexModule = device.createShaderModule({ code: vertexCode });
  const fragmentModule = device.createShaderModule({ code: fragmentCode });
  const reports = await Promise.all([
    computeModule.getCompilationInfo(),
    vertexModule.getCompilationInfo(),
    fragmentModule.getCompilationInfo(),
  ]);
  const errors = reports.flatMap((report) =>
    report.messages.filter((message) => message.type === "error"),
  );
  if (errors.length) {
    throw new Error(`Shader compilation failed: ${errors[0].message}`);
  }

  computePipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "simulate" },
  });

  renderPipeline = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module: vertexModule, entryPoint: "vertex_main" },
    fragment: {
      module: fragmentModule,
      entryPoint: "fragment_main",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });

  createParticleResources(GUI.particlesCount);
};

const loop = () => {
  if (!running) {
    return;
  }

  canvasParent.className = GUI.showTextTexture ? "text" : "notext";
  stats.begin();

  const now = window.performance.now();
  const dt = Math.min((now - lastDrawTime) / 1000, 1) * GUI.timeScale;
  lastDrawTime = now;
  time += dt;

  if (GUI.fadeOut) {
    const fadeOutDuration = 400;
    fadeOutTime += (dt * 1000) / fadeOutDuration;
  }
  const fadeOutT = Math.min(fadeOutTime, 1);

  if (bufferParticlesCount < GUI.particlesCount) {
    createParticleResources(GUI.particlesCount);
  }

  if (reset) {
    time = 0;
  }
  writeUniforms(dt, fadeOutT);

  const encoder = device.createCommandEncoder();
  let writeBufferIndex = 1 - bufferIndex;

  if (fadeOutT < 1 && GUI.particlesCount > 0) {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroups[bufferIndex]);
    computePass.dispatchWorkgroups(Math.ceil(GUI.particlesCount / 128));
    computePass.end();
  } else {
    writeBufferIndex = bufferIndex;
  }

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  if (fadeOutT < 1 && GUI.particlesCount > 0) {
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroups[writeBufferIndex]);
    renderPass.draw(6, Math.ceil(GUI.particlesCount));
  }
  renderPass.end();

  device.queue.submit([encoder.finish()]);
  bufferIndex = writeBufferIndex;
  reset = false;
  frameIndex += 1;
  stats.end();
  requestAnimationFrame(loop);
};

const die = () => {
  running = false;
  for (const buffer of particleBuffers) {
    buffer.destroy();
  }
  particleBuffers = [];
  uniformBuffer?.destroy();
  maskTexture?.destroy();
  device?.destroy();
  canvas.remove();
};

const gui = new dat.GUI({ hideable: false });
gui.domElement.remove();
document.querySelector("#tools").appendChild(gui.domElement);

stats.showPanel(0);
const perfLi = document.createElement("li");
perfLi.style.height = "50px";
stats.domElement.style.position = "static";
perfLi.appendChild(stats.domElement);
perfLi.classList.add("gui-stats");
gui.__ul.appendChild(perfLi);

gui.add(GUI, "text");
gui.add(GUI, "showTextTexture");
gui.add(GUI, "particlesCount", 0, 1_000_000);
gui.add(GUI, "radius", 0, 30);
gui.add(GUI, "timeScale", 0, 10);
gui.add(GUI, "noiseScale", 0.01, 30);
gui.add(GUI, "noiseSpeed", 0, 3);
gui.add(GUI, "noiseMovement", 0, 100);
gui.add(GUI, "longevity", 0, 3);
gui.add(GUI, "maxVelocity", 0, 100);
gui.add(GUI, "forceMult", 0, 15);
gui.add(GUI, "velocityMult", 0, 15);
gui.add(GUI, "dampingMult", 0.9, 0.9999);
gui.add(GUI, "seed", 0, 100);
gui.add(GUI, "reset");
gui.add(GUI, "destroy");
gui.add(GUI, "fadeOut");
gui.addColor(GUI, "color");

canvas.onclick = (event) => {
  GUI.fadeOut = true;
  GUI.fadeOutXY = [
    event.offsetX * window.devicePixelRatio,
    H - event.offsetY * window.devicePixelRatio,
  ];
};

init()
  .then(loop)
  .catch((error) => {
    console.error(error);
  });
