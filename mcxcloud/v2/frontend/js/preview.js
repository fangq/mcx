// @ts-check
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $ } from './util.js';
import { decodeJDataArray, colormapRGBA, COLORMAP_NAMES } from './util.js';
import { volface, qmeshcut, meshBBox } from './mesh.js';
import { state } from './state.js';

// ---- GLSL volume raycaster (ported from MCX Cloud v1 / three.js VolumeShader) ----
// Written as explicit GLSL ES 3.00 for a RawShaderMaterial: we declare the three.js
// built-ins (matrices/position) and our own fragment output ourselves, instead of relying
// on three's shader prefix. This compiles identically across three.js versions (the r160
// ShaderMaterial GLSL3 prefix does NOT alias gl_FragColor) and on every WebGL2 browser.
// sampler3D requires WebGL2 (universal since ~2021).
const VERT = /* glsl */ `
  precision highp float;
  uniform mat4 modelMatrix;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform mat4 viewMatrix;
  in vec3 position;
  out vec4 v_nearpos;
  out vec4 v_farpos;
  out vec3 v_position;
  void main() {
    mat4 viewtransformf = modelViewMatrix;
    mat4 viewtransformi = inverse(modelViewMatrix);
    vec4 position4 = vec4(position, 1.0);
    vec4 pos_in_cam = viewtransformf * position4;
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = viewtransformi * pos_in_cam;
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = viewtransformi * pos_in_cam;
    v_position = position;
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * position4;
  }`;

function fragment(mode) {
  return /* glsl */ `
  precision highp float;
  precision mediump sampler3D;
  uniform vec3 u_size;
  uniform float u_renderthreshold;
  uniform vec2 u_clim;
  uniform float u_empty;      // samples <= this are "empty" (0-fluence) -> transparent
  uniform float u_alpha;      // isosurface opacity / DVR density (0..1)
  uniform float u_shade;      // DVR gradient shading on/off (1/0)
  uniform float u_gamma;      // colormap contrast (gamma); >1 expands the lower values
  uniform sampler3D u_data;
  uniform sampler2D u_cmdata;
  uniform vec3 u_minslice;
  uniform vec3 u_maxslice;
  in vec3 v_position;
  in vec4 v_nearpos;
  in vec4 v_farpos;
  out vec4 fragColor;
  const int MAX_STEPS = 887;
  const int REFINEMENT_STEPS = 4;
  const float relative_step_size = 1.0;

  float sample1(vec3 texcoords){ return texture(u_data, texcoords.xyz).r; }
  // normalize into the clim window, then apply the contrast (gamma) curve. u_gamma>1 pushes
  // the mapped position down, so mid/low values spread across more of the colormap.
  float cmap_pos(float val){ return pow(clamp((val-u_clim.x)/(u_clim.y-u_clim.x),0.0,1.0), u_gamma); }
  vec4 apply_colormap(float val){ return texture(u_cmdata, vec2(cmap_pos(val),0.5)); }

  vec4 add_lighting(float val, vec3 loc, vec3 step, vec3 view_ray){
    vec3 V = normalize(view_ray);
    vec3 N; float val1, val2;
    val1=sample1(loc+vec3(-step[0],0.0,0.0)); val2=sample1(loc+vec3(step[0],0.0,0.0)); N[0]=val1-val2; val=max(max(val1,val2),val);
    val1=sample1(loc+vec3(0.0,-step[1],0.0)); val2=sample1(loc+vec3(0.0,step[1],0.0)); N[1]=val1-val2; val=max(max(val1,val2),val);
    val1=sample1(loc+vec3(0.0,0.0,-step[2])); val2=sample1(loc+vec3(0.0,0.0,step[2])); N[2]=val1-val2; val=max(max(val1,val2),val);
    N=normalize(N);
    float Nselect=float(dot(N,V)>0.0); N=(2.0*Nselect-1.0)*N;
    float lambert=clamp(dot(N,normalize(view_ray)),0.0,1.0);
    vec4 color=apply_colormap(val);
    vec4 final_color=color*(0.2+lambert); final_color.a=color.a; return final_color;
  }

  void cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray){
    float max_val=-1e30; vec3 loc=start_loc;
    for(int iter=0; iter<=MAX_STEPS; iter++){ if(iter>=nsteps) break; float v=sample1(loc); if(v>u_empty) max_val=max(v,max_val); loc+=step; }
    if(max_val<=u_empty) discard; // ray hit only empty (0) voxels -> transparent
    fragColor=apply_colormap(max_val); fragColor.a*=u_alpha;
  }
  void cast_iso(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray){
    // Show the surfaces of every label/level WITHIN the clim range [u_clim.x, u_clim.y],
    // compositing them front-to-back with opacity u_alpha (so at alpha<1 inner labels show
    // through outer ones; at alpha=1 only the outermost in-range surface is visible).
    vec3 dstep=1.5/u_size; vec3 loc=start_loc; float prev=-1e30; vec4 acc=vec4(0.0);
    for(int iter=0; iter<=MAX_STEPS; iter++){
      if(iter>=nsteps) break;
      float val=sample1(loc);
      if(val>u_clim.x && val<=u_clim.y && abs(val-prev)>0.5){ // crossed into a new in-range label
        vec4 c=add_lighting(val, loc, dstep, view_ray);
        float a=u_alpha*c.a;
        acc.rgb += (1.0-acc.a)*a*c.rgb;
        acc.a   += (1.0-acc.a)*a;
        if(acc.a>=0.98) break;
      }
      prev=val; loc+=step;
    }
    if(acc.a<=0.004) discard; // nothing in range -> transparent
    fragColor=vec4(acc.rgb/acc.a, acc.a);
  }
  void cast_dvr(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray){
    // MIDA (Maximum Intensity Difference Accumulation, Bruckner & Groller 2009): front-to-back
    // compositing that, on reaching a NEW maximum along the ray, attenuates the accumulated
    // color by beta=1-delta so the brighter interior shows through — MIP-like peak visibility
    // with DVR depth/opacity + optional gradient shading. Big intensity jumps -> MIP-like;
    // gentle regions -> DVR-like. The smoothstep floor keeps the dim, noisy outer halo clear.
    vec3 dstep=1.5/u_size; vec3 loc=start_loc; vec4 acc=vec4(0.0); float maxt=0.0;
    for(int iter=0; iter<=MAX_STEPS; iter++){
      if(iter>=nsteps) break;
      float val=sample1(loc);
      if(val>u_empty){
        float t=clamp((val-u_clim.x)/(u_clim.y-u_clim.x), 0.0, 1.0);
        vec4 col=texture(u_cmdata, vec2(pow(t,u_gamma),0.5)); // contrast-mapped color
        if(u_shade>0.5){ col.rgb=add_lighting(val, loc, dstep, view_ray).rgb; }
        float a=u_alpha*smoothstep(0.2, 1.0, t)*col.a; // opacity ramp with transparent low-end
        float beta=1.0;
        if(t>maxt){ beta=1.0-(t-maxt); maxt=t; }       // new max -> reveal it (toward MIP)
        acc.rgb=beta*acc.rgb + (1.0-beta*acc.a)*a*col.rgb;
        acc.a  =beta*acc.a   + (1.0-beta*acc.a)*a;
      }
      loc+=step;
    }
    if(acc.a<=0.004) discard;                  // ray hit only empty voxels -> transparent
    fragColor=vec4(acc.rgb/acc.a, acc.a);
  }

  void main(){
    vec3 farpos=v_farpos.xyz/v_farpos.w; vec3 nearpos=v_nearpos.xyz/v_nearpos.w;
    vec3 view_ray=normalize(nearpos.xyz-farpos.xyz);
    float distance=dot(nearpos-v_position, view_ray);
    vec3 cmp=(-vec3(0.5)-v_position)/view_ray;
    vec3 cmpu=(u_size)/view_ray+cmp; cmp=min(cmp,cmpu);
    distance=max(distance, max(cmp.x, max(cmp.y, cmp.z)));
    vec3 front=v_position+view_ray*distance;
    int nsteps=max(1, int(-distance/relative_step_size+0.5));
    vec3 step=((v_position-front)/u_size)/float(nsteps);
    vec3 start_loc=front/u_size;
    vec3 lo=mix(vec3(0.0),(u_maxslice-start_loc)/step, greaterThan(start_loc,u_maxslice));
    float skips=max(lo.x,max(lo.y,lo.z));
    lo=mix(vec3(0.0),(u_minslice-start_loc)/step, lessThan(start_loc,u_minslice));
    skips=max(skips, max(lo.x,max(lo.y,lo.z)));
    start_loc+=skips*step; nsteps-=int(skips+0.5);
    // clamp the FAR crop bound too: stop the ray at the exit face of the [minslice,maxslice]
    // box (without this only the near side is cropped, so the crop appears to flip when the
    // near/far faces swap during rotation).
    vec3 hi=mix(vec3(float(nsteps)), ceil((u_maxslice-start_loc)/step), greaterThan((start_loc+float(nsteps)*step), u_maxslice));
    hi=min(hi, mix(vec3(float(nsteps)), ceil((u_minslice-start_loc)/step), lessThan((start_loc+float(nsteps)*step), u_minslice)));
    nsteps=int(min(hi.x, min(hi.y, hi.z))+0.5);
    ${mode === 'mip' ? 'cast_mip' : mode === 'dvr' ? 'cast_dvr' : 'cast_iso'}(start_loc, step, nsteps, view_ray);
  }`;
}

/** current render mode from the radio buttons: 'mip' | 'iso' | 'dvr' */
function currentRenderMode() {
  if (/** @type {HTMLInputElement} */ ($('#dvr-radio'))?.checked) return 'dvr';
  if (/** @type {HTMLInputElement} */ ($('#mip-radio')).checked) return 'mip';
  return 'iso';
}

const baseUniforms = () => ({
  u_size: { value: new THREE.Vector3(1, 1, 1) },
  u_renderthreshold: { value: 0.2 },
  u_clim: { value: new THREE.Vector2(0, 1) },
  u_empty: { value: -1e30 },
  u_alpha: { value: 1 },
  u_shade: { value: 0 },
  u_gamma: { value: 1 },
  u_data: { value: null },
  u_cmdata: { value: null },
  u_minslice: { value: new THREE.Vector3(0, 0, 0) },
  u_maxslice: { value: new THREE.Vector3(1, 1, 1) },
});

// ---- scene state ---------------------------------------------------------------
let scene, camera, renderer, controls, cmTexture;
let boundingbox, bbxsize = [1, 1, 1], lastVolume = null, lastDim = [];
// 4D+ volume support: keep the full decoded array and render one 3D frame at a time
let fullVol = null, volIsLog = false, curFrame = 0, numFrames = 1;
// tetrahedral mesh (MMC) support: decoded mesh + its surface and cross-section meshes
let meshState = null;
let dirty = true;
let fpsDiv = null, frames = 0, fpsLast = 0;
let lastThumbnail = null;
let currentCmap = 'viridis';

/** rebuild the colormap texture and apply it to the current volume */
function setColormap(name) {
  currentCmap = name;
  const cm = colormapRGBA(name);
  cmTexture = new THREE.DataTexture(cm, cm.length / 4, 1, THREE.RGBAFormat);
  cmTexture.minFilter = cmTexture.magFilter = THREE.LinearFilter;
  cmTexture.needsUpdate = true;
  if (lastVolume) { lastVolume.material.uniforms.u_cmdata.value = cmTexture; dirty = true; }
  if (meshState && meshState.outVol) setMeshFrame(curFrame); // recolor mesh-valued output
}
const materialcolor = [];
(function seedColors() {
  let s = 1648335518;
  for (let i = 0; i < 256; i++) {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    s = (t ^ (t >>> 14)) >>> 0;
    materialcolor.push(s & 0xffffff);
  }
})();

export function initPreview() {
  const host = $('#canvas');
  const w = host.clientWidth || 640, h = host.clientHeight || 460;
  scene = new THREE.Scene();
  boundingbox = scene;
  camera = new THREE.OrthographicCamera(w / -2, w / 2, h / 2, h / -2, 1, 5000);
  camera.up.set(0, 0, 1);
  camera.position.set(200, 150, 150);
  camera.lookAt(0, 0, 0);
  renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true, antialias: true });
  renderer.localClippingEnabled = true; // mesh cross-sections clip per-material
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  host.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.minZoom = 0.02; controls.maxZoom = 200; // allow fitting very large/small volumes
  controls.addEventListener('change', () => { dirty = true; });

  setColormap(currentCmap);

  window.addEventListener('resize', onResize);
  fpsDiv = $('#fps'); fpsLast = performance.now();
  wireControls();
  loop();
}

function onResize() {
  const host = $('#canvas');
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  camera.left = w / -2; camera.right = w / 2; camera.top = h / 2; camera.bottom = h / -2;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  dirty = true;
}

function loop() {
  requestAnimationFrame(loop);
  if (dirty) { renderer.render(scene, camera); dirty = false; }
  controls.update();
  // render-rate gadget: frames per second of the animation loop (drops under load)
  frames++;
  const now = performance.now();
  if (now - fpsLast >= 1000) {
    if (fpsDiv) fpsDiv.textContent = Math.round((frames * 1000) / (now - fpsLast)) + ' fps';
    frames = 0; fpsLast = now;
  }
}

/**
 * Capture the current canvas as a compact PNG data URL (used as the library thumbnail).
 * @param {number} [tw] output width  @param {number} [th] output height
 * @returns {string|null}
 */
export function captureThumbnail(tw = 500, th = 400) {
  if (!renderer || !camera) return null;
  const SS = 2;                     // supersample factor: render 2× then downscale for anti-aliasing
  const rtW = tw * SS, rtH = th * SS;

  // Thumbnail camera: same position / orientation / zoom as the live view, but with the
  // projection fixed to a tw:th aspect (matching the old centered crop). Rendering to a
  // FIXED-size offscreen target — instead of downsampling the variable live canvas — makes a
  // given screen-space line width map to the same thumbnail thickness on every machine.
  const tAspect = tw / th;
  const hx = (camera.right - camera.left) / (2 * camera.zoom);  // world half-extents on screen
  const hy = (camera.top - camera.bottom) / (2 * camera.zoom);
  let thx, thy;
  if (hx / hy > tAspect) { thy = hy; thx = hy * tAspect; }      // screen wider: keep full height
  else { thx = hx; thy = hx / tAspect; }                        // screen taller: keep full width
  const tcam = camera.clone();
  tcam.left = -thx; tcam.right = thx; tcam.top = thy; tcam.bottom = -thy;
  tcam.zoom = 1;
  tcam.updateProjectionMatrix();

  // render into an offscreen target and read the pixels back
  const rt = new THREE.WebGLRenderTarget(rtW, rtH);
  const prevTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(rt);
  renderer.render(scene, tcam);
  const pixels = new Uint8Array(rtW * rtH * 4);
  renderer.readRenderTargetPixels(rt, 0, 0, rtW, rtH, pixels);
  renderer.setRenderTarget(prevTarget);
  rt.dispose();
  dirty = true;                     // repaint the live canvas on the next frame

  // raw buffer (WebGL rows are bottom-up) -> flip + downscale into the fixed tw×th output
  const full = document.createElement('canvas'); full.width = rtW; full.height = rtH;
  const fctx = full.getContext('2d');
  if (fctx) fctx.putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer), rtW, rtH), 0, 0);
  const c = document.createElement('canvas'); c.width = tw; c.height = th;
  const ctx = c.getContext('2d');
  if (ctx) {
    const cc = new THREE.Color(); renderer.getClearColor(cc);
    ctx.fillStyle = '#' + cc.getHexString(); ctx.fillRect(0, 0, tw, th); // opaque bg = scene clear color
    ctx.imageSmoothingQuality = 'high';
    ctx.translate(0, th); ctx.scale(1, -1);
    ctx.drawImage(full, 0, 0, tw, th);
  }
  lastThumbnail = c.toDataURL('image/png');
  // reflect into every thumbnail view (Preview panel + Share tab)
  document.querySelectorAll('.thumb-view').forEach((img) => {
    const im = /** @type {HTMLImageElement} */ (img);
    im.src = /** @type {string} */ (lastThumbnail); im.hidden = false;
  });
  return lastThumbnail;
}

/** @returns {string|null} last captured thumbnail (captures one on demand) */
export function getThumbnail() {
  return lastThumbnail || captureThumbnail();
}

// ---- geometry helpers ----------------------------------------------------------
function createbox(size, orig) {
  const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
  g.translate(size[0] * 0.5 + orig[0], size[1] * 0.5 + orig[1], size[2] * 0.5 + orig[2]);
  return new THREE.Mesh(g, new THREE.MeshNormalMaterial({ transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }));
}
function createlayer(s, dim) {
  const size = bbxsize.slice(); size[dim] = Math.abs(s[0] - s[1]);
  const orig = [0, 0, 0]; orig[dim] = Math.min(s[0], s[1]);
  return createbox(size, orig);
}

function resetscene(s, orig = [0, 0, 0]) {
  scene.clear();
  scene.add(new THREE.AmbientLight(0xffffff));
  const cx = orig[0] + s[0] / 2, cy = orig[1] + s[1] / 2, cz = orig[2] + s[2] / 2;
  const diag = Math.hypot(s[0], s[1], s[2]) || 1; // bounding-sphere diameter
  camera.up.set(0, 0, 1);
  camera.position.set(cx + s[0] * 1.2, cy + s[1] * 0.8, cz + s[2]);
  camera.lookAt(cx, cy, cz);
  controls.target.set(cx, cy, cz);
  // Fit the bounding sphere to the ORTHO FRUSTUM (in CSS units, so it's independent of the
  // device pixel ratio and adapts to any volume size — a 496-long digimouse and a 60³ cube
  // both fill ~90% of the view).
  const fw = camera.right - camera.left, fh = camera.top - camera.bottom;
  camera.zoom = 0.9 * Math.min(fw, fh) / diag;
  camera.updateProjectionMatrix();
  controls.update();
}

function drawshape(shape) {
  const keys = Object.keys(shape);
  const dir = { XLayers: 0, YLayers: 1, ZLayers: 2, XSlabs: 0, YSlabs: 1, ZSlabs: 2 };
  let g, m, obj;
  switch (keys[0]) {
    case 'Grid': {
      resetscene(shape.Grid.Size);
      const box = createbox(shape.Grid.Size, [0, 0, 0]);
      boundingbox = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineDashedMaterial({ color: 0xffff00, dashSize: 3, gapSize: 1 }));
      boundingbox.computeLineDistances();
      scene.add(boundingbox);
      bbxsize = shape.Grid.Size;
      controls.target.set(bbxsize[0] * 0.5, bbxsize[1] * 0.5, bbxsize[2] * 0.5);
      break;
    }
    case 'Box': boundingbox.add(createbox(shape.Box.Size, shape.Box.O)); break;
    case 'Subgrid': boundingbox.add(createbox(shape.Subgrid.Size, shape.Subgrid.O)); break;
    case 'XLayers': case 'YLayers': case 'ZLayers':
      (shape[keys[0]] || []).forEach((l) => boundingbox.add(createlayer(l, dir[keys[0]])));
      break;
    case 'XSlabs': case 'YSlabs': case 'ZSlabs': {
      const b = shape[keys[0]] && shape[keys[0]].Bound;
      if (b) (Array.isArray(b[0]) ? b : [b]).forEach((s) => boundingbox.add(createlayer(s, dir[keys[0]])));
      break;
    }
    case 'Sphere':
      g = new THREE.SphereGeometry(shape.Sphere.R, 32, 32);
      m = new THREE.MeshBasicMaterial({ color: materialcolor[shape.Sphere.Tag] || 0xff0000, wireframe: true, transparent: true });
      obj = new THREE.Mesh(g, m); obj.position.set(...shape.Sphere.O); boundingbox.add(obj);
      break;
    case 'Cylinder': {
      const c0 = new THREE.Vector3(...shape.Cylinder.C0), c1 = new THREE.Vector3(...shape.Cylinder.C1);
      const height = c0.distanceTo(c1);
      g = new THREE.CylinderGeometry(shape.Cylinder.R, shape.Cylinder.R, height, 32);
      // CylinderGeometry is centered at the origin along +Y: orient +Y to the C0->C1 axis,
      // then move to the segment MIDPOINT. (v1 translated by C0+C1 — the sum — which pushed
      // the cylinder outside the domain.)
      const axis = c1.clone().sub(c0).normalize();
      g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis));
      const mid = c0.clone().add(c1).multiplyScalar(0.5);
      g.translate(mid.x, mid.y, mid.z);
      m = new THREE.MeshBasicMaterial({ color: materialcolor[shape.Cylinder.Tag] || 0x00ffff, wireframe: true });
      boundingbox.add(new THREE.Mesh(g, m));
      break;
    }
  }
}

// a filled disk of radius R at `orig`, oriented to face `norm` (the source direction)
function createdisk(R, orig, norm) {
  const g = new THREE.CircleGeometry(R, 48);
  const n = new THREE.Vector3(...norm).normalize();
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n));
  g.translate(orig[0], orig[1], orig[2]);
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.DoubleSide, depthWrite: false }));
}
// a filled polygon from an explicit list of triangle vertices
function createpoly(points) {
  const g = new THREE.BufferGeometry().setFromPoints(points);
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.DoubleSide, depthWrite: false }));
}

// Multi-src inputs store Pos/Dir/Param1/Param2 as arrays of vectors (one row per
// source); expand them and draw each source with the shared Type/Frequency/etc.
function drawsrc(src) {
  if (Array.isArray(src.Pos) && Array.isArray(src.Pos[0])) {
    const pick = (v, i) => (Array.isArray(v) && Array.isArray(v[0]) ? v[i] : v);
    src.Pos.forEach((_, i) => drawonesrc({
      ...src,
      Pos: pick(src.Pos, i),
      Dir: pick(src.Dir, i),
      Param1: pick(src.Param1, i),
      Param2: pick(src.Param2, i),
    }));
    return;
  }
  drawonesrc(src);
}

// Render one source: always an arrow for direction, plus a shape per Source.Type
// (disk/gaussian -> disk; planar/pattern/fourier[x] -> quad from Param1/Param2; line/slit
// -> a segment). Mirrors v1's drawsrc so all MCX source types preview correctly.
function drawonesrc(src) {
  const dir = new THREE.Vector3(...src.Dir.slice(0, 3).map(Number)).normalize();
  const orig = new THREE.Vector3(...src.Pos);
  const len = Math.max(...bbxsize) * 0.2;
  boundingbox.add(new THREE.ArrowHelper(dir, orig, len, 0xffffff, 0.3 * len, 0.15 * len));

  const P1 = src.Param1 || [], P2 = src.Param2 || [];
  const v1 = new THREE.Vector3(P1[0] || 0, P1[1] || 0, P1[2] || 0);
  const v2 = new THREE.Vector3(P2[0] || 0, P2[1] || 0, P2[2] || 0);
  // two triangles spanning the parallelogram orig + a, orig + b
  const quad = (a, b) => createpoly([
    orig, orig.clone().add(a), orig.clone().add(a).add(b),
    orig.clone().add(a).add(b), orig.clone().add(b), orig,
  ]);

  switch (src.Type) {
    case 'disk': case 'gaussian': case 'zgaussian': case 'ring':
      if (P1[0]) boundingbox.add(createdisk(P1[0], src.Pos, src.Dir));
      break;
    case 'planar': case 'pattern': case 'pattern3d': case 'fourier':
      boundingbox.add(quad(v1, v2));
      break;
    case 'fourierx': case 'fourierx2d': {
      // Param2 is derived: perpendicular to dir and Param1, length Param1[3]
      const w = new THREE.Vector3().crossVectors(dir, v1).normalize().multiplyScalar(P1[3] || 0);
      boundingbox.add(quad(v1, w));
      break;
    }
    case 'line': case 'slit':
      boundingbox.add(new THREE.ArrowHelper(v1.clone().normalize(), orig, v1.length() || len, 0xffffff, 0, 0));
      break;
  }
}
function drawdet(det) {
  const g = new THREE.SphereGeometry(det.R, 24, 24);
  g.translate(...det.Pos);
  boundingbox.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true })));
}

// ---- tetrahedral mesh (MMC) rendering --------------------------------------------
// Follows iso2mesh's approach: render only the exterior surface (volface) instead of
// all tets, and reveal the interior regions with qmeshcut planar cross-sections driven
// by the existing X/Y/Z crop sliders (surface clipped + cut patches at the crop planes).

/** decode MeshNode/MeshElem in either plain array-of-rows or JData-annotated form
 *  @param {any} spec @param {number} cols */
function meshArray(spec, cols) {
  if (Array.isArray(spec)) {
    const rows = spec.length;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = spec[i][j];
    return { data: out, rows, cols };
  }
  const v = decodeJDataArray(spec);
  const c = (v.size && v.size[1]) || cols;
  return { data: v.data, rows: (v.size && v.size[0]) || Math.floor(v.data.length / c), cols: c };
}

/** region tag -> [r,g,b] floats from the shared label palette
 *  @param {number} tag */
function tagRGB(tag) {
  const h = materialcolor[Math.abs(tag | 0) % materialcolor.length] || 0xff0000;
  return [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];
}

/** map an (already log-scaled) output value to [r,g,b] via the active colormap LUT
 *  @param {number} v */
function lutRGB(v) {
  const { vmin, vmax, lut } = meshState;
  let x = (v - vmin) / (vmax - vmin || 1);
  x = x < 0 ? 0 : x > 1 ? 1 : x;
  const k = ((x * 255) | 0) * 4;
  return [lut[k] / 255, lut[k + 1] / 255, lut[k + 2] / 255];
}

/** triangle soup + per-vertex colors -> renderable geometry
 *  @param {number[] | Float32Array} positions @param {Float32Array} colors */
function soupGeometry(positions, colors) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return g;
}

/** per-vertex surface colors for the current mode: region tags (input preview),
 *  per-element values (BasisOrder 0), or per-node values (BasisOrder 1) */
function surfColors() {
  const { faces, owner, mode, elem, stride, frameVals } = meshState;
  const col = new Float32Array(faces.length * 3);
  for (let f = 0; f < owner.length; f++) {
    for (let v = 0; v < 3; v++) {
      const rgb = mode === 'node' ? lutRGB(frameVals[faces[f * 3 + v] - 1])
        : mode === 'elem' ? lutRGB(frameVals[owner[f]])
          : tagRGB(elem[owner[f] * stride + stride - 1]);
      const k = (f * 3 + v) * 3;
      col[k] = rgb[0]; col[k + 1] = rgb[1]; col[k + 2] = rgb[2];
    }
  }
  return col;
}

/** per-vertex colors for one qmeshcut result under the current mode
 *  @param {{ positions: number[], elemid: number[], values: number[] | null }} cut */
function cutColors(cut) {
  const { mode, elem, stride, frameVals } = meshState;
  const col = new Float32Array(cut.positions.length);
  for (let t = 0; t < cut.elemid.length; t++) {
    for (let v = 0; v < 3; v++) {
      const rgb = mode === 'node' && cut.values ? lutRGB(cut.values[t * 3 + v])
        : mode === 'elem' ? lutRGB(frameVals[cut.elemid[t]])
          : tagRGB(elem[cut.elemid[t] * stride + stride - 1]);
      const k = (t * 3 + v) * 3;
      col[k] = rgb[0]; col[k + 1] = rgb[1]; col[k + 2] = rgb[2];
    }
  }
  return col;
}

/** load frame f of a mesh-valued (node/elem basis) output: log-scale the slice,
 *  refresh the colormap LUT, then recolor the surface and rebuild the cuts
 *  @param {number} f */
function setMeshFrame(f) {
  const { outVol, frameLen } = meshState;
  const raw = outVol.data.subarray(f * frameLen, (f + 1) * frameLen);
  const vals = new Float32Array(frameLen);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < frameLen; i++) {
    const v = raw[i];
    if (v > 0) { const l = Math.log(v); vals[i] = l; if (l < vmin) vmin = l; if (l > vmax) vmax = l; }
    else vals[i] = NaN;
  }
  if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
  for (let i = 0; i < frameLen; i++) if (Number.isNaN(vals[i])) vals[i] = vmin; // 0-fluence -> colormap floor
  meshState.frameVals = vals;
  meshState.vmin = vmin; meshState.vmax = vmax;
  meshState.lut = colormapRGBA(currentCmap);
  meshState.surf.geometry.setAttribute('color', new THREE.BufferAttribute(surfColors(), 3));
  updateMeshCross();
  dirty = true;
}

/**
 * Shared mesh renderer: dashed bbox + exterior surface (volface) + slider-driven cuts.
 * mode 'tag' colors by region label (input preview); 'node'/'elem' color by an
 * mmc mesh-valued output (BasisOrder 1/0) through the active colormap.
 * @param {{data: Float32Array, rows: number, cols: number}} nd decoded MeshNode
 * @param {{data: Float32Array, rows: number, cols: number}} el decoded MeshElem
 * @param {'tag'|'node'|'elem'} mode
 * @param {{data: Float32Array}|null} outVol decoded output array (value modes)
 * @param {number} frameLen values per frame (nn or ne)  @param {number} nframes
 */
function drawmeshCore(nd, el, mode, outVol, frameLen, nframes) {
  const bbox = meshBBox(nd.data, nd.rows);
  const ext = [0, 1, 2].map((a) => Math.max(bbox.max[a] - bbox.min[a], 1e-6));

  // dashed domain box spanning the mesh bounding box (meshes need not start at 0)
  resetscene(ext, bbox.min);
  const box = createbox(ext, bbox.min);
  boundingbox = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineDashedMaterial({ color: 0xffff00, dashSize: 3, gapSize: 1 }));
  boundingbox.computeLineDistances();
  scene.add(boundingbox);
  bbxsize = ext;
  lastDim = ext; // slab-thickness boxes act in mesh (world) units

  meshState = {
    node: nd.data, elem: el.data, ne: el.rows, stride: el.cols, bbox, ext,
    faces: null, owner: null, surf: null, wire: null, cuts: null,
    mode, outVol, frameLen, frameVals: null, vmin: 0, vmax: 1, lut: null,
  };

  // exterior surface (iso2mesh volface), one flat triangle per boundary face
  const { faces, owner } = volface(el.data, el.rows, el.cols);
  meshState.faces = faces; meshState.owner = owner;
  const pos = new Float32Array(faces.length * 3);
  for (let i = 0; i < faces.length; i++) {
    const n = (faces[i] - 1) * 3;
    pos[i * 3] = nd.data[n]; pos[i * 3 + 1] = nd.data[n + 1]; pos[i * 3 + 2] = nd.data[n + 2];
  }
  meshState.surf = new THREE.Mesh(soupGeometry(pos, new Float32Array(pos.length)), new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: mode === 'tag' ? 0.35 : 0.75,
    side: THREE.DoubleSide, depthWrite: false,
  }));
  boundingbox.add(meshState.surf);
  // triangulation edges (iso2mesh plotmesh look): a wireframe sibling sharing the same
  // geometry — without it the coarse boundary triangles merge into flat unreadable fills.
  // Full-strength region colors for the input preview; subtle over fluence maps.
  meshState.wire = new THREE.Mesh(meshState.surf.geometry, new THREE.MeshBasicMaterial({
    wireframe: true, vertexColors: true, transparent: true,
    opacity: mode === 'tag' ? 0.9 : 0.3, depthWrite: false,
  }));
  boundingbox.add(meshState.wire);
  meshState.cuts = new THREE.Group();
  boundingbox.add(meshState.cuts);

  numFrames = nframes; curFrame = 0;
  if (mode === 'tag') {
    meshState.surf.geometry.setAttribute('color', new THREE.BufferAttribute(surfColors(), 3));
    updateMeshCross();
  } else {
    setMeshFrame(0); // computes colors + cuts for the first gate/source frame
  }
}

/** input-domain preview: mesh colored by region tags
 *  @param {any} shapes the Shapes object carrying MeshNode/MeshElem */
function drawmesh(shapes) {
  drawmeshCore(meshArray(shapes.MeshNode, 3), meshArray(shapes.MeshElem, 5), 'tag', null, 0, 1);
}

/**
 * mmc mesh-valued output (RayTracer != 'g'): values live on nodes (BasisOrder 1) or
 * elements (BasisOrder 0). The basis is detected from the decoded DATA LENGTH against
 * the input mesh — the jnii header is not trusted because mmc writes Dim[0]=nodenum
 * even for element-basis output (mmc_utils.c mcx_savedata).
 * @param {any} shapes input Shapes with the mesh @param {{data: Float32Array}} vol
 * @param {number[]|undefined} dim NIFTIHeader.Dim (frames = product of dims 1+)
 * @returns {boolean} true when rendered as a mesh-valued output
 */
function drawmeshOutput(shapes, vol, dim) {
  const nd = meshArray(shapes.MeshNode, 3);
  const el = meshArray(shapes.MeshElem, 5);
  const nfr = (Array.isArray(dim) && dim.length > 1)
    ? dim.slice(1).reduce((a, b) => a * (b || 1), 1) : 1;
  const frameLen = vol.data.length / nfr;
  const mode = frameLen === nd.rows ? 'node' : frameLen === el.rows ? 'elem' : null;
  if (!mode) return false; // not a per-node/per-elem array for this mesh
  drawmeshCore(nd, el, mode, vol, frameLen, nfr);
  return true;
}

/** re-apply the X/Y/Z crop sliders to a mesh: clip the exterior surface and rebuild
 *  qmeshcut cross-section patches at each active crop plane */
function updateMeshCross() {
  if (!meshState || !meshState.surf) return;
  const g = (id) => parseFloat(/** @type {HTMLInputElement} */ ($(id)).value);
  const { bbox, ext } = meshState;
  const AXIS = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  /** @type {THREE.Plane[][]} */
  const perAxis = [[], [], []];
  /** @type {[number, number][]} */
  const cutsAt = [];
  ['x', 'y', 'z'].forEach((ax, a) => {
    const lo = g('#cross-' + ax + '-low'), hi = g('#cross-' + ax + '-hi');
    const loW = bbox.min[a] + lo * ext[a], hiW = bbox.min[a] + hi * ext[a];
    // THREE clips fragments with negative signed distance (n·p + c < 0)
    if (lo > 0) { perAxis[a].push(new THREE.Plane(AXIS[a].clone(), -loW)); cutsAt.push([a, loW]); }
    if (hi < 1) { perAxis[a].push(new THREE.Plane(AXIS[a].clone().negate(), hiW)); cutsAt.push([a, hiW]); }
  });
  const all = perAxis.flat();
  meshState.surf.material.clippingPlanes = all.length ? all : null;
  if (meshState.wire) meshState.wire.material.clippingPlanes = all.length ? all : null;

  for (const m of meshState.cuts.children) { m.geometry.dispose(); m.material.dispose(); }
  meshState.cuts.clear();
  for (const [a, w] of cutsAt) {
    const cut = qmeshcut(meshState.node, meshState.elem, meshState.ne, meshState.stride, /** @type {0|1|2} */ (a), w,
      meshState.mode === 'node' ? meshState.frameVals : undefined);
    if (!cut.elemid.length) continue;
    const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    // crop the patch by the OTHER axes' planes only (its own plane would z-fight it away)
    const other = [0, 1, 2].filter((x) => x !== a).flatMap((x) => perAxis[x]);
    if (other.length) mat.clippingPlanes = other;
    const gm = soupGeometry(cut.positions, cutColors(cut));
    meshState.cuts.add(new THREE.Mesh(gm, mat));
    // edge overlay on the cut too, so the sliced tets read as a mesh (qmeshcut style)
    const wmat = new THREE.MeshBasicMaterial({
      wireframe: true, color: 0x000000, transparent: true,
      opacity: meshState.mode === 'tag' ? 0.35 : 0.15, depthWrite: false,
    });
    if (other.length) wmat.clippingPlanes = other;
    meshState.cuts.add(new THREE.Mesh(gm, wmat));
  }
  dirty = true;
}

/**
 * Build the volume raycasting mesh from a decoded array.
 * @param {{data: Float32Array, size: number[]}} vol
 */
// MCX/JData volumes come in C-order [nx,ny,nz] (z fastest); a Data3DTexture wants x
// fastest (i + j*nx + k*nx*ny). Reorder so texel(x,y,z)=vol(x,y,z). This is the axis
// transpose v1 did via numjs (volume.transpose().flatten()); without it the volume renders
// rotated 90°.
function volumeToTexture(data, nx, ny, nz) {
  const out = new Float32Array(data.length);
  let s = 0;
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++)
        out[x + y * nx + z * nx * ny] = data[s++];
  return out;
}

function drawvolume(vol, isLog = false) {
  const dim = vol.size;
  lastDim = dim;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < vol.data.length; i++) { const v = vol.data[i]; if (v < min) min = v; if (v > max) max = v; }
  if (!isFinite(min)) { min = 0; max = 1; }
  // For log fluence, use the REAL positive range from logscale (0-voxels are excluded and
  // rendered transparent), so the color scale isn't biased by the empty background.
  const cLow = (isLog && isFinite(vol.vmin)) ? vol.vmin : min;
  const cHigh = (isLog && isFinite(vol.vmax)) ? vol.vmax : max;

  // Align the texture with the MCX grid coords (so it matches the source/detector, which
  // render correctly). MCX stores its grid x-fastest = exactly what Data3DTexture wants, so
  // row-major "c" arrays (MCX JNIfTI output) pass through unchanged; arrays with no
  // _ArrayOrder_ tag (e.g. input Shapes volumes) need the transpose to x-fastest.
  const rowMajor = String(vol.order || '').startsWith('c');
  const texData = rowMajor ? vol.data : volumeToTexture(vol.data, dim[0], dim[1], dim[2]);
  const tex = new THREE.Data3DTexture(texData, dim[0], dim[1], dim[2]);
  tex.format = THREE.RedFormat; tex.type = THREE.FloatType;
  // linear filtering of FLOAT 3D textures needs OES_texture_float_linear (absent on some
  // mobile/older GPUs) — fall back to nearest so the volume still renders everywhere.
  const floatLinear = renderer && renderer.extensions.has('OES_texture_float_linear');
  tex.minFilter = tex.magFilter = floatLinear ? THREE.LinearFilter : THREE.NearestFilter;
  tex.unpackAlignment = 1; tex.needsUpdate = true;

  const mode = currentRenderMode();
  const uniforms = baseUniforms();
  uniforms.u_data.value = tex;
  uniforms.u_cmdata.value = cmTexture;
  uniforms.u_size.value.set(dim[0], dim[1], dim[2]);
  uniforms.u_clim.value.set(cLow, cHigh);
  // render empty voxels transparent: for log fluence that's the 0-fluence sentinel; for a
  // segmentation it's label 0 (background) — anything <= 0.5 (i.e. label 0) is dropped.
  uniforms.u_empty.value = (isLog && isFinite(vol.empty)) ? vol.empty + 0.5 : 0.5;
  // Iso threshold: for a segmentation (integer labels, background 0) use 0.5 so the LOWEST
  // labels (1,2,…) are shown; for log fluence use a threshold within the color window.
  uniforms.u_renderthreshold.value = isLog ? cLow + 0.2 * (cHigh - cLow) : 0.5;
  const alphaEl = /** @type {HTMLInputElement} */ ($('#iso-alpha'));
  uniforms.u_alpha.value = alphaEl ? (parseFloat(alphaEl.value) || 1) : 1;
  const shadeEl = /** @type {HTMLInputElement} */ ($('#dvr-shade'));
  uniforms.u_shade.value = shadeEl && shadeEl.checked ? 1 : 0;
  // Contrast (gamma): log fluence spans the upper band of the window and reads flat, so boost
  // it by default; segmentation/linear data stays linear (1). The slider can fine-tune.
  const defGamma = isLog ? 2.0 : 1.0;
  const contrastEl = /** @type {HTMLInputElement} */ ($('#contrast'));
  if (contrastEl) contrastEl.value = String(defGamma);
  uniforms.u_gamma.value = defGamma;
  readCrossInto(uniforms);

  const material = new THREE.RawShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: fragment(mode), side: THREE.BackSide,
    glslVersion: THREE.GLSL3, transparent: true, depthWrite: false,
  });
  const geometry = new THREE.BoxGeometry(dim[0], dim[1], dim[2]);
  geometry.translate(dim[0] * 0.5, dim[1] * 0.5, dim[2] * 0.5);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;

  // sync clim sliders to the active color range [cLow, cHigh]
  for (const [id, val] of [['#clim-low', cLow], ['#clim-hi', cHigh]]) {
    const el = /** @type {HTMLInputElement} */ ($(id));
    el.disabled = false; el.min = String(cLow); el.max = String(cHigh); el.value = String(val);
  }
  return mesh;
}

/**
 * Extract one 3D frame from an N-D volume, flattening every dimension above 3 (time
 * gates, photon-sharing patterns, multi-src/RF/replay-detector blocks) into a single
 * frame counter. MCX outputs (_ArrayOrder_ 'c') store [nx,ny,nz,nt,ns,nr] with memory
 * layout (fastest->slowest) [pattern(ns), x, y, z, gate(nt), rep(nr)] — the pattern dim
 * is interleaved BELOW x (mcx_core.cu: field[(idx1d+tshift*dimlen.z)*srcnum+i]) — so a
 * frame is a strided gather when ns>1. Arrays without the 'c' tag treat the extra
 * dimensions as slowest (contiguous frame blocks).
 * @param {{data: Float32Array, size: number[], order: string}} vol
 * @param {number} f frame index in [0, numFrames)
 */
function extractFrame(vol, f) {
  const [nx = 1, ny = 1, nz = 1, nt = 1, ns = 1] = vol.size;
  const flen = nx * ny * nz;
  const size3 = [nx, ny, nz];
  const nfr = vol.size.slice(3).reduce((a, b) => a * (b || 1), 1);
  if (nfr <= 1 || vol.data.length < flen * nfr) return { data: vol.data, size: size3, order: vol.order };
  f = Math.min(Math.max(f, 0), nfr - 1);
  if (String(vol.order || '').startsWith('c') && ns > 1) {
    // frame f -> (gate t, pattern s, rep r), stepping through time gates first
    const t = f % nt, s = Math.floor(f / nt) % ns, r = Math.floor(f / (nt * ns));
    const out = new Float32Array(flen);
    const base = (r * nt + t) * flen;
    for (let v = 0; v < flen; v++) out[v] = vol.data[(base + v) * ns + s];
    return { data: out, size: size3, order: vol.order };
  }
  // ns==1 'c' frames and untagged arrays are contiguous [flen] blocks in frame order
  return { data: vol.data.subarray(f * flen, (f + 1) * flen), size: size3, order: vol.order };
}

/** show/hide + sync the 4D frame spinner with the current volume */
function syncFrameUI() {
  const wrap = $('#frame-ctl');
  if (!wrap) return;
  wrap.hidden = numFrames <= 1;
  const num = /** @type {HTMLInputElement} */ ($('#frame-num'));
  num.max = String(numFrames);
  num.value = String(curFrame + 1);
  $('#frame-total').textContent = String(numFrames);
}

/** build the renderable 3D mesh for frame f of the current fullVol */
function frameMesh(f) {
  const frame = extractFrame(fullVol, f);
  return drawvolume(volIsLog ? logscale(frame) : frame, volIsLog);
}

/** apply log10 scaling (for fluence output) in place -> new Float32Array */
function logscale(vol) {
  // Take log only of POSITIVE fluence; 0 (and negatives) are marked empty (a sentinel just
  // below the real min) so the shader renders them transparent — and the color range spans
  // only the real fluence (no -36.8 background floor biasing the scale).
  const n = vol.data.length;
  const out = new Float32Array(n);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = vol.data[i];
    if (v > 0) { const l = Math.log(v); out[i] = l; if (l < vmin) vmin = l; if (l > vmax) vmax = l; }
    else out[i] = NaN; // mark empty
  }
  if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
  const empty = vmin - 1; // sentinel close to the min (avoids linear-filter blow-up)
  for (let i = 0; i < n; i++) if (Number.isNaN(out[i])) out[i] = empty;
  return { data: out, size: vol.size, order: vol.order, vmin, vmax, empty };
}

/** @param {object} cfg an MCX input (Shapes) or output (NIFTIData) document */
export function drawPreview(cfg) {
  if (!scene) return;
  scene.clear();
  lastVolume = null;
  fullVol = null; volIsLog = false; curFrame = 0; numFrames = 1;
  meshState = null;
  if (cfg && cfg.Shapes) {
    if (cfg.Shapes.constructor === Object && cfg.Shapes._ArraySize_) {
      drawshape({ Grid: { Size: cfg.Shapes._ArraySize_.slice(0, 3), Tag: 1 } });
      fullVol = decodeJDataArray(cfg.Shapes);
    } else if (cfg.Shapes.constructor === Object && cfg.Shapes.MeshNode) {
      drawmesh(cfg.Shapes); // tetrahedral mesh (MMC) domain
    } else {
      if (cfg.Domain && cfg.Domain.Dim) drawshape({ Grid: { Size: cfg.Domain.Dim, Tag: 1 } });
      if (Array.isArray(cfg.Shapes)) cfg.Shapes.forEach(drawshape);
    }
    if (cfg.Optode) {
      if (cfg.Optode.Source) drawsrc(cfg.Optode.Source);
      if (Array.isArray(cfg.Optode.Detector)) cfg.Optode.Detector.forEach(drawdet);
    }
  } else if (cfg && cfg.NIFTIData) {
    const dim = (cfg.NIFTIHeader && cfg.NIFTIHeader.Dim) || cfg.NIFTIData._ArraySize_;
    const vol = decodeJDataArray(cfg.NIFTIData);
    // mmc mesh-valued output (per-node/per-elem, no 'c' order tag): render on the input
    // mesh from the editor doc; grid outputs (mcx and DMMC alike) use the voxel path
    const meshdoc = state.doc && state.doc.Shapes && state.doc.Shapes.MeshNode ? state.doc.Shapes : null;
    const meshed = meshdoc && !String(vol.order || '').startsWith('c') && drawmeshOutput(meshdoc, vol, dim);
    if (!meshed) {
      if (dim) drawshape({ Grid: { Size: dim.slice(0, 3), Tag: 1 } });
      fullVol = vol;
      volIsLog = true; // fluence-like output -> log scale
    }
  }
  if (fullVol) {
    numFrames = fullVol.size.slice(3).reduce((a, b) => a * (b || 1), 1);
    lastVolume = frameMesh(curFrame);
    boundingbox.add(lastVolume);
  }
  syncFrameUI();
  dirty = true;
}

// ---- controls ------------------------------------------------------------------
function readCrossInto(uniforms) {
  const g = (id) => parseFloat(/** @type {HTMLInputElement} */ ($(id)).value);
  uniforms.u_minslice.value.set(g('#cross-x-low'), g('#cross-y-low'), g('#cross-z-low'));
  uniforms.u_maxslice.value.set(g('#cross-x-hi'), g('#cross-y-hi'), g('#cross-z-hi'));
}

function wireControls() {
  // Cross-section sliders + fixed-thickness slab: when a Min/Max slider (or the per-axis
  // thickness box) changes and thickness>0, keep a constant slab width in that axis and
  // slide it — the v1 behaviour. thickness is in voxels, normalized by the volume dim.
  /** @param {'x'|'y'|'z'} axis @param {'low'|'hi'} moved */
  const applyCross = (axis, moved) => {
    const idx = { x: 0, y: 1, z: 2 }[axis];
    const t = parseFloat(/** @type {HTMLInputElement} */ ($('#thick-' + axis)).value) || 0;
    const dimN = (lastDim && lastDim[idx]) || 1;
    if (t > 0) {
      const tn = Math.min(t / dimN, 1);
      const low = /** @type {HTMLInputElement} */ ($('#cross-' + axis + '-low'));
      const hi = /** @type {HTMLInputElement} */ ($('#cross-' + axis + '-hi'));
      if (moved === 'hi') low.value = String(Math.max(parseFloat(hi.value) - tn, 0));
      else hi.value = String(Math.min(parseFloat(low.value) + tn, 1));
    }
    if (lastVolume) { readCrossInto(lastVolume.material.uniforms); dirty = true; }
    if (meshState) updateMeshCross();
  };
  for (const axis of /** @type {const} */ (['x', 'y', 'z'])) {
    $('#cross-' + axis + '-low').addEventListener('input', () => applyCross(axis, 'low'));
    $('#cross-' + axis + '-hi').addEventListener('input', () => applyCross(axis, 'hi'));
    $('#thick-' + axis).addEventListener('input', () => applyCross(axis, 'low'));
  }

  // 4D frame spinner: re-render the selected 3D frame of a 4D+ volume (or recolor a
  // mesh-valued mmc output for the selected gate/source frame)
  const frameEl = /** @type {HTMLInputElement} */ ($('#frame-num'));
  if (frameEl) frameEl.addEventListener('input', () => {
    if (numFrames <= 1) return;
    const fm = Math.min(Math.max((parseInt(frameEl.value, 10) || 1) - 1, 0), numFrames - 1);
    if (meshState && meshState.outVol) {
      if (fm !== curFrame) { curFrame = fm; setMeshFrame(fm); }
      return;
    }
    if (!fullVol) return;
    const f = fm;
    if (f === curFrame) return;
    curFrame = f;
    if (lastVolume) { // free the old frame's GPU texture before building the next
      if (lastVolume.parent) lastVolume.parent.remove(lastVolume);
      lastVolume.material.uniforms.u_data.value.dispose();
      lastVolume.material.dispose();
      lastVolume.geometry.dispose();
    }
    lastVolume = frameMesh(curFrame);
    boundingbox.add(lastVolume);
    dirty = true;
  });

  // capture-thumbnail button (updates every .thumb-view, incl. the Share tab)
  $('#update-thumb').addEventListener('click', () => { captureThumbnail(); });

  // colormap picker
  const cmapSel = /** @type {HTMLSelectElement} */ ($('#cmap'));
  if (cmapSel) {
    for (const nm of COLORMAP_NAMES) {
      const o = document.createElement('option'); o.value = o.textContent = nm; cmapSel.appendChild(o);
    }
    cmapSel.value = currentCmap;
    cmapSel.addEventListener('change', () => setColormap(cmapSel.value));
  }
  for (const [id, axis] of [['#clim-low', 'x'], ['#clim-hi', 'y']]) {
    $(id).addEventListener('input', (e) => {
      if (!lastVolume) return;
      lastVolume.material.uniforms.u_clim.value[axis] = parseFloat(/** @type {HTMLInputElement} */ (e.target).value);
      dirty = true;
    });
  }
  // isosurface opacity slider
  $('#iso-alpha').addEventListener('input', (e) => {
    if (!lastVolume) return;
    lastVolume.material.uniforms.u_alpha.value = parseFloat(/** @type {HTMLInputElement} */ (e.target).value);
    dirty = true;
  });
  const swap = () => {
    if (!lastVolume) return;
    const mode = currentRenderMode();
    if (mode === 'dvr') { // DVR reads best semi-transparent
      const a = /** @type {HTMLInputElement} */ ($('#iso-alpha'));
      a.value = '0.5'; lastVolume.material.uniforms.u_alpha.value = 0.5;
    }
    lastVolume.material.fragmentShader = fragment(mode);
    lastVolume.material.needsUpdate = true;
    dirty = true;
  };
  $('#mip-radio').addEventListener('change', swap);
  $('#iso-radio').addEventListener('change', swap);
  $('#dvr-radio').addEventListener('change', swap);
  // contrast (gamma) slider
  $('#contrast').addEventListener('input', (e) => {
    if (!lastVolume) return;
    lastVolume.material.uniforms.u_gamma.value = parseFloat(/** @type {HTMLInputElement} */ (e.target).value) || 1;
    dirty = true;
  });
  // DVR gradient-shading toggle
  $('#dvr-shade').addEventListener('change', (e) => {
    if (!lastVolume) return;
    lastVolume.material.uniforms.u_shade.value = /** @type {HTMLInputElement} */ (e.target).checked ? 1 : 0;
    dirty = true;
  });

  const angles = { 'pos-x': [Math.PI / 2, Math.PI / 2], 'neg-x': [Math.PI / 2, 3 * Math.PI / 2], 'pos-y': [Math.PI / 2, Math.PI], 'neg-y': [Math.PI / 2, 0], 'pos-z': [0, 0], 'neg-z': [Math.PI, 0] };
  $('#views').addEventListener('click', (e) => {
    const v = /** @type {HTMLElement} */ (e.target).dataset.view;
    if (!v || !angles[v]) return;
    const [polar, azim] = angles[v];
    const save = [controls.minAzimuthAngle, controls.maxAzimuthAngle, controls.minPolarAngle, controls.maxPolarAngle];
    controls.minAzimuthAngle = controls.maxAzimuthAngle = azim;
    controls.minPolarAngle = controls.maxPolarAngle = polar;
    controls.update();
    [controls.minAzimuthAngle, controls.maxAzimuthAngle, controls.minPolarAngle, controls.maxPolarAngle] = save;
    dirty = true;
  });
}
