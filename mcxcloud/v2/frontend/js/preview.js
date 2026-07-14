// @ts-check
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $ } from './util.js';
import { decodeJDataArray, colormapRGBA } from './util.js';

// ---- GLSL volume raycaster (ported from MCX Cloud v1 / three.js VolumeShader) ----
// Kept verbatim; glslVersion GLSL3 lets three map varying/gl_FragColor/texture2D and
// permits sampler3D.
const VERT = /* glsl */ `
  varying vec4 v_nearpos;
  varying vec4 v_farpos;
  varying vec3 v_position;
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

function fragment(isMip) {
  return /* glsl */ `
  precision highp float;
  precision mediump sampler3D;
  uniform vec3 u_size;
  uniform float u_renderthreshold;
  uniform vec2 u_clim;
  uniform sampler3D u_data;
  uniform sampler2D u_cmdata;
  uniform vec3 u_minslice;
  uniform vec3 u_maxslice;
  varying vec3 v_position;
  varying vec4 v_nearpos;
  varying vec4 v_farpos;
  const int MAX_STEPS = 887;
  const int REFINEMENT_STEPS = 4;
  const float relative_step_size = 1.0;

  float sample1(vec3 texcoords){ return texture(u_data, texcoords.xyz).r; }
  vec4 apply_colormap(float val){ val=(val-u_clim.x)/(u_clim.y-u_clim.x); return texture(u_cmdata, vec2(val,0.5)); }

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
    float max_val=0.0; vec3 loc=start_loc;
    for(int iter=0; iter<=MAX_STEPS; iter++){ if(iter>=nsteps) break; max_val=max(sample1(loc),max_val); loc+=step; }
    gl_FragColor=apply_colormap(max_val);
  }
  void cast_iso(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray){
    vec3 dstep=1.5/u_size; vec3 loc=start_loc; float val=0.0;
    for(int iter=0; iter<=MAX_STEPS; iter++){ if(iter>=nsteps) break; val=sample1(loc); if(val>u_renderthreshold) break; loc+=step; }
    if(val>u_renderthreshold){
      vec3 iloc=loc-0.5*step; vec3 istep=step/float(REFINEMENT_STEPS);
      for(int i=0;i<REFINEMENT_STEPS;i++){ val=sample1(iloc); if(val>u_renderthreshold){ gl_FragColor=add_lighting(val,iloc,dstep,view_ray); return; } iloc+=istep; }
      gl_FragColor=add_lighting(val,iloc,dstep,view_ray);
    } else { gl_FragColor=vec4(0.0); }
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
    ${isMip ? 'cast_mip' : 'cast_iso'}(start_loc, step, nsteps, view_ray);
  }`;
}

const baseUniforms = () => ({
  u_size: { value: new THREE.Vector3(1, 1, 1) },
  u_renderthreshold: { value: 0.2 },
  u_clim: { value: new THREE.Vector2(0, 1) },
  u_data: { value: null },
  u_cmdata: { value: null },
  u_minslice: { value: new THREE.Vector3(0, 0, 0) },
  u_maxslice: { value: new THREE.Vector3(1, 1, 1) },
});

// ---- scene state ---------------------------------------------------------------
let scene, camera, renderer, controls, cmTexture;
let boundingbox, bbxsize = [1, 1, 1], lastVolume = null, lastDim = [];
let dirty = true;
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
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  host.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.minZoom = 0.3; controls.maxZoom = 60;
  controls.addEventListener('change', () => { dirty = true; });

  const cm = colormapRGBA('viridis');
  cmTexture = new THREE.DataTexture(cm, cm.length / 4, 1, THREE.RGBAFormat);
  cmTexture.minFilter = cmTexture.magFilter = THREE.LinearFilter;
  cmTexture.needsUpdate = true;

  window.addEventListener('resize', onResize);
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

function resetscene(s) {
  scene.clear();
  scene.add(new THREE.AmbientLight(0xffffff));
  const diag = Math.hypot(s[0], s[1], s[2]) || 1;
  camera.position.set(s[0] * 1.5, s[1] * 1.2, s[2] * 1.4);
  camera.lookAt(s[0] * 0.5, s[1] * 0.5, s[2] * 0.5);
  camera.zoom = 0.7 * Math.min((renderer.domElement.width) / diag, (renderer.domElement.height) / diag);
  camera.updateProjectionMatrix();
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
      g.translate(0, height * 0.5 - 1, 0); g.rotateX(Math.PI * 0.5);
      g.lookAt(c1.clone().sub(c0).normalize()); g.translate(c0.x + c1.x, c0.y + c1.y, c0.z + c1.z);
      m = new THREE.MeshBasicMaterial({ color: materialcolor[shape.Cylinder.Tag] || 0x00ffff, wireframe: true });
      boundingbox.add(new THREE.Mesh(g, m));
      break;
    }
  }
}

function drawsrc(src) {
  const dir = new THREE.Vector3(...src.Dir).normalize();
  const orig = new THREE.Vector3(...src.Pos);
  const len = Math.max(...bbxsize) * 0.2;
  boundingbox.add(new THREE.ArrowHelper(dir, orig, len, 0xffffff, 0.3 * len, 0.15 * len));
}
function drawdet(det) {
  const g = new THREE.SphereGeometry(det.R, 24, 24);
  g.translate(...det.Pos);
  boundingbox.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true })));
}

/**
 * Build the volume raycasting mesh from a decoded array.
 * @param {{data: Float32Array, size: number[]}} vol
 */
function drawvolume(vol) {
  const dim = vol.size;
  lastDim = dim;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < vol.data.length; i++) { const v = vol.data[i]; if (v < min) min = v; if (v > max) max = v; }
  if (!isFinite(min)) { min = 0; max = 1; }

  const tex = new THREE.Data3DTexture(vol.data, dim[0], dim[1], dim[2]);
  tex.format = THREE.RedFormat; tex.type = THREE.FloatType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.unpackAlignment = 1; tex.needsUpdate = true;

  const isMip = /** @type {HTMLInputElement} */ ($('#mip-radio')).checked;
  const uniforms = baseUniforms();
  uniforms.u_data.value = tex;
  uniforms.u_cmdata.value = cmTexture;
  uniforms.u_size.value.set(dim[0], dim[1], dim[2]);
  uniforms.u_clim.value.set(min, max);
  uniforms.u_renderthreshold.value = min + 0.2 * (max - min);
  readCrossInto(uniforms);

  const material = new THREE.ShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: fragment(isMip), side: THREE.BackSide, glslVersion: THREE.GLSL3,
  });
  const geometry = new THREE.BoxGeometry(dim[0], dim[1], dim[2]);
  geometry.translate(dim[0] * 0.5, dim[1] * 0.5, dim[2] * 0.5);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;

  // sync clim sliders
  for (const [id, val] of [['#clim-low', min], ['#clim-hi', max]]) {
    const el = /** @type {HTMLInputElement} */ ($(id));
    el.disabled = false; el.min = String(min); el.max = String(max); el.value = String(val);
  }
  return mesh;
}

/** apply log10 scaling (for fluence output) in place -> new Float32Array */
function logscale(vol) {
  const out = new Float32Array(vol.data.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.log(Math.max(vol.data[i], 1e-16));
  return { data: out, size: vol.size };
}

/** @param {object} cfg an MCX input (Shapes) or output (NIFTIData) document */
export function drawPreview(cfg) {
  if (!scene) return;
  scene.clear();
  lastVolume = null;
  if (cfg && cfg.Shapes) {
    if (cfg.Shapes.constructor === Object && cfg.Shapes._ArraySize_) {
      drawshape({ Grid: { Size: cfg.Shapes._ArraySize_, Tag: 1 } });
      lastVolume = drawvolume(decodeJDataArray(cfg.Shapes));
      boundingbox.add(lastVolume);
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
    if (dim) drawshape({ Grid: { Size: dim.slice(0, 3), Tag: 1 } });
    lastVolume = drawvolume(logscale(decodeJDataArray(cfg.NIFTIData)));
    boundingbox.add(lastVolume);
  }
  dirty = true;
}

// ---- controls ------------------------------------------------------------------
function readCrossInto(uniforms) {
  const g = (id) => parseFloat(/** @type {HTMLInputElement} */ ($(id)).value);
  uniforms.u_minslice.value.set(g('#cross-x-low'), g('#cross-y-low'), g('#cross-z-low'));
  uniforms.u_maxslice.value.set(g('#cross-x-hi'), g('#cross-y-hi'), g('#cross-z-hi'));
}

function wireControls() {
  for (const id of ['cross-x-low', 'cross-x-hi', 'cross-y-low', 'cross-y-hi', 'cross-z-low', 'cross-z-hi']) {
    $('#' + id).addEventListener('input', () => {
      if (lastVolume) { readCrossInto(lastVolume.material.uniforms); dirty = true; }
    });
  }
  for (const [id, axis] of [['#clim-low', 'x'], ['#clim-hi', 'y']]) {
    $(id).addEventListener('input', (e) => {
      if (!lastVolume) return;
      lastVolume.material.uniforms.u_clim.value[axis] = parseFloat(/** @type {HTMLInputElement} */ (e.target).value);
      dirty = true;
    });
  }
  const swap = () => {
    if (!lastVolume) return;
    const isMip = /** @type {HTMLInputElement} */ ($('#mip-radio')).checked;
    lastVolume.material.fragmentShader = fragment(isMip);
    lastVolume.material.needsUpdate = true;
    dirty = true;
  };
  $('#mip-radio').addEventListener('change', swap);
  $('#iso-radio').addEventListener('change', swap);

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
