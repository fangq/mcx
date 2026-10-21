// @ts-check
// Tetrahedral-mesh geometry helpers for the MMC preview, ported from iso2mesh /
// pyiso2mesh (volface + qmeshcut). Pure typed-array math with no dependencies, so the
// same file runs under node for tests. Node indices are 1-based (MMC convention);
// positions returned are ready-to-upload triangle soups.

/** the 4 faces of a tetrahedron (iso2mesh surfedge face set), wound so normals point
 *  OUTWARD for positively oriented tets (mmc requires positive orientation) */
const TET_FACES = [[0, 2, 1], [1, 3, 0], [0, 3, 2], [1, 2, 3]];

/** the 6 edges of a tetrahedron in combinations(4,2) order — qmeshcut's quad-cycle
 *  reordering [0,1,3,2] below is only correct in this edge order */
const TET_EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/**
 * Extract the renderable surface of a tet mesh: the EXTERIOR faces (referenced by
 * exactly one tetrahedron — iso2mesh volface) plus the REGION INTERFACES (faces shared
 * by two tets carrying different region tags), so interior inclusions and layer
 * boundaries are visible through a semi-transparent exterior. Rendering only these
 * faces (instead of all tets) is what keeps large meshes fast.
 * @param {ArrayLike<number>} elem flattened element rows (1-based node ids, tag last)
 * @param {number} ne  element count
 * @param {number} stride row length (5 for [n1..n4,tag]; 11 for tet10 — corners first)
 * @returns {{ faces: number[], owner: number[] }} faces = flattened [a,b,c] node-id
 *   triples (1-based, outward winding); owner = owning element index per face —
 *   interface faces are owned by the HIGHER-tagged neighbor (usually the inclusion)
 */
export function volface(elem, ne, stride) {
  const tag = (/** @type {number} */ e) => elem[e * stride + stride - 1];
  /** @type {Map<string, { n: number, a: number, b: number, c: number, e: number, iface: boolean }>} */
  const seen = new Map();
  for (let e = 0; e < ne; e++) {
    const o = e * stride;
    for (const [i, j, k] of TET_FACES) {
      const a = elem[o + i], b = elem[o + j], c = elem[o + k];
      // order-independent key (sorted triple)
      const lo = Math.min(a, b, c), hi = Math.max(a, b, c);
      const key = lo + ',' + (a + b + c - lo - hi) + ',' + hi;
      const hit = seen.get(key);
      if (hit) {
        hit.n++;
        if (hit.n === 2 && tag(e) !== tag(hit.e)) {
          hit.iface = true;
          if (tag(e) > tag(hit.e)) hit.e = e; // color by the inclusion side
        }
      } else seen.set(key, { n: 1, a, b, c, e, iface: false });
    }
  }
  const faces = [], owner = [];
  for (const f of seen.values()) {
    if (f.n === 1 || f.iface) { faces.push(f.a, f.b, f.c); owner.push(f.e); }
  }
  return { faces, owner };
}

/**
 * Cut a tet mesh with the axis-aligned plane `axis = pos` (iso2mesh qmeshcut,
 * specialized to axis-aligned planes): every tet straddling the plane contributes a
 * triangle (3 cut edges) or a quad (4 cut edges, emitted as 2 triangles) whose corners
 * are linear interpolations along the cut edges.
 * @param {ArrayLike<number>} node flattened [x,y,z] per node
 * @param {ArrayLike<number>} elem flattened element rows (1-based node ids)
 * @param {number} ne element count
 * @param {number} stride element row length
 * @param {0|1|2} axis 0=x, 1=y, 2=z
 * @param {number} pos world coordinate of the cutting plane
 * @param {ArrayLike<number>} [nodeval] optional per-node scalars; when given, the value
 *   is linearly interpolated at every cut point (iso2mesh qmeshcut's cutvalue)
 * @returns {{ positions: number[], elemid: number[], values: number[] | null }}
 *   positions = flattened triangle soup [x,y,z]*3 per triangle; elemid = source element
 *   index per TRIANGLE (for per-region coloring or per-element values); values =
 *   interpolated scalar per triangle VERTEX (3 per triangle) when nodeval was given
 */
export function qmeshcut(node, elem, ne, stride, axis, pos, nodeval) {
  const positions = [], elemid = [];
  const values = nodeval ? [] : null;
  const d = [0, 0, 0, 0], s = [0, 0, 0, 0], nid = [0, 0, 0, 0], pid = [0, 0, 0, 0];
  const px = [0, 0, 0, 0], py = [0, 0, 0, 0], pz = [0, 0, 0, 0], pv = [0, 0, 0, 0];
  for (let e = 0; e < ne; e++) {
    const o = e * stride;
    let ssum = 0;
    for (let i = 0; i < 4; i++) {
      pid[i] = elem[o + i] - 1;
      nid[i] = pid[i] * 3;
      d[i] = node[nid[i] + axis] - pos;
      s[i] = d[i] >= 0 ? 1 : -1;
      ssum += s[i];
    }
    if (ssum === 4 || ssum === -4) continue; // tet entirely on one side
    let ncut = 0;
    for (const [i, j] of TET_EDGES) {
      if (s[i] + s[j] !== 0) continue;
      const t = d[i] / (d[i] - d[j]); // interpolation weight toward node j
      px[ncut] = node[nid[i]] + t * (node[nid[j]] - node[nid[i]]);
      py[ncut] = node[nid[i] + 1] + t * (node[nid[j] + 1] - node[nid[i] + 1]);
      pz[ncut] = node[nid[i] + 2] + t * (node[nid[j] + 2] - node[nid[i] + 2]);
      if (nodeval) pv[ncut] = nodeval[pid[i]] + t * (nodeval[pid[j]] - nodeval[pid[i]]);
      ncut++;
    }
    const tri = (a, b, c) => {
      positions.push(px[a], py[a], pz[a], px[b], py[b], pz[b], px[c], py[c], pz[c]);
      if (values) values.push(pv[a], pv[b], pv[c]);
      elemid.push(e);
    };
    if (ncut === 3) tri(0, 1, 2);
    else if (ncut === 4) { tri(0, 1, 3); tri(0, 3, 2); } // quad cycle 0,1,3,2
  }
  return { positions, elemid, values };
}

/**
 * Bounding box of a flattened [x,y,z] node array.
 * @param {ArrayLike<number>} node @param {number} nn node count
 * @returns {{ min: number[], max: number[] }}
 */
export function meshBBox(node, nn) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nn; i++) {
    for (let a = 0; a < 3; a++) {
      const v = node[i * 3 + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max };
}
