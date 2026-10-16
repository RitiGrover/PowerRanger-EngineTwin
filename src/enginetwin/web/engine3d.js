/* Procedural inline-four engine for EngineTwin.
 * Built from primitives, so no model file has to be downloaded. Every part is a named group that can be
 * lifted apart (explode), made translucent (x-ray) or tinted (fault highlight). Geometry is authored in the
 * assembled pose; a part's group offset is the only thing the explode view changes.
 * Inlined into the landing page and the engine map. `THREE` is passed in by the host page. */
function buildEngine(THREE, opts) {
  opts = opts || {};
  const root = new THREE.Group();
  const parts = {};
  const allMats = [];
  const R = 0.28, L = 0.98, SP = 1.0;           // crank throw, rod length, bore spacing
  const FIRING = [1, 3, 4, 2];                   // inline-4 firing order
  const X = (i) => (i - 1.5) * SP;               // cylinder i = 0..3
  const PHASE = [0, Math.PI, Math.PI, 0];        // crank throw angles for cylinders 1..4
  const FIRE_IDX = { 1: 0, 3: 1, 4: 2, 2: 3 };   // cylinder -> slot in the firing sequence

  const KINDS = {
    alu:    { color: 0xa3abb3, metalness: 0.85, roughness: 0.34 },
    block:  { color: 0x6a727a, metalness: 0.75, roughness: 0.46 },
    cast:   { color: 0x474e55, metalness: 0.7,  roughness: 0.5 },
    black:  { color: 0x23272c, metalness: 0.35, roughness: 0.55 },
    rubber: { color: 0x1a1c1f, metalness: 0.05, roughness: 0.85 },
    steel:  { color: 0xb7bec5, metalness: 1.0,  roughness: 0.25 },
    dark:   { color: 0x23272c, metalness: 0.8,  roughness: 0.4 },
    cap:    { color: 0x7a4a40, metalness: 0.2,  roughness: 0.6 },
    brass:  { color: 0x8f7d60, metalness: 0.9,  roughness: 0.35 },
  };
  const EDGE_COL = new THREE.Color(0xa9c4d4);
  const HOT = new THREE.Color(0xd9795f);
  const SHELL = new Set(['block', 'head', 'cover', 'oilpan', 'timingcover']);

  function mat(kind) {
    const m = new THREE.MeshStandardMaterial(Object.assign({}, KINDS[kind]));
    m.userData = { base: m.color.clone(), kind, opacity: 1 };
    allMats.push(m);
    return m;
  }
  function part(id, label, explode, extra) {
    const g = new THREE.Group();
    g.name = id;
    root.add(g);
    parts[id] = Object.assign({ id, label, group: g, explode: new THREE.Vector3(...(explode || [0, 0, 0])), mats: [], edges: [],
      state: 'normal', xray: SHELL.has(id), amount: 0, delay: 0 }, extra || {});
    return parts[id];
  }
  function reg(p, mesh, withEdges, parent) {
    (parent || p.group).add(mesh);
    if (mesh.material && !p.mats.includes(mesh.material)) p.mats.push(mesh.material);
    if (withEdges !== false && opts.edges !== false && mesh.geometry && mesh.geometry.type !== 'TubeGeometry') {
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 32),
        new THREE.LineBasicMaterial({ color: EDGE_COL, transparent: true, opacity: 0.2, depthWrite: false }));
      mesh.add(e); p.edges.push(e);
    }
    return mesh;
  }
  function box(p, w, h, d, kind, x, y, z, parent) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(kind));
    m.position.set(x, y, z); return reg(p, m, true, parent);
  }
  // cylinder along axis 'x' | 'y' | 'z'
  function cyl(p, rt, rb, h, kind, axis, x, y, z, seg, parent) {
    const geo = new THREE.CylinderGeometry(rt, rb, h, seg || 40);
    if (axis === 'x') geo.rotateZ(Math.PI / 2); else if (axis === 'z') geo.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(geo, mat(kind)); m.position.set(x, y, z); return reg(p, m, true, parent);
  }
  function tube(p, pts, r, kind, closed, parent) {
    const curve = new THREE.CatmullRomCurve3(pts.map((v) => new THREE.Vector3(...v)), !!closed, 'catmullrom', 0.4);
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, r, 14, !!closed), mat(kind));
    return reg(p, m, false, parent);
  }

  /* ---------------------------------------------------------------- block, crankcase, oil pan */
  let p = part('block', 'Engine block', [0, 0, 0]);
  box(p, 4.5, 1.3, 1.3, 'block', 0, 0.9, 0);
  box(p, 4.5, 0.62, 1.5, 'block', 0, -0.05, 0);
  for (let i = 0; i < 5; i++) { box(p, 0.1, 1.15, 0.08, 'cast', (i - 2) * 1.0, 0.85, 0.69); box(p, 0.1, 1.15, 0.08, 'cast', (i - 2) * 1.0, 0.85, -0.69); }
  for (let i = 0; i < 4; i++) cyl(p, 0.37, 0.37, 0.02, 'black', 'y', X(i), 1.56, 0, 40);   // bore openings on the deck
  cyl(p, 0.2, 0.2, 0.08, 'steel', 'z', 1.55, 0.2, 0.69, 20);                              // oil-pressure boss
  p = part('timingcover', 'Timing cover', [-1.1, 0, 0]);
  box(p, 0.3, 1.95, 1.4, 'alu', -2.4, 0.78, 0);
  p = part('oilpan', 'Oil pan', [0, -1.5, 0]);
  box(p, 3.7, 0.7, 1.25, 'alu', 0, -0.65, 0);
  for (let i = 0; i < 7; i++) box(p, 0.05, 0.5, 1.3, 'alu', -1.5 + i * 0.5, -0.65, 0);
  cyl(p, 0.09, 0.09, 0.08, 'steel', 'y', 0.9, -1.02, 0, 6);

  /* ---------------------------------------------------------------- head, valve cover, coils */
  p = part('head', 'Cylinder head', [0, 1.05, 0]);
  box(p, 4.5, 0.6, 1.3, 'cast', 0, 1.85, 0);
  p = part('cover', 'Valve cover', [0, 2.35, 0]);
  box(p, 4.2, 0.42, 1.0, 'black', 0, 2.36, 0);
  for (let i = 0; i < 11; i++) box(p, 0.04, 0.05, 0.9, 'dark', -2.0 + i * 0.4, 2.59, 0);
  for (let i = 0; i < 6; i++) { cyl(p, 0.05, 0.05, 0.05, 'steel', 'y', -2.0 + i * 0.8, 2.6, 0.46, 6); cyl(p, 0.05, 0.05, 0.05, 'steel', 'y', -2.0 + i * 0.8, 2.6, -0.46, 6); }
  cyl(p, 0.17, 0.17, 0.1, 'dark', 'y', -1.85, 2.62, 0.2, 24);                                // oil filler cap
  for (let i = 0; i < 4; i++) {
    p = part('coil' + (i + 1), 'Ignition coil ' + (i + 1), [0, 2.6, 0]);
    cyl(p, 0.12, 0.13, 0.46, 'black', 'y', X(i), 2.82, 0, 20);
    cyl(p, 0.15, 0.15, 0.14, 'cap', 'y', X(i), 3.12, 0, 20);
    box(p, 0.1, 0.12, 0.1, 'black', X(i), 3.0, 0.14);
  }

  /* ---------------------------------------------------------------- valvetrain: cams and valves */
  const camShafts = [];
  p = part('cams', 'Camshafts', [0, 1.6, 0]);
  [-0.28, 0.28].forEach((zz, k) => {
    const sub = new THREE.Group(); sub.position.set(0, 2.28, zz); p.group.add(sub);
    cyl(p, 0.07, 0.07, 4.1, 'steel', 'x', 0, 0, 0, 14, sub);
    for (let i = 0; i < 4; i++) { const lobe = box(p, 0.16, 0.24, 0.1, 'steel', X(i), 0.06, 0, sub); lobe.rotation.x = i * 0.9 + k; }
    camShafts.push(sub);
  });
  const valves = [];
  for (let i = 0; i < 4; i++) {
    p = part('valve' + (i + 1), 'Valves, cylinder ' + (i + 1), [0, 0.85, 0]);
    const set = [];
    [-0.22, 0.22].forEach((zz) => {
      const g = new THREE.Group(); p.group.add(g);
      cyl(p, 0.025, 0.025, 0.85, 'steel', 'y', X(i), 2.0, zz, 8, g);
      cyl(p, 0.13, 0.13, 0.03, 'steel', 'y', X(i), 1.6, zz, 20, g);
      cyl(p, 0.07, 0.07, 0.3, 'brass', 'y', X(i), 2.08, zz, 12, g);
      set.push(g);
    });
    valves.push(set);
  }

  /* ---------------------------------------------------------------- rotating assembly */
  const pistons = [], rods = [], flashes = [];
  for (let i = 0; i < 4; i++) {
    p = part('piston' + (i + 1), 'Piston ' + (i + 1), [0, 0.55, 0]);
    const pg = new THREE.Group(); p.group.add(pg);
    cyl(p, 0.33, 0.33, 0.4, 'alu', 'y', 0, 0.06, 0, 36, pg);
    cyl(p, 0.335, 0.335, 0.035, 'dark', 'y', 0, 0.21, 0, 36, pg);
    cyl(p, 0.335, 0.335, 0.035, 'dark', 'y', 0, 0.15, 0, 36, pg);
    cyl(p, 0.06, 0.06, 0.62, 'steel', 'z', 0, 0, 0, 10, pg);
    const flash = new THREE.Mesh(new THREE.SphereGeometry(0.2, 18, 12),
      new THREE.MeshBasicMaterial({ color: 0xe8ccb8, transparent: true, opacity: 0, depthWrite: false }));
    flash.position.set(0, 0.32, 0); pg.add(flash);
    pg.position.x = X(i); pistons.push(pg); flashes.push(flash);
    p = part('rod' + (i + 1), 'Connecting rod ' + (i + 1), [0, -0.1, 0]);
    const rg = new THREE.Group(); p.group.add(rg);
    box(p, 0.09, L, 0.09, 'steel', 0, 0, 0, rg);
    cyl(p, 0.1, 0.1, 0.12, 'steel', 'x', 0, -L / 2, 0, 20, rg);
    cyl(p, 0.07, 0.07, 0.12, 'steel', 'x', 0, L / 2, 0, 16, rg);
    rg.position.x = X(i); rods.push(rg);
  }
  p = part('crank', 'Crankshaft', [0, -0.9, 0]);
  const crank = p.group;
  cyl(p, 0.11, 0.11, 4.9, 'steel', 'x', 0, 0, 0, 24);
  for (let i = 0; i < 4; i++) {
    const a = PHASE[i];
    const py = R * Math.cos(a), pz = R * Math.sin(a);
    cyl(p, 0.11, 0.11, 0.16, 'steel', 'x', X(i), py, pz, 20);
    [-0.26, 0.26].forEach((dx) => {
      const web = box(p, 0.1, 0.12, 0.34, 'steel', X(i) + dx, py / 2, pz / 2);
      web.rotation.x = -Math.atan2(pz, py); web.scale.y = 1.2;
      const cw = cyl(p, 0.26, 0.26, 0.08, 'dark', 'x', X(i) + dx, -py * 0.9, -pz * 0.9, 24);
    });
  }
  p = part('pulley', 'Crank pulley and belt', [-1.6, 0, 0]);
  const pulleyG = new THREE.Group(); pulleyG.position.set(-2.72, 0, 0); p.group.add(pulleyG);
  cyl(p, 0.5, 0.5, 0.2, 'black', 'x', 0, 0, 0, 40, pulleyG);
  cyl(p, 0.36, 0.36, 0.26, 'steel', 'x', 0, 0, 0, 30, pulleyG);
  cyl(p, 0.1, 0.1, 0.32, 'dark', 'x', 0, 0, 0, 6, pulleyG);
  // belt: convex hull around the three pulley circles
  const circles = [[0, 0, 0.5], [0.9, 0.95, 0.3], [-0.12, 1.08, 0.27]];  // y, z, r
  const pts2 = [];
  circles.forEach(([cy, cz, r]) => { for (let a = 0; a < 24; a++) pts2.push([cy + (r + 0.03) * Math.cos(a / 24 * 6.2832), cz + (r + 0.03) * Math.sin(a / 24 * 6.2832)]); });
  pts2.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = []; pts2.forEach((q) => { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); });
  const upper = []; pts2.slice().reverse().forEach((q) => { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); });
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  const beltPts = []; hull.forEach((q, i) => { if (i % 3 === 0) beltPts.push([-2.72, q[0], q[1]]); });
  const belt = tube(p, beltPts, 0.035, 'rubber', true);
  belt.scale.set(4, 1, 1); belt.position.x = 2.72 * 3;     // flatten the belt along the crank axis
  circles.slice(1).forEach(([cy, cz, r]) => cyl(p, r, r, 0.22, 'dark', 'x', -2.72, cy, cz, 28));
  p = part('alternator', 'Alternator', [-1.4, 0.5, 0.8]);
  cyl(p, 0.27, 0.27, 0.5, 'alu', 'x', -2.55, 0.95, 0.9, 30);
  cyl(p, 0.2, 0.2, 0.2, 'cast', 'x', -2.15, 0.95, 0.9, 24);
  p = part('flywheel', 'Flywheel', [1.5, 0, 0]);
  const flyG = new THREE.Group(); flyG.position.set(2.5, 0, 0); p.group.add(flyG);
  cyl(p, 0.78, 0.78, 0.14, 'dark', 'x', 0, 0, 0, 56, flyG);
  cyl(p, 0.82, 0.82, 0.06, 'steel', 'x', 0.05, 0, 0, 56, flyG);
  cyl(p, 0.22, 0.22, 0.2, 'steel', 'x', 0, 0, 0, 24, flyG);

  /* ---------------------------------------------------------------- air path: intake, throttle, MAF */
  p = part('plenum', 'Intake plenum', [0, 0.9, 1.6]);
  cyl(p, 0.3, 0.3, 3.7, 'black', 'x', 0, 2.55, 1.9, 40);
  box(p, 3.6, 0.06, 0.1, 'dark', 0, 2.78, 1.9);
  p = part('runners', 'Intake runners', [0, 0.5, 1.2]);
  for (let i = 0; i < 4; i++) {
    tube(p, [[X(i), 2.4, 1.9], [X(i), 2.04, 2.1], [X(i), 1.76, 1.72], [X(i), 1.85, 1.05], [X(i), 1.85, 0.7]], 0.13, 'black');
    box(p, 0.4, 0.4, 0.07, 'cast', X(i), 1.85, 0.68);
  }
  p = part('throttle', 'Throttle body', [1.3, 0.5, 0.8]);
  cyl(p, 0.33, 0.33, 0.52, 'alu', 'x', 2.1, 2.55, 1.9, 36);
  cyl(p, 0.27, 0.27, 0.025, 'dark', 'x', 2.1, 2.55, 1.9, 30);
  box(p, 0.22, 0.28, 0.22, 'black', 2.1, 2.55, 2.28);
  p = part('inlet', 'Air inlet hose', [2.2, 0.5, 0.8]);
  cyl(p, 0.27, 0.27, 1.0, 'rubber', 'x', 3.1, 2.55, 1.9, 30);
  for (let i = 0; i < 5; i++) cyl(p, 0.285, 0.285, 0.04, 'dark', 'x', 2.8 + i * 0.2, 2.55, 1.9, 30);
  p = part('maf', 'Mass air-flow sensor', [2.2, 1.6, 0.8]);
  box(p, 0.34, 0.2, 0.3, 'black', 3.05, 2.88, 1.9);
  cyl(p, 0.05, 0.05, 0.14, 'steel', 'y', 3.05, 3.05, 1.9, 10);
  p = part('rail', 'Fuel rail', [0, 1.5, 0.7]);
  cyl(p, 0.06, 0.06, 3.7, 'steel', 'x', 0, 2.0, 1.0, 16);
  for (let i = 0; i < 4; i++) {
    p = part('inj' + (i + 1), 'Fuel injector ' + (i + 1), [0, 1.2, 0.5]);
    const inj = cyl(p, 0.05, 0.04, 0.3, 'brass', 'y', X(i), 1.9, 0.9, 12);
    inj.rotation.x = -0.45; inj.position.z = 0.88;
  }

  /* ---------------------------------------------------------------- exhaust and sensors */
  p = part('exhaust', 'Exhaust manifold', [0, 0.1, -1.6]);
  for (let i = 0; i < 4; i++) {
    tube(p, [[X(i), 1.8, -0.7], [X(i), 1.72, -1.1], [X(i) * 0.4, 1.35, -1.55]], 0.1, 'cast');
    box(p, 0.34, 0.34, 0.07, 'cast', X(i), 1.8, -0.68);
  }
  cyl(p, 0.17, 0.17, 1.2, 'cast', 'x', 0, 1.3, -1.7, 24);
  tube(p, [[0, 1.3, -1.7], [0, 0.6, -1.95], [0, -0.3, -1.95], [0, -0.9, -1.9]], 0.19, 'cast');
  p = part('o2', 'Oxygen sensor', [0.7, 0.9, -1.4]);
  const o2 = cyl(p, 0.06, 0.06, 0.34, 'steel', 'y', 0.2, 1.05, -2.15, 12);
  o2.rotation.z = 0.0; o2.rotation.x = Math.PI / 2; o2.position.set(0.0, 1.0, -2.17);
  tube(p, [[0, 1.0, -2.3], [0.2, 1.2, -2.7], [0.8, 1.5, -2.6]], 0.02, 'black');
  p = part('knock', 'Knock / vibration sensor', [0, 0, 1.4]);
  cyl(p, 0.1, 0.1, 0.2, 'steel', 'z', 0.0, 0.85, 0.74, 6);
  box(p, 0.16, 0.16, 0.12, 'black', 0, 0.85, 0.88);
  p = part('ect', 'Coolant temperature sensor', [-1.2, 1.3, 0.9]);
  cyl(p, 0.08, 0.08, 0.3, 'brass', 'y', -2.0, 2.28, 0.35, 6);
  box(p, 0.14, 0.12, 0.14, 'black', -2.0, 2.46, 0.35);

  // bolts on the head flange, purely visual
  p = parts['head'];
  for (let i = 0; i < 10; i++) { cyl(p, 0.05, 0.05, 0.04, 'steel', 'y', -2.0 + i * 0.44, 2.16, 0.55, 6); cyl(p, 0.05, 0.05, 0.04, 'steel', 'y', -2.0 + i * 0.44, 2.16, -0.55, 6); }

  /* ---------------------------------------------------------------- anchors for labels */
  const anchors = {};
  function anchor(id, partId, x, y, z) {
    const o = new THREE.Object3D(); o.position.set(x, y, z); parts[partId].group.add(o); anchors[id] = o; return o;
  }
  anchor('mic', 'cover', 0.2, 2.9, 0.6);
  anchor('accel', 'knock', 0, 0.85, 1.0);
  anchor('ckp', 'pulley', -3.0, 0, 0);
  anchor('maf', 'maf', 3.05, 3.2, 1.9);
  anchor('throttle', 'throttle', 2.1, 2.9, 1.9);
  anchor('ect', 'ect', -2.0, 2.6, 0.35);
  anchor('timing', 'coil4', 1.5, 3.3, 0);
  anchor('o2', 'o2', 0.8, 1.5, -2.6);
  anchor('coil2', 'coil2', -0.5, 3.35, 0);
  anchor('valve1', 'valve1', -1.5, 2.55, 0.25);
  anchor('inj2', 'inj2', -0.5, 1.9, 0.95);
  anchor('piston2', 'piston2', -0.5, 1.2, 0.35);

  /* ---------------------------------------------------------------- behaviour */
  const state = { crank: 0, speed: 0.55, explode: 0, explodeTarget: 0, xray: false, focus: false, hot: new Set(), fault: 'healthy', severity: 0.8 };
  const clock = new THREE.Clock();
  const tmp = new THREE.Color();
  let knockBurst = [0, 0, 0, 0];

  function setExplode(v) { state.explodeTarget = v; }
  function setXray(v) { state.xray = !!v; }
  function setFocus(v) { state.focus = !!v; }
  function setHot(ids) { state.hot = new Set(ids || []); }
  function setFault(fault, severity) { state.fault = fault; if (severity != null) state.severity = severity; }
  function setSpeed(revPerSec) { state.speed = revPerSec; }

  function update(dt) {
    state.crank += dt * state.speed * Math.PI * 2;
    // explode with a small stagger so parts separate in a wave rather than all at once
    state.explode += (state.explodeTarget - state.explode) * Math.min(1, dt * 3.2);
    const keys = Object.keys(parts);
    keys.forEach((k, idx) => {
      const q = parts[k];
      const s = Math.max(0, Math.min(1, state.explode * 1.15 - (idx % 7) * 0.02));
      const e = s * s * (3 - 2 * s);
      q.group.position.set(q.explode.x * e, q.explode.y * e, q.explode.z * e);
      // visual state
      const hot = state.hot.has(k);
      const dim = state.focus && state.hot.size > 0 && !hot;
      const targetOp = (state.xray && q.xray ? 0.2 : 1) * (dim ? 0.3 : 1);
      const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 360);
      q.mats.forEach((m) => {
        m.opacity += (targetOp - m.opacity) * Math.min(1, dt * 8);
        m.transparent = m.opacity < 0.985; m.depthWrite = m.opacity > 0.6;
        if (hot) { tmp.copy(HOT); m.emissive.copy(tmp).multiplyScalar(0.18 + 0.2 * pulse); m.color.copy(m.userData.base).lerp(HOT, 0.38); }
        else { m.emissive.multiplyScalar(0.85); m.color.lerp(m.userData.base, 0.15); }
      });
      q.edges.forEach((ln) => {
        const tgt = (hot ? 0.9 : (dim ? 0.06 : (state.xray && q.xray ? 0.28 : 0.2)));
        ln.material.opacity += (tgt - ln.material.opacity) * Math.min(1, dt * 8);
        ln.material.color.copy(hot ? HOT : EDGE_COL);
      });
    });
    const th = state.crank;
    crank.rotation.x = th; pulleyG.rotation.x = th; flyG.rotation.x = th;
    camShafts.forEach((c) => { c.rotation.x = th / 2; });
    const cyc = ((th % (4 * Math.PI)) + 4 * Math.PI) % (4 * Math.PI);
    const faults = state.fault.split('+');
    for (let i = 0; i < 4; i++) {
      const a = th + PHASE[i];
      const py = R * Math.cos(a), pz = R * Math.sin(a);
      const yp = py + Math.sqrt(L * L - (R * Math.sin(a)) ** 2);
      pistons[i].position.y = yp;
      const rg = rods[i];
      rg.position.y = (py + yp) / 2; rg.position.z = pz / 2;
      rg.rotation.x = Math.atan2(-pz, yp - py);
      // combustion flash in the firing slot; a misfire weakens its cylinder, knock adds a ringing second burst
      const slot = FIRE_IDX[i + 1] * Math.PI;
      let d = cyc - slot; if (d < 0) d += 4 * Math.PI;
      let f = d < 0.9 ? Math.exp(-d * 3.6) : 0;
      if (faults.includes('misfire') && i === 1) f *= 1 - state.severity;
      if (faults.includes('knock')) {
        if (d < 0.05 && Math.random() < state.severity * 0.5) knockBurst[i] = 1;
        knockBurst[i] *= 0.82;
        f = Math.min(1, f + knockBurst[i] * 0.7 * (Math.random() > 0.4 ? 1 : 0.3));
      }
      flashes[i].material.opacity = Math.min(0.85, f * 0.85);
      flashes[i].scale.setScalar(0.7 + f * 0.7);
      // valves lift once per cycle; a worn valve on cylinder 1 chatters
      const lift = Math.max(0, Math.sin((cyc - slot + Math.PI * 0.3) * 0.5)) * 0.045;
      valves[i].forEach((g, k) => {
        let l = k === 0 ? lift : Math.max(0, Math.sin((cyc - slot - Math.PI * 0.9) * 0.5)) * 0.045;
        if (faults.includes('valve_wear') && i === 0) l += (Math.random() - 0.5) * 0.04 * state.severity;
        g.position.y = -l;
      });
    }
  }

  // centre the model so orbit controls pivot on its middle
  root.position.set(-0.3, -1.2, 0);

  return { root, parts, anchors, update, setExplode, setXray, setFocus, setHot, setFault, setSpeed, state,
    partIds: Object.keys(parts) };
}

/* Shared Three.js scene boilerplate: renderer, lights, environment, controls and a resize observer. */
function createStage(THREE, OrbitControls, RoomEnvironment, canvasHost, opts) {
  opts = opts || {};
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = opts.exposure || 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  canvasHost.appendChild(renderer.domElement);
  renderer.domElement.style.display = 'block';
  const scene = new THREE.Scene();
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = opts.envIntensity != null ? opts.envIntensity : 0.9;
  const key = new THREE.DirectionalLight(0xdfe8f0, 1.7); key.position.set(5, 8, 6); scene.add(key);
  const rim = new THREE.DirectionalLight(0x8aa3b5, 1.1); rim.position.set(-6, 3, -5); scene.add(rim);
  scene.add(new THREE.AmbientLight(0x9aa6b0, 0.25));
  const camera = new THREE.PerspectiveCamera(opts.fov || 32, 1, 0.1, 100);
  camera.position.set(...(opts.camera || [7.2, 4.4, 8.6]));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.target.set(...(opts.target || [0, 0.3, 0]));
  controls.enablePan = false; controls.minDistance = 6; controls.maxDistance = 22;
  controls.autoRotate = opts.autoRotate !== false; controls.autoRotateSpeed = opts.rotateSpeed || 0.9;
  function resize() {
    const w = canvasHost.clientWidth || 1, h = canvasHost.clientHeight || 1;
    renderer.setSize(w, h, false); renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%';
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvasHost); resize();
  return { renderer, scene, camera, controls, resize };
}
