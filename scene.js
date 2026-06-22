import * as THREE from 'three/webgpu';
import {
  pass, mrt, output, normalView,
  uniform, vec2, vec3, vec4, float,
  screenUV, Fn, Loop, If,
  transformedNormalView
} from 'three/tsl';
import { ao } from 'three/examples/jsm/tsl/display/GTAONode.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Render layers
const LAYER_DEFAULT = 0;
const LAYER_SSR_EXCLUDE = 1; // Objects on this layer are excluded from SSR pass

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.FogExp2(0x000000, 0.045);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 8, 10);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const root = document.getElementById('root') ?? document.body;
root.appendChild(renderer.domElement);
await renderer.init();

// Orbit controls (disabled by default, toggle with F key)
const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enabled = false;
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.08;
orbitControls.target.set(0, 0.5, 0);
let freeOrbitMode = false;

// --- Post-processing setup ---
// Main scene pass renders everything
const scenePassNode = pass(scene, camera);
scenePassNode.setMRT(mrt({
  output: output,
  normal: transformedNormalView
}));

const scenePass = scenePassNode.getTextureNode('output');
const normalPass = scenePassNode.getTextureNode('normal');
const depthPass = scenePassNode.getTextureNode('depth');

// Reuse main scene pass depth/normals for SSR (avoids a full second render pass)
const ssrDepthClean = depthPass;
const ssrNormalClean = normalPass;

// GTAO (ambient occlusion)
const aoPass = ao(depthPass, normalPass, camera);
aoPass.resolutionScale = 0.4;
aoPass.thickness.value = 2;
aoPass.samples.value = 6;
aoPass.distanceExponent.value = 1.5;

const aoTexture = aoPass.getTextureNode().r;
// Remap AO so it never crushes to full black
const aoRemapped = aoTexture.mul(0.6).add(0.4);

// --- SSR Uniforms ---
const ssrEnabled = uniform(1.0);
const ssrStrength = uniform(0.35);
const ssrThickness = uniform(0.15);
const ssrMaxDist = uniform(1.0);
const ssrFresnelPow = uniform(1.5);
const ssrFade = uniform(0.9);
const projMatU = uniform(camera.projectionMatrix);
const projInvMatU = uniform(camera.projectionMatrixInverse);
const viewMatInvU = uniform(camera.matrixWorld);

// --- SSR TSL Node ---
const ssrNode = Fn(([colorIn, depthIn, normalIn]) => {
  const uv = screenUV;

  // Read depth and check for sky
  const rawDepth = depthIn.sample(uv).x;
  const isSky = rawDepth.greaterThanEqual(0.999);

  // Linearize depth
  const A = projMatU.element(2).element(2);
  const B = projMatU.element(3).element(2);
  const ndcZ = rawDepth.mul(2.0).sub(1.0);
  const linearZ = B.div(ndcZ.add(A));

  // Reconstruct view-space position
  const clipX = uv.x.mul(2.0).sub(1.0);
  const clipY = float(1.0).sub(uv.y).mul(2.0).sub(1.0);
  const viewX = clipX.mul(projInvMatU.element(0).element(0)).mul(linearZ);
  const viewY = clipY.mul(projInvMatU.element(1).element(1)).mul(linearZ);
  const viewPos = vec3(viewX, viewY, linearZ.negate());

  // Reflection direction
  const N = normalIn.sample(uv).xyz.normalize();
  const V = viewPos.normalize();
  const R = V.sub(N.mul(V.dot(N).mul(2.0))).normalize();

  // Fresnel
  const NdotV = N.dot(V.negate()).clamp(0.0, 1.0);
  const fresnel = float(1.0).sub(NdotV).pow(ssrFresnelPow).clamp(0.0, 1.0);

  const reflZ = R.z;

  // Mutable outputs
  const hitColor = vec3(0.0, 0.0, 0.0).toVar();
  const hitWeight = float(0.0).toVar();
  const hitT = float(0.0).toVar();
  const prevT = float(0.0).toVar();

  // Only reflect on upward-facing surfaces (table top) — skip walls, undersides, legs
  const normalUp = N.y.abs();
  const isUpwardFacing = normalUp.greaterThan(0.3);
  // Only trace if reflection ray goes upward or forward, not straight down into geometry
  const reflGoingUp = R.y.greaterThan(-0.5);

  // Reconstruct world-space position from view-space position using inverse view matrix
  // matrixWorld is column-major: element(col).element(row)
  const worldY = viewMatInvU.element(0).element(1).mul(viewPos.x)
    .add(viewMatInvU.element(1).element(1).mul(viewPos.y))
    .add(viewMatInvU.element(2).element(1).mul(viewPos.z))
    .add(viewMatInvU.element(3).element(1));

  // Table top is at y ≈ 0.1, floor is at y = -2. Only allow SSR above y = -0.8.
  const isAboveFloor = worldY.greaterThan(-0.8);

  // Also check linear depth — floor is much further from camera than the table
  // Camera at y=8, table at y=0 (~10 units away), floor at y=-2 (~13+ units)
  const isCloseEnough = linearZ.lessThan(14.0);

  If(ssrEnabled.greaterThan(0.5).and(reflZ.lessThan(0.1)).and(isSky.not()).and(isUpwardFacing).and(reflGoingUp).and(isAboveFloor).and(isCloseEnough), () => {
    // Ray march - 16 linear steps (reduced from 32 for performance)
    Loop(16, ({ i }) => {
      const fi = float(i).add(1.0);
      const t = fi.div(16.0).mul(ssrMaxDist);
      const samplePos = viewPos.add(R.mul(t));

      const negZ = samplePos.z.negate();
      const sClipX = samplePos.x.mul(projMatU.element(0).element(0)).div(negZ);
      const sClipY = samplePos.y.mul(projMatU.element(1).element(1)).div(negZ);
      const sUV = vec2(
        sClipX.mul(0.5).add(0.5),
        float(1.0).sub(sClipY.mul(0.5).add(0.5))
      );

      const inBounds = sUV.x.greaterThanEqual(0.0).and(sUV.x.lessThanEqual(1.0))
        .and(sUV.y.greaterThanEqual(0.0)).and(sUV.y.lessThanEqual(1.0));

      If(inBounds.and(hitWeight.lessThan(0.5)), () => {
        const sampledDepth = depthIn.sample(sUV).x;
        const sampledNdcZ = sampledDepth.mul(2.0).sub(1.0);
        const sampledLinZ = B.div(sampledNdcZ.add(A));

        const diff = negZ.sub(sampledLinZ);
        const isHit = diff.greaterThan(0.0).and(diff.lessThan(ssrThickness));
        const notSky = sampledDepth.lessThan(0.999);

        If(isHit.and(notSky), () => {
          hitT.assign(t);
          hitWeight.assign(1.0);
        });
      });

      If(hitWeight.lessThan(0.5), () => {
        prevT.assign(t);
      });
    });

    // Binary refinement - 4 steps (reduced from 8 for performance)
    If(hitWeight.greaterThan(0.5), () => {
      const loT = prevT.toVar();
      const hiT = hitT.toVar();

      Loop(4, () => {
        const midT = loT.add(hiT).mul(0.5);
        const midPos = viewPos.add(R.mul(midT));

        const midNegZ = midPos.z.negate();
        const midClipX = midPos.x.mul(projMatU.element(0).element(0)).div(midNegZ);
        const midClipY = midPos.y.mul(projMatU.element(1).element(1)).div(midNegZ);
        const midUV = vec2(
          midClipX.mul(0.5).add(0.5),
          float(1.0).sub(midClipY.mul(0.5).add(0.5))
        );

        const midDepth = depthIn.sample(midUV).x;
        const midNdcZ = midDepth.mul(2.0).sub(1.0);
        const midLinZ = B.div(midNdcZ.add(A));
        const midDiff = midNegZ.sub(midLinZ);

        If(midDiff.greaterThan(0.0), () => {
          hiT.assign(midT);
        }).Else(() => {
          loT.assign(midT);
        });
      });

      // Sample color at refined hit
      const finalT = loT.add(hiT).mul(0.5);
      const finalPos = viewPos.add(R.mul(finalT));

      const finalNegZ = finalPos.z.negate();
      const finalClipX = finalPos.x.mul(projMatU.element(0).element(0)).div(finalNegZ);
      const finalClipY = finalPos.y.mul(projMatU.element(1).element(1)).div(finalNegZ);
      const finalUV = vec2(
        finalClipX.mul(0.5).add(0.5),
        float(1.0).sub(finalClipY.mul(0.5).add(0.5))
      );

      // Edge fade
      const edgeX = finalUV.x.mul(float(1.0).sub(finalUV.x)).mul(4.0).clamp(0.0, 1.0);
      const edgeY = finalUV.y.mul(float(1.0).sub(finalUV.y)).mul(4.0).clamp(0.0, 1.0);
      const edgeFade = edgeX.mul(edgeY);

      // Distance fade
      const distFade = float(1.0).sub(finalT.div(ssrMaxDist)).clamp(0.0, 1.0);

      const sampledColor = colorIn.sample(finalUV).xyz;
      hitColor.assign(sampledColor.mul(edgeFade).mul(distFade));
    });
  });

  const reflectionMix = hitWeight.mul(fresnel).mul(ssrStrength).mul(ssrFade);
  return vec4(hitColor, reflectionMix);
});

// --- Composite: scene + SSR, then multiply by AO ---
const ssrResult = ssrNode(scenePass, ssrDepthClean, ssrNormalClean);
const sceneWithSSR = scenePass.add(vec4(ssrResult.xyz.mul(ssrResult.w), 0.0));
const compositedScene = sceneWithSSR.mul(aoRemapped);

const PostProcessingClass = THREE.PostProcessing || THREE.RenderPipeline;
const postProcessing = new PostProcessingClass(renderer);
postProcessing.outputNode = compositedScene;
postProcessing.needsUpdate = true;

// Load HDR environment map from Polyhaven
const hdrLoader = new HDRLoader();
hdrLoader.load('https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr', (hdrTexture) => {
  hdrTexture.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = hdrTexture;
  scene.environmentIntensity = 0.6;
  // Keep the dark background — only use HDR for subtle reflections only
});

// Lighting
const ambientLight = new THREE.AmbientLight(0x102040, 0.4);
ambientLight.name = 'ambientLight1';
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xaaddff, 1.2);
dirLight.name = 'directionalLight1';
dirLight.position.set(4, 14, 6);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(1024, 1024);
dirLight.shadow.camera.left = -10;
dirLight.shadow.camera.right = 10;
dirLight.shadow.camera.top = 10;
dirLight.shadow.camera.bottom = -10;
dirLight.shadow.bias = -0.001;
dirLight.shadow.normalBias = 0.02;
scene.add(dirLight);

const pointLight1 = new THREE.PointLight(0x0088ff, 0.6, 22);
pointLight1.name = 'pointLight1';
pointLight1.position.set(-5, 5, 0);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0xff2255, 0.6, 22);
pointLight2.name = 'pointLight2';
pointLight2.position.set(5, 5, 0);
scene.add(pointLight2);

// Table dimensions
const TABLE_WIDTH = 5;
const TABLE_LENGTH = 9;
const TABLE_HEIGHT = 0.18;
const TABLE_Y = 0;
const NET_HEIGHT = 0.0; // air hockey has no net — goal slot only
const PADDLE_WIDTH = 0.9;
const PADDLE_HEIGHT = 0.9;
const PADDLE_DEPTH = 0.08;
const PADDLE_HANDLE_LENGTH = 0.0; // mallets have no handle
const BALL_RADIUS = 0.18; // puck is larger and flatter

// Table — air hockey rink in deep teal/slate
const tableGeo = new THREE.BoxGeometry(TABLE_WIDTH, TABLE_HEIGHT, TABLE_LENGTH);
const tableMat = new THREE.MeshPhysicalMaterial({
  color: new THREE.Color(0x0a2233), roughness: 0.05, metalness: 0.2,
  clearcoat: 0.9, clearcoatRoughness: 0.05,
  envMapIntensity: 0.8, reflectivity: 0.8
});
const table = new THREE.Mesh(tableGeo, tableMat);
table.name = 'table';
table.position.y = TABLE_Y;
table.receiveShadow = true;
scene.add(table);

// Table edge lines — glowing cyan perimeter rails
const edgeLineMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
const edgeLineY = TABLE_Y + TABLE_HEIGHT / 2 + 0.003;
const edgeLineThick = 0.04;

const edgeLineFront = new THREE.Mesh(new THREE.BoxGeometry(TABLE_WIDTH, 0.005, edgeLineThick), edgeLineMat);
edgeLineFront.name = 'edgeLineFront';
edgeLineFront.position.set(0, edgeLineY, TABLE_LENGTH / 2 - edgeLineThick / 2);
scene.add(edgeLineFront);

const edgeLineBack = new THREE.Mesh(new THREE.BoxGeometry(TABLE_WIDTH, 0.005, edgeLineThick), edgeLineMat);
edgeLineBack.name = 'edgeLineBack';
edgeLineBack.position.set(0, edgeLineY, -TABLE_LENGTH / 2 + edgeLineThick / 2);
scene.add(edgeLineBack);

const edgeLineLeft = new THREE.Mesh(new THREE.BoxGeometry(edgeLineThick, 0.005, TABLE_LENGTH), edgeLineMat);
edgeLineLeft.name = 'edgeLineLeft';
edgeLineLeft.position.set(-TABLE_WIDTH / 2 + edgeLineThick / 2, edgeLineY, 0);
scene.add(edgeLineLeft);

const edgeLineRight = new THREE.Mesh(new THREE.BoxGeometry(edgeLineThick, 0.005, TABLE_LENGTH), edgeLineMat);
edgeLineRight.name = 'edgeLineRight';
edgeLineRight.position.set(TABLE_WIDTH / 2 - edgeLineThick / 2, edgeLineY, 0);
scene.add(edgeLineRight);

// Table lines — center divider + center circle arc
const lineMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
const centerLine = new THREE.Mesh(new THREE.BoxGeometry(TABLE_WIDTH, 0.005, 0.025), lineMat);
centerLine.name = 'centerLine';
centerLine.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + 0.003, 0);
scene.add(centerLine);

// Center dot (goal circle)
const centerDotGeo = new THREE.RingGeometry(0.38, 0.42, 32);
const centerDot = new THREE.Mesh(centerDotGeo, new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide }));
centerDot.name = 'centerDot';
centerDot.rotation.x = -Math.PI / 2;
centerDot.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + 0.004, 0);
scene.add(centerDot);

// Goal arcs each end
for (let side of [-1, 1]) {
  const arcGeo = new THREE.RingGeometry(0.7, 0.75, 32, 1, 0, Math.PI);
  const arc = new THREE.Mesh(arcGeo, new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide }));
  arc.name = `goalArc${side}`;
  arc.rotation.x = -Math.PI / 2;
  arc.rotation.z = side > 0 ? Math.PI : 0;
  arc.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + 0.004, side * (TABLE_LENGTH / 2 - 1.1));
  scene.add(arc);
}

// Dummy halfLine kept for retro/zen compatibility (set invisible)
const halfLine = new THREE.Mesh(new THREE.BoxGeometry(0.005, 0.005, 0.005), lineMat);
halfLine.name = 'halfLine';
halfLine.visible = false;
halfLine.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + 0.003, 0);
scene.add(halfLine);

// Goal slots (cutouts simulated with dark rectangles at each end)
const goalWidth = 1.4;
const goalMat = new THREE.MeshBasicMaterial({ color: 0x000811 });
for (let side of [-1, 1]) {
  const goalGeo = new THREE.BoxGeometry(goalWidth, TABLE_HEIGHT + 0.02, 0.05);
  const goal = new THREE.Mesh(goalGeo, goalMat);
  goal.name = `goal${side}`;
  goal.position.set(0, TABLE_Y, side * TABLE_LENGTH / 2);
  scene.add(goal);
}

// Air hockey rink bumper rails — thick side walls instead of a net
const railHeight = 0.22;
const railThick = 0.18;
const railMat = new THREE.MeshStandardMaterial({ color: 0x112233, metalness: 0.7, roughness: 0.25, envMapIntensity: 0.8 });
const railY = TABLE_Y + TABLE_HEIGHT / 2 + railHeight / 2;

const railLeft = new THREE.Mesh(new THREE.BoxGeometry(railThick, railHeight, TABLE_LENGTH), railMat);
railLeft.name = 'railLeft';
railLeft.position.set(-TABLE_WIDTH / 2 - railThick / 2, railY, 0);
railLeft.castShadow = true;
scene.add(railLeft);

const railRight = new THREE.Mesh(new THREE.BoxGeometry(railThick, railHeight, TABLE_LENGTH), railMat);
railRight.name = 'railRight';
railRight.position.set(TABLE_WIDTH / 2 + railThick / 2, railY, 0);
railRight.castShadow = true;
scene.add(railRight);

// End rails with goal openings
const endRailSideW = (TABLE_WIDTH - 1.4) / 2;
for (let side of [-1, 1]) {
  for (let lr of [-1, 1]) {
    const endRail = new THREE.Mesh(new THREE.BoxGeometry(endRailSideW, railHeight, railThick), railMat);
    endRail.name = `endRail${side}${lr}`;
    endRail.position.set(lr * (1.4 / 2 + endRailSideW / 2), railY, side * (TABLE_LENGTH / 2 + railThick / 2));
    endRail.castShadow = true;
    scene.add(endRail);
  }
}

// Glowing cyan edge strips on rails
const railStripMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
const railStripLeft = new THREE.Mesh(new THREE.BoxGeometry(0.015, railHeight * 0.8, TABLE_LENGTH), railStripMat);
railStripLeft.name = 'railStripLeft';
railStripLeft.position.set(-TABLE_WIDTH / 2 - railThick + 0.01, railY, 0);
scene.add(railStripLeft);

const railStripRight = new THREE.Mesh(new THREE.BoxGeometry(0.015, railHeight * 0.8, TABLE_LENGTH), railStripMat);
railStripRight.name = 'railStripRight';
railStripRight.position.set(TABLE_WIDTH / 2 + railThick - 0.01, railY, 0);
scene.add(railStripRight);

// Dummy variables to keep retro/zen line-color code from crashing
const postMat = new THREE.MeshStandardMaterial({ color: 0x334455, metalness: 0.7, roughness: 0.3 });
const clampMat = new THREE.MeshStandardMaterial({ color: 0x223344, metalness: 0.8, roughness: 0.3 });
const net = new THREE.Mesh(new THREE.BoxGeometry(0.001, 0.001, 0.001), new THREE.MeshBasicMaterial());
net.name = 'net'; net.visible = false; scene.add(net);

// Air hockey table legs — wide rectangular cabinet style
const legGeo = new THREE.BoxGeometry(0.22, 2.0, 0.22);
const legMat = new THREE.MeshStandardMaterial({ color: 0x0d1a26, metalness: 0.5, roughness: 0.5, envMapIntensity: 0.5 });
const legPositions = [
  [-TABLE_WIDTH / 2 + 0.2, -1, -TABLE_LENGTH / 2 + 0.3],
  [TABLE_WIDTH / 2 - 0.2, -1, -TABLE_LENGTH / 2 + 0.3],
  [-TABLE_WIDTH / 2 + 0.2, -1, TABLE_LENGTH / 2 - 0.3],
  [TABLE_WIDTH / 2 - 0.2, -1, TABLE_LENGTH / 2 - 0.3]
];
legPositions.forEach((pos, i) => {
  const leg = new THREE.Mesh(legGeo, legMat);
  leg.name = `tableLeg${i}`;
  leg.position.set(...pos);
  leg.castShadow = true;
  scene.add(leg);

  // Cyan accent band at top of each leg
  const bandGeo = new THREE.BoxGeometry(0.24, 0.04, 0.24);
  const band = new THREE.Mesh(bandGeo, new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
  band.name = `legBand${i}`;
  band.position.set(pos[0], TABLE_Y - TABLE_HEIGHT / 2 - 0.02, pos[2]);
  scene.add(band);
});

// Floor — dark arena floor
const floorGeo = new THREE.PlaneGeometry(80, 80);
const floorMat = new THREE.MeshStandardMaterial({ color: 0x020a12, roughness: 0.85 });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.name = 'floor';
floor.rotation.x = -Math.PI / 2;
floor.position.y = -2;
floor.receiveShadow = true;
scene.add(floor);

// Air hockey mallets — flat disc with a central knob handle on top
function createMallet(topColor, name) {
  const group = new THREE.Group();
  group.name = name;

  // Main disc body
  const discGeo = new THREE.CylinderGeometry(PADDLE_WIDTH / 2, PADDLE_WIDTH / 2 * 1.05, PADDLE_DEPTH * 1.5, 32);
  const discMat = new THREE.MeshStandardMaterial({ color: topColor, roughness: 0.35, metalness: 0.5, envMapIntensity: 0.8 });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.name = name + 'Head';
  disc.castShadow = true;
  group.add(disc);

  // Glowing ring around edge
  const ringGeo = new THREE.TorusGeometry(PADDLE_WIDTH / 2 * 0.98, 0.018, 8, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.name = name + 'Ring';
  ring.rotation.x = Math.PI / 2;
  ring.position.y = PADDLE_DEPTH * 0.3;
  group.add(ring);

  // Center knob
  const knobGeo = new THREE.CylinderGeometry(0.09, 0.11, 0.22, 16);
  const knobMat = new THREE.MeshStandardMaterial({ color: 0x1a2a3a, roughness: 0.3, metalness: 0.7 });
  const knob = new THREE.Mesh(knobGeo, knobMat);
  knob.name = name + 'Knob';
  knob.position.y = PADDLE_DEPTH * 0.75 + 0.11;
  group.add(knob);

  // Bottom rubber pad — slightly larger, dark
  const rubberGeo = new THREE.CylinderGeometry(PADDLE_WIDTH / 2 * 1.02, PADDLE_WIDTH / 2 * 1.04, 0.025, 32);
  const rubberMat = new THREE.MeshStandardMaterial({ color: 0x0a1520, roughness: 0.9, metalness: 0.0 });
  const rubber = new THREE.Mesh(rubberGeo, rubberMat);
  rubber.name = name + 'RubberFront';
  rubber.position.y = -PADDLE_DEPTH * 0.65;
  group.add(rubber);

  // Dummy back (for theme compatibility)
  const dummyMat = new THREE.MeshStandardMaterial({ color: 0x0a1520 });
  const dummy = new THREE.Mesh(new THREE.BoxGeometry(0.001,0.001,0.001), dummyMat);
  dummy.name = name + 'RubberBack';
  dummy.visible = false;
  group.add(dummy);

  return group;
}

const playerPaddle = createMallet(0x1a88ff, 'playerPaddle');
playerPaddle.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + PADDLE_DEPTH * 0.75, TABLE_LENGTH / 2 - 0.6);
scene.add(playerPaddle);

const aiPaddle = createMallet(0xff3355, 'aiPaddle');
aiPaddle.position.set(0, TABLE_Y + TABLE_HEIGHT / 2 + PADDLE_DEPTH * 0.75, -TABLE_LENGTH / 2 + 0.6);
scene.add(aiPaddle);

// Puck — flat cylinder instead of a sphere
const ballGeo = new THREE.CylinderGeometry(BALL_RADIUS, BALL_RADIUS, 0.08, 32);
const ballMat = new THREE.MeshStandardMaterial({ color: 0xe8eef5, roughness: 0.2, metalness: 0.8, emissive: 0x335577, emissiveIntensity: 0.18, envMapIntensity: 1.2 });
const ball = new THREE.Mesh(ballGeo, ballMat);
ball.name = 'ball';
ball.castShadow = true;
scene.add(ball);

// Puck trail — flat disc ghosts
const trailCount = 20;
const trailPositions = [];
const trailGeo2 = new THREE.CylinderGeometry(BALL_RADIUS * 0.6, BALL_RADIUS * 0.6, 0.06, 16);
const trailMeshes = [];
for (let i = 0; i < trailCount; i++) {
  const mat = new THREE.MeshBasicMaterial({ color: 0x00ccff, transparent: true, opacity: (1 - i / trailCount) * 0.28 });
  const m = new THREE.Mesh(trailGeo2, mat);
  m.name = `trail${i}`;
  m.visible = false;
  scene.add(m);
  trailMeshes.push(m);
  trailPositions.push(new THREE.Vector3());
}

// Game state — air hockey: puck stays flat, no gravity bounce
const GRAVITY = -0.5; // near-zero: puck floats on air cushion
const BOUNCE_DAMPING = 0.96; // puck barely loses speed on rail bounce
const BALL_SPEED_BASE = 8; // faster than ping pong

// Tournament config
const SETS_TO_WIN = 2; // Best of 3 games
const POINTS_TO_WIN_SET = 7; // First to 7 goals per game

let gameState = {
  ballPos: new THREE.Vector3(),
  ballVel: new THREE.Vector3(),
  playerScore: 0,
  aiScore: 0,
  playerSets: 0,
  aiSets: 0,
  currentSet: 1,
  setHistory: [], // [{player: X, ai: Y}, ...]
  serving: true,
  serverIsPlayer: true,
  rallying: false,
  lastHit: 'none', // 'player', 'ai', 'none'
  bouncedOnOpponentSide: false,
  bouncedOnServerSide: false,
  playerSideBounces: 0,
  aiSideBounces: 0,
  gameOver: false,
  matchOver: false,
  paused: true,
  countdown: 0,
  waitingForPlayerServe: false
};

// Stats tracking
let stats = {
  currentRallyTouches: 0,
  longestRally: 0,
  totalRallies: 0,
  totalTouches: 0,
  topBallSpeed: 0,
  playerStreak: 0,
  aiStreak: 0,
  bestStreak: 0,
  playerAces: 0,
  aiAces: 0,
  playerWins: 0,
  aiWins: 0,
};

function resetStats() {
  stats.currentRallyTouches = 0;
  stats.longestRally = 0;
  stats.totalRallies = 0;
  stats.totalTouches = 0;
  stats.topBallSpeed = 0;
  stats.playerStreak = 0;
  stats.aiStreak = 0;
  stats.bestStreak = 0;
  stats.playerAces = 0;
  stats.aiAces = 0;
  stats.playerWins = 0;
  stats.aiWins = 0;
}

function resetMatch() {
  gameState.playerScore = 0;
  gameState.aiScore = 0;
  gameState.playerSets = 0;
  gameState.aiSets = 0;
  gameState.currentSet = 1;
  gameState.setHistory = [];
  gameState.gameOver = false;
  gameState.matchOver = false;
  gameState.serverIsPlayer = true;
  resetStats();
  updateScoreDisplay();
  updateSetDisplay();
  updateStatsDisplay();
}

function resetBall(serverIsPlayer) {
  const side = serverIsPlayer ? 1 : -1;
  // Puck drops flat onto center of server's half
  const puckY = TABLE_Y + TABLE_HEIGHT / 2 + 0.04;
  gameState.ballPos.set((Math.random() - 0.5) * 0.5, puckY, side * (TABLE_LENGTH / 2 - 1.8));
  const currentBallSpeed = window._ballSpeedBase ?? BALL_SPEED_BASE;
  // Puck slides flat — no vertical component
  gameState.ballVel.set(
    (Math.random() - 0.5) * 1.5,
    0,
    -side * currentBallSpeed * 0.75
  );
  gameState.serving = true;
  gameState.rallying = false;
  gameState.lastHit = serverIsPlayer ? 'player' : 'ai';
  gameState.bouncedOnOpponentSide = false;
  gameState.bouncedOnServerSide = false;
  gameState.playerSideBounces = 0;
  gameState.aiSideBounces = 0;

  trailPositions.forEach(p => p.set(0, -10, 0));
  trailMeshes.forEach(m => m.visible = false);
}

function serve() {
  gameState.paused = false;
  resetBall(gameState.serverIsPlayer);
}

// Input
const keys = { a: false, d: false, w: false, s: false, left: false, right: false, space: false, f: false };
let mouseX = 0;
let mouseY = 0;
let useMouseControl = true;
let scrollDelta = 0;

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (key === 'a' || key === 'arrowleft') { keys.a = true; useMouseControl = false; }
  if (key === 'd' || key === 'arrowright') { keys.d = true; useMouseControl = false; }
  if (key === 'w' || key === 'arrowup') { keys.w = true; }
  if (key === 's' || key === 'arrowdown') { keys.s = true; }
  if (key === 'f') {
    freeOrbitMode = !freeOrbitMode;
    orbitControls.enabled = freeOrbitMode;
    if (freeOrbitMode) {
      orbitControls.target.set(0, 0.5, 0);
      showMessage('Free orbit on — press F to exit', 2000);
    } else {
      // Reset camera to default position
      camera.position.set(0, 8, 10);
      camera.lookAt(0, 0.5, 0);
      showMessage('Free orbit off', 1000);
    }
  }
  if (key === ' ') {
    keys.space = true;
    if (gameState.waitingForPlayerServe) {
      gameState.waitingForPlayerServe = false;
      serve();
    } else if (gameState.matchOver) {
      resetMatch();
      serve();
    } else if (gameState.gameOver) {
      // Set over, start next set
      startNextSet();
    } else if (gameState.paused && gameState.serverIsPlayer) {
      serve();
    }
  }
  if (key === ' ') keys.space = true;
});

window.addEventListener('keyup', (e) => {
  const key = e.key.toLowerCase();
  if (key === 'a' || key === 'arrowleft') keys.a = false;
  if (key === 'd' || key === 'arrowright') keys.d = false;
  if (key === 'w' || key === 'arrowup') keys.w = false;
  if (key === 's' || key === 'arrowdown') keys.s = false;
  if (key === ' ') keys.space = false;
});

window.addEventListener('mousemove', (e) => {
  mouseX = (e.clientX / window.innerWidth) * 2 - 1;
  mouseY = (e.clientY / window.innerHeight) * 2 - 1; // -1 top, 1 bottom
  useMouseControl = true;
});

window.addEventListener('mousedown', (e) => {
  if (gameState.waitingForPlayerServe) {
    gameState.waitingForPlayerServe = false;
    serve();
    return;
  }
  if (gameState.matchOver) {
    resetMatch();
    serve();
    return;
  }
  if (gameState.gameOver) {
    startNextSet();
    return;
  }
  // Don't allow mouse click to serve when it's AI's turn
  if (gameState.paused && gameState.serverIsPlayer) {
    serve();
    return;
  }
});

// Touch support
window.addEventListener('contextmenu', (e) => e.preventDefault());

window.addEventListener('wheel', (e) => {
  scrollDelta -= e.deltaY * 0.005;
}, { passive: true });

window.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  mouseX = (touch.clientX / window.innerWidth) * 2 - 1;
  mouseY = (touch.clientY / window.innerHeight) * 2 - 1;
  useMouseControl = true;
}, { passive: false });

window.addEventListener('touchstart', (e) => {
  if (gameState.waitingForPlayerServe) {
    gameState.waitingForPlayerServe = false;
    serve();
    return;
  }
  if (gameState.matchOver) {
    resetMatch();
    serve();
    return;
  }
  if (gameState.gameOver) {
    startNextSet();
    return;
  }
  // Don't allow touch to serve when it's AI's turn
  if (gameState.paused && gameState.serverIsPlayer) {
    serve();
    return;
  }
});

// UI
const uiContainer = document.createElement('div');
uiContainer.style.cssText = `
  position: fixed; top: 0; left: 0; width: 100%; pointer-events: none;
  font-family: 'Inter', sans-serif; z-index: 10; box-sizing: border-box;
`;
document.body.appendChild(uiContainer);

// Load Inter font
const fontLink = document.createElement('link');
fontLink.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Instrument+Serif:ital@0;1&display=swap';
fontLink.rel = 'stylesheet';
document.head.appendChild(fontLink);

// Settings panel
const settingsBtn = document.createElement('div');
settingsBtn.style.cssText = `
  position: fixed; top: 16px; right: 16px; width: 36px; height: 36px;
  border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; pointer-events: all; z-index: 100;
  background: rgba(10,10,18,0.8); color: #888; font-size: 18px;
  font-family: 'Inter', sans-serif; transition: border-color 0.2s, color 0.2s;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
`;
settingsBtn.textContent = '⚙';
settingsBtn.style.display = 'none';
settingsBtn.addEventListener('mouseenter', () => { settingsBtn.style.borderColor = 'rgba(255,255,255,0.35)'; settingsBtn.style.color = '#ccc'; });
settingsBtn.addEventListener('mouseleave', () => { settingsBtn.style.borderColor = 'rgba(255,255,255,0.15)'; settingsBtn.style.color = '#888'; });
document.body.appendChild(settingsBtn);

const settingsPanel = document.createElement('div');
settingsPanel.style.cssText = `
  position: fixed; top: 60px; right: 16px; width: 260px;
  background: rgba(10,10,18,0.92); border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px; padding: 16px; z-index: 100; pointer-events: all;
  font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 12px;
  display: none; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  max-height: calc(100vh - 80px); overflow-y: auto;
`;
document.body.appendChild(settingsPanel);

let settingsOpen = false;
function toggleSettings() {
  settingsOpen = !settingsOpen;
  settingsBtn.style.display = settingsOpen ? 'flex' : 'none';
  settingsPanel.style.display = settingsOpen ? 'block' : 'none';
  settingsBtn.style.background = settingsOpen ? 'rgba(255,255,255,0.08)' : 'rgba(10,10,18,0.8)';
}
settingsBtn.addEventListener('click', toggleSettings);
window.addEventListener('keydown', (e) => {
  if (e.key === 'p' || e.key === 'P') toggleSettings();
});

function createSection(title) {
  const sec = document.createElement('div');
  sec.style.cssText = 'margin-bottom: 16px;';
  const label = document.createElement('div');
  label.style.cssText = "color: rgba(255,255,255,0.3); font-family: 'Instrument Serif', serif; font-style: italic; font-size: 13px; font-weight: 400; letter-spacing: 0.5px; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;";
  label.textContent = title;
  sec.appendChild(label);
  settingsPanel.appendChild(sec);
  return sec;
}

function createSlider(parent, label, min, max, step, value, onChange) {
  const row = document.createElement('div');
  row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;';
  const lbl = document.createElement('span');
  lbl.style.cssText = 'color: #aaa; font-size: 11px;';
  lbl.textContent = label;
  const right = document.createElement('div');
  right.style.cssText = 'display: flex; align-items: center; gap: 8px;';
  const val = document.createElement('span');
  val.style.cssText = 'color: #fff; font-size: 11px; min-width: 32px; text-align: right; font-variant-numeric: tabular-nums;';
  val.textContent = value;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = min; input.max = max; input.step = step; input.value = value;
  input.style.cssText = `
    width: 90px; height: 4px; -webkit-appearance: none; appearance: none;
    background: rgba(255,255,255,0.1); border-radius: 2px; outline: none; cursor: pointer;
  `;
  const style = document.createElement('style');
  style.textContent = `
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none; width: 12px; height: 12px;
      border-radius: 50%; background: #fff; border: none; cursor: pointer;
    }
    input[type=range]::-moz-range-thumb {
      width: 12px; height: 12px; border-radius: 50%; background: #fff; border: none; cursor: pointer;
    }
  `;
  if (!document.querySelector('#settings-slider-style')) { style.id = 'settings-slider-style'; document.head.appendChild(style); }
  input.addEventListener('input', () => {
    val.textContent = parseFloat(input.value).toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0);
    onChange(parseFloat(input.value));
  });
  right.appendChild(input);
  right.appendChild(val);
  row.appendChild(lbl);
  row.appendChild(right);
  parent.appendChild(row);
  return input;
}

function createToggle(parent, label, value, onChange) {
  const row = document.createElement('div');
  row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;';
  const lbl = document.createElement('span');
  lbl.style.cssText = 'color: #aaa; font-size: 11px;';
  lbl.textContent = label;
  const toggle = document.createElement('div');
  toggle.style.cssText = `
    width: 34px; height: 18px; border-radius: 9px; cursor: pointer; transition: background 0.2s;
    background: ${value ? 'rgba(34,153,255,0.6)' : 'rgba(255,255,255,0.1)'};
    position: relative;
  `;
  const knob = document.createElement('div');
  knob.style.cssText = `
    width: 14px; height: 14px; border-radius: 50%; background: #fff; position: absolute;
    top: 2px; transition: left 0.2s;
    left: ${value ? '18px' : '2px'};
  `;
  toggle.appendChild(knob);
  let state = value;
  toggle.addEventListener('click', () => {
    state = !state;
    toggle.style.background = state ? 'rgba(34,153,255,0.6)' : 'rgba(255,255,255,0.1)';
    knob.style.left = state ? '18px' : '2px';
    onChange(state);
  });
  row.appendChild(lbl);
  row.appendChild(toggle);
  parent.appendChild(row);
  return { setState: (v) => { state = v; toggle.style.background = v ? 'rgba(34,153,255,0.6)' : 'rgba(255,255,255,0.1)'; knob.style.left = v ? '18px' : '2px'; } };
}

function createColorPicker(parent, label, value, onChange) {
  const row = document.createElement('div');
  row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;';
  const lbl = document.createElement('span');
  lbl.style.cssText = 'color: #aaa; font-size: 11px;';
  lbl.textContent = label;
  const input = document.createElement('input');
  input.type = 'color';
  input.value = value;
  input.style.cssText = `
    width: 28px; height: 22px; border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px; background: none; cursor: pointer; padding: 0;
  `;
  input.addEventListener('input', () => onChange(input.value));
  row.appendChild(lbl);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

// --- Lighting section ---
const lightSec = createSection('Lighting');
createSlider(lightSec, 'Ambient', 0, 2, 0.05, 0.5, (v) => { ambientLight.intensity = v; });
createSlider(lightSec, 'Directional', 0, 3, 0.05, 1.0, (v) => { dirLight.intensity = v; });
createSlider(lightSec, 'Point Light 1', 0, 2, 0.05, 0.5, (v) => { pointLight1.intensity = v; });
createSlider(lightSec, 'Point Light 2', 0, 2, 0.05, 0.5, (v) => { pointLight2.intensity = v; });
createColorPicker(lightSec, 'Point 1 Color', '#4488ff', (v) => { pointLight1.color.set(v); });
createColorPicker(lightSec, 'Point 2 Color', '#ff4488', (v) => { pointLight2.color.set(v); });

// --- Environment section ---
const envSec = createSection('Environment');
createColorPicker(envSec, 'Background', '#000000', (v) => {
  scene.background.set(v);
  scene.fog.color.set(v);
});
createSlider(envSec, 'Fog Density', 0, 0.15, 0.005, 0.045, (v) => { scene.fog.density = v; });
createColorPicker(envSec, 'Floor Color', '#000000', (v) => { floorMat.color.set(v); });

// --- HDR / IBL section ---
const hdrSec = createSection('HDR / IBL');
createSlider(hdrSec, 'Env Intensity', 0, 5, 0.1, 0.6, (v) => { scene.environmentIntensity = v; });
createSlider(hdrSec, 'Env Rotation', 0, 360, 5, 0, (v) => {
  scene.environmentRotation.y = THREE.MathUtils.degToRad(v);
});
createToggle(hdrSec, 'Show as Background', false, (v) => {
  if (v && scene.environment) {
    scene.background = scene.environment;
  } else {
    scene.background = new THREE.Color(0x000000);
  }
});
createSlider(hdrSec, 'Table Env Map', 0, 3, 0.1, 0.8, (v) => { tableMat.envMapIntensity = v; tableMat.needsUpdate = true; });
createSlider(hdrSec, 'Ball Env Map', 0, 3, 0.1, 1.5, (v) => { ballMat.envMapIntensity = v; });
createSlider(hdrSec, 'Legs Env Map', 0, 3, 0.1, 1.2, (v) => { legMat.envMapIntensity = v; });
createSlider(hdrSec, 'Posts Env Map', 0, 3, 0.1, 1.5, (v) => { postMat.envMapIntensity = v; });

// --- Table section ---
const tableSec = createSection('Table');
createColorPicker(tableSec, 'Rink Color', '#0a2233', (v) => { tableMat.color.set(v); });
createSlider(tableSec, 'Table Roughness', 0, 1, 0.05, 1.0, (v) => { tableMat.roughness = v; });
createSlider(tableSec, 'Table Metalness', 0, 1, 0.05, 0.0, (v) => { tableMat.metalness = v; });
createSlider(tableSec, 'Clearcoat', 0, 1, 0.05, 0.3, (v) => { tableMat.clearcoat = v; });
createSlider(tableSec, 'Clearcoat Rough', 0, 1, 0.05, 0.1, (v) => { tableMat.clearcoatRoughness = v; });

// --- Ball section ---
const ballSec = createSection('Ball');
createColorPicker(ballSec, 'Puck Color', '#e8eef5', (v) => {
  ballMat.color.set(v);
  ballMat.emissive.set(v);
  trailMeshes.forEach(m => m.material.color.set(v));
});
createSlider(ballSec, 'Ball Roughness', 0, 1, 0.05, 0.7, (v) => { ballMat.roughness = v; });
createSlider(ballSec, 'Ball Metalness', 0, 1, 0.05, 0.15, (v) => { ballMat.metalness = v; });
createSlider(ballSec, 'Glow Intensity', 0, 1, 0.05, 0.3, (v) => { ballMat.emissiveIntensity = v; });
const trailToggle = createToggle(ballSec, 'Ball Trail', true, (v) => {
  trailMeshes.forEach(m => { if (!v) m.visible = false; });
  window._trailEnabled = v;
});
window._trailEnabled = true;

// --- Shadows section ---
const shadowSec = createSection('Shadows');
createToggle(shadowSec, 'Enabled', true, (v) => {
  renderer.shadowMap.enabled = v;
  dirLight.castShadow = v;
  ball.castShadow = v;
  table.receiveShadow = v;
  floor.receiveShadow = v;
});

// --- Ambient Occlusion section ---
const aoSec = createSection('Ambient Occlusion');
createSlider(aoSec, 'Thickness', 0.1, 5, 0.1, 2, (v) => { aoPass.thickness.value = v; });
createSlider(aoSec, 'Samples', 4, 32, 4, 16, (v) => { aoPass.samples.value = v; });
createSlider(aoSec, 'Distance Exp', 0.5, 4, 0.1, 1.5, (v) => { aoPass.distanceExponent.value = v; });

// --- Screen Space Reflections section ---
const ssrSec = createSection('Screen Space Reflections');
createToggle(ssrSec, 'Enabled', true, (v) => { ssrEnabled.value = v ? 1.0 : 0.0; });
createSlider(ssrSec, 'Strength', 0, 1, 0.05, 0.35, (v) => { ssrStrength.value = v; });
createSlider(ssrSec, 'Thickness', 0.01, 0.5, 0.01, 0.15, (v) => { ssrThickness.value = v; });
createSlider(ssrSec, 'Max Distance', 0.5, 8, 0.5, 1.0, (v) => { ssrMaxDist.value = v; });
createSlider(ssrSec, 'Fresnel Power', 0.5, 5, 0.1, 1.5, (v) => { ssrFresnelPow.value = v; });
createSlider(ssrSec, 'Fade', 0, 1, 0.05, 0.9, (v) => { ssrFade.value = v; });

// Theme definitions — each theme sets player mallet, AI mallet, and rink together
const themes = [
  {
    name: 'Arctic',
    player: { head: '#1a88ff', rubber: '#0d5fcc', back: '#0a1520' },
    ai: { head: '#ff3355', rubber: '#cc1133', back: '#1a0a0e' },
    table: { surface: '#0a2233', clearcoat: 0.9, roughness: 0.05, metalness: 0.2 },
    ball: { color: '#e8eef5', emissive: '#335577' },
    accent: '#1a88ff'
  },
  {
    name: '🔥 Inferno',
    player: { head: '#ff4400', rubber: '#cc2200', back: '#220800' },
    ai: { head: '#ffaa00', rubber: '#cc8800', back: '#221a00' },
    table: { surface: '#1a0800', clearcoat: 0.4, roughness: 0.85, metalness: 0.05 },
    ball: { color: '#ffee55', emissive: '#ffaa00' },
    accent: '#ff4400',
    inferno: true
  },
  {
    name: '🪩 Party',
    player: { head: '#ff00ff', rubber: '#cc00cc', back: '#220022' },
    ai: { head: '#00ffff', rubber: '#00cccc', back: '#002222' },
    table: { surface: '#0a001a', clearcoat: 0.7, roughness: 0.5, metalness: 0.2 },
    ball: { color: '#ffffff', emissive: '#ff88ff' },
    accent: '#ff00ff',
    party: true
  },
  {
    name: '👾 Retro',
    player: { head: '#33ff66', rubber: '#22cc44', back: '#0a1a0e' },
    ai: { head: '#ff3333', rubber: '#cc2222', back: '#1a0a0a' },
    table: { surface: '#0a0e0a', clearcoat: 0.1, roughness: 1.0, metalness: 0.0 },
    ball: { color: '#33ff66', emissive: '#22aa44' },
    accent: '#33ff66',
    retro: true
  },
  {
    name: '🌊 Zen',
    player: { head: '#7ab8d4', rubber: '#5a9ec0', back: '#1e2d3d' },
    ai: { head: '#b8a0d4', rubber: '#9a80c0', back: '#241e38' },
    table: { surface: '#0a2e4a', clearcoat: 0.9, roughness: 0.2, metalness: 0.08 },
    ball: { color: '#d0e8f0', emissive: '#5a9ec0' },
    accent: '#7ab8d4',
    zen: true
  },
];

function applyPaddleSkin(paddle, skin) {
  paddle.children.forEach(child => {
    if (child.name.includes('Head')) {
      child.material.color.set(skin.head);
    } else if (child.name.includes('RubberFront')) {
      child.material.color.set(skin.rubber);
    } else if (child.name.includes('RubberBack')) {
      child.material.color.set(skin.back);
    }
    // Ring stays cyan — no override needed
  });
}

function applyTableSkin(skin) {
  tableMat.color.set(skin.surface);
  tableMat.clearcoat = skin.clearcoat;
  tableMat.roughness = skin.roughness;
  tableMat.metalness = skin.metalness;
  tableMat.needsUpdate = true;
}

let currentTheme = 0;

// Theme color lerp animation
let themeLerpActive = false;
let themeLerpProgress = 0;
const THEME_LERP_DURATION = 0.3; // 300ms
const themeLerpFrom = {
  playerHead: new THREE.Color(),
  playerRubber: new THREE.Color(),
  playerBack: new THREE.Color(),
  aiHead: new THREE.Color(),
  aiRubber: new THREE.Color(),
  aiBack: new THREE.Color(),
  tableSurface: new THREE.Color(),
  tableClearcoat: 0,
  tableRoughness: 0,
  tableMetalness: 0,
  ballColor: new THREE.Color(),
  ballEmissive: new THREE.Color(),
};
const themeLerpTo = {
  playerHead: new THREE.Color(),
  playerRubber: new THREE.Color(),
  playerBack: new THREE.Color(),
  aiHead: new THREE.Color(),
  aiRubber: new THREE.Color(),
  aiBack: new THREE.Color(),
  tableSurface: new THREE.Color(),
  tableClearcoat: 0,
  tableRoughness: 0,
  tableMetalness: 0,
  ballColor: new THREE.Color(),
  ballEmissive: new THREE.Color(),
};
const themeLerpCurrent = new THREE.Color();

function captureCurrentColors(target) {
  playerPaddle.children.forEach(child => {
    if (child.name.includes('Head')) target.playerHead.copy(child.material.color);
    else if (child.name.includes('RubberFront')) target.playerRubber.copy(child.material.color);
    else if (child.name.includes('RubberBack')) target.playerBack.copy(child.material.color);
  });
  aiPaddle.children.forEach(child => {
    if (child.name.includes('Head')) target.aiHead.copy(child.material.color);
    else if (child.name.includes('RubberFront')) target.aiRubber.copy(child.material.color);
    else if (child.name.includes('RubberBack')) target.aiBack.copy(child.material.color);
  });
  target.tableSurface.copy(tableMat.color);
  target.tableClearcoat = tableMat.clearcoat;
  target.tableRoughness = tableMat.roughness;
  target.tableMetalness = tableMat.metalness;
  target.ballColor.copy(ballMat.color);
  target.ballEmissive.copy(ballMat.emissive);
}

function setTargetColors(target, theme) {
  target.playerHead.set(theme.player.head);
  target.playerRubber.set(theme.player.rubber);
  target.playerBack.set(theme.player.back);
  target.aiHead.set(theme.ai.head);
  target.aiRubber.set(theme.ai.rubber);
  target.aiBack.set(theme.ai.back);
  target.tableSurface.set(theme.table.surface);
  target.tableClearcoat = theme.table.clearcoat;
  target.tableRoughness = theme.table.roughness;
  target.tableMetalness = theme.table.metalness;
  target.ballColor.set(theme.ball.color);
  target.ballEmissive.set(theme.ball.emissive);
}

// Current active theme colors for particles and UI
let activePlayerColor = '#2299ff';
let activeAiColor = '#ff4466';

// --- Party Mode State ---
let partyModeActive = false;
let partyTime = 0;
const partyColors = [
  new THREE.Color('#ff00ff'),
  new THREE.Color('#00ffff'),
  new THREE.Color('#ffff00'),
  new THREE.Color('#ff4400'),
  new THREE.Color('#00ff88'),
  new THREE.Color('#ff0066'),
  new THREE.Color('#4400ff'),
  new THREE.Color('#00ffaa'),
];
const partyLights = [];
let partyBassOsc = null;
let partyBassGain = null;
let partyBeatInterval = null;
let partyBallMirrorGeo = null;
let partyBallMirrorMesh = null;
let partyLaserOverlay = null;
let partyNeonVignette = null;
let partyFloorGlow = null;
let partyNeonParticles = [];
let _partyLaserBeams = null; // cached beam DOM elements

// Disco puck (mirror ball) — small faceted sphere that replaces puck visual in party mode
function createDiscoBall() {
  if (partyBallMirrorMesh) return;
  partyBallMirrorGeo = new THREE.IcosahedronGeometry(BALL_RADIUS * 1.1, 1);
  const mirrorMat = new THREE.MeshStandardMaterial({
    color: 0xffffff, metalness: 1.0, roughness: 0.05,
    envMapIntensity: 3.0,
  });
  partyBallMirrorMesh = new THREE.Mesh(partyBallMirrorGeo, mirrorMat);
  partyBallMirrorMesh.name = 'discoBall';
  partyBallMirrorMesh.visible = false;
  scene.add(partyBallMirrorMesh);
}
createDiscoBall();

// --- Party Overlays ---
// Hoisted tile metadata (used by updatePartyMode outside createPartyOverlays scope)
let tileCount = 0;
let _tileEdgeFade = null;
let _tileTX = null;
let _tileTZ = null;

function createPartyOverlays() {
  if (partyLaserOverlay) return;

  // Laser beam overlay — animated neon lines sweeping across screen
  partyLaserOverlay = document.createElement('div');
  partyLaserOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9993; opacity: 0;
    transition: opacity 0.8s;
    overflow: hidden;
  `;
  // Create laser beams
  for (let i = 0; i < 6; i++) {
    const beam = document.createElement('div');
    const hue = (i / 6) * 360;
    beam.style.cssText = `
      position: absolute;
      width: 200%; height: 2px;
      top: ${15 + i * 14}%;
      left: -50%;
      background: linear-gradient(90deg, transparent 0%, hsla(${hue},100%,60%,0) 10%, hsla(${hue},100%,60%,0.8) 50%, hsla(${hue},100%,60%,0) 90%, transparent 100%);
      transform-origin: center center;
      box-shadow: 0 0 12px 4px hsla(${hue},100%,50%,0.5), 0 0 30px 8px hsla(${hue},100%,50%,0.2);
      filter: blur(0.5px);
    `;
    beam.className = 'party-laser-beam';
    beam.dataset.index = i;
    partyLaserOverlay.appendChild(beam);
  }
  document.body.appendChild(partyLaserOverlay);
  _partyLaserBeams = Array.from(partyLaserOverlay.querySelectorAll('.party-laser-beam'));

  // Neon vignette — dark edges with colored glow
  partyNeonVignette = document.createElement('div');
  partyNeonVignette.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9992; opacity: 0;
    transition: opacity 0.8s;
    background: radial-gradient(ellipse at center, transparent 10%, rgba(0,0,0,0.6) 40%, rgba(0,0,0,0.95) 100%);
  `;
  document.body.appendChild(partyNeonVignette);

  // Dance floor — InstancedMesh: 625 tiles in 1 draw call
  const tileSize = 1.6;
  const tilesX = 25, tilesZ = 25;
  tileCount = tilesX * tilesZ;
  const tileCenterX = (tilesX - 1) / 2;
  const tileCenterZ = (tilesZ - 1) / 2;
  const maxDist = Math.sqrt(tileCenterX * tileCenterX + tileCenterZ * tileCenterZ);
  const tileGeo = new THREE.PlaneGeometry(tileSize * 0.9, tileSize * 0.9);
  const tileMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 1, side: THREE.DoubleSide, fog: false
  });
  partyFloorGlow = new THREE.InstancedMesh(tileGeo, tileMat, tileCount);
  partyFloorGlow.name = 'partyDanceFloor';
  partyFloorGlow.visible = false;
  // Per-instance color attribute
  const tileColors = new Float32Array(tileCount * 3);
  partyFloorGlow.instanceColor = new THREE.InstancedBufferAttribute(tileColors, 3);
  // Pre-compute tile metadata arrays for update loop
  _tileEdgeFade = new Float32Array(tileCount);
  _tileTX = new Uint8Array(tileCount);
  _tileTZ = new Uint8Array(tileCount);
  const _tileMatrix = new THREE.Matrix4();
  const _tileRotMatrix = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
  for (let ix = 0; ix < tilesX; ix++) {
    for (let iz = 0; iz < tilesZ; iz++) {
      const idx = ix * tilesZ + iz;
      const dx = ix - tileCenterX;
      const dz = iz - tileCenterZ;
      const dist = Math.sqrt(dx * dx + dz * dz);
      const edgeFade = Math.max(0, 1 - (dist / maxDist));
      _tileEdgeFade[idx] = edgeFade * edgeFade;
      _tileTX[idx] = ix;
      _tileTZ[idx] = iz;
      _tileMatrix.identity();
      _tileMatrix.makeTranslation(
        (ix - tileCenterX) * tileSize,
        -1.99,
        (iz - tileCenterZ) * tileSize
      );
      _tileMatrix.multiply(_tileRotMatrix);
      partyFloorGlow.setMatrixAt(idx, _tileMatrix);
    }
  }
  partyFloorGlow.instanceMatrix.needsUpdate = true;
  scene.add(partyFloorGlow);
}
createPartyOverlays();

// Reusable scratch objects for per-frame updates (shared across all themes)
const _tmpColor = new THREE.Color();
const _reusableVec = new THREE.Vector3();
const _reusableVec2 = new THREE.Vector3();
const _reusableVec3 = new THREE.Vector3();
const _reusableQuat = new THREE.Quaternion();
const _reusableEuler = new THREE.Euler();
const _reusableMatrix = new THREE.Matrix4();

// Party CSS animations
const partyStyleSheet = document.createElement('style');
partyStyleSheet.id = 'party-mode-styles';
partyStyleSheet.textContent = `
  @keyframes partyPulseVignette {
    0%, 100% { opacity: 0.85; }
    50% { opacity: 0.6; }
  }
`;
document.head.appendChild(partyStyleSheet);

// Floating neon particles — InstancedMesh: 80 particles in 1 draw call
const NEON_PARTICLE_COUNT = 80;
let partyNeonIM = null;
const _neonData = [];
const _neonMatrix = new THREE.Matrix4();
const _neonPos = new THREE.Vector3();
function createPartyNeonParticles() {
  if (partyNeonIM) return;
  const pGeo = new THREE.SphereGeometry(0.04, 4, 4);
  const pMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.8 });
  partyNeonIM = new THREE.InstancedMesh(pGeo, pMat, NEON_PARTICLE_COUNT);
  partyNeonIM.name = 'partyNeonInstanced';
  partyNeonIM.visible = false;
  partyNeonIM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NEON_PARTICLE_COUNT * 3), 3);
  for (let i = 0; i < NEON_PARTICLE_COUNT; i++) {
    const hue = Math.random();
    const baseY = Math.random() * 8 - 1;
    const px = (Math.random() - 0.5) * 14;
    const pz = (Math.random() - 0.5) * 14;
    _neonData.push({
      hue,
      speed: 0.3 + Math.random() * 1.2,
      drift: (Math.random() - 0.5) * 0.8,
      phase: Math.random() * Math.PI * 2,
      baseY,
      x: px, y: baseY, z: pz,
      scale: 1
    });
    _neonMatrix.identity();
    _neonMatrix.makeTranslation(px, baseY, pz);
    partyNeonIM.setMatrixAt(i, _neonMatrix);
    _tmpColor.setHSL(hue, 1, 0.6);
    partyNeonIM.setColorAt(i, _tmpColor);
  }
  partyNeonIM.instanceMatrix.needsUpdate = true;
  partyNeonIM.instanceColor.needsUpdate = true;
  scene.add(partyNeonIM);
}
createPartyNeonParticles();

// Create party spot lights (colored, rotating) — more lights, higher intensity
function initPartyLights() {
  if (partyLights.length > 0) return;
  const spotColors = [0xff00ff, 0x00ffff, 0xffff00, 0xff4400, 0x00ff88, 0x4400ff, 0xff0066, 0x00ffaa];
  for (let i = 0; i < spotColors.length; i++) {
    const light = new THREE.PointLight(spotColors[i], 0, 18);
    light.name = `partyLight${i}`;
    scene.add(light);
    partyLights.push({ light, baseColor: new THREE.Color(spotColors[i]), angle: (i / spotColors.length) * Math.PI * 2 });
  }
}
initPartyLights();

function startPartyMode() {
  if (partyModeActive) return;
  partyModeActive = true;
  partyTime = 0;

  // Force background and fog to black immediately to avoid carryover from other themes
  if (scene.background && scene.background.isColor) scene.background.set(0x000000);
  else scene.background = new THREE.Color(0x000000);
  scene.fog.color.set(0x000000);
  scene.fog.density = 0.03;

  // Show mirror ball on game ball
  if (partyBallMirrorMesh) partyBallMirrorMesh.visible = true;
  ball.visible = false;

  // Activate party lights — cranked up for dramatic light pools
  partyLights.forEach(p => { p.light.intensity = 4.0; });

  // Show overlays
  if (partyNeonVignette) partyNeonVignette.style.opacity = '1';
  if (partyLaserOverlay) partyLaserOverlay.style.opacity = '1';

  // Show dance floor
  if (partyFloorGlow) {
    partyFloorGlow.visible = true;
  }

  // Make actual floor darker and slightly reflective for the club look
  floor.material.color.set(0x020208);
  floor.material.roughness = 0.4;
  floor.material.metalness = 0.3;



  // Show neon particles
  if (partyNeonIM) { partyNeonIM.visible = true; partyNeonIM.material.opacity = 0.8; }

  // High contrast saturated club look
  renderer.domElement.style.filter = 'contrast(1.4) saturate(1.8) brightness(0.9)';

  // Bass beat loop
  startPartyBeat();
}

function stopPartyMode() {
  if (!partyModeActive) return;
  partyModeActive = false;

  // Hide disco elements
  if (partyBallMirrorMesh) partyBallMirrorMesh.visible = false;
  ball.visible = true;

  // Turn off party lights
  partyLights.forEach(p => { p.light.intensity = 0; });

  // Hide overlays
  if (partyLaserOverlay) partyLaserOverlay.style.opacity = '0';
  if (partyNeonVignette) partyNeonVignette.style.opacity = '0';

  // Hide dance floor
  if (partyFloorGlow) {
    partyFloorGlow.visible = false;
  }

  // Reset floor material
  floor.material.color.set(0x000000);
  floor.material.roughness = 0.9;
  floor.material.metalness = 0;

  // Reset fog density
  scene.fog.density = 0.055;

  // Hide neon particles
  if (partyNeonIM) { partyNeonIM.visible = false; partyNeonIM.material.opacity = 0; }

  // Stop bass beat
  stopPartyBeat();

  // Reset renderer filter
  renderer.domElement.style.filter = '';

  // Reset fog and background
  scene.fog.color.set(0x000000);
  if (scene.background && scene.background.isColor) scene.background.set(0x000000);
}

function updatePartyMode(dt) {
  if (!partyModeActive) return;
  partyTime += dt;

  // Enforce fog every frame — other theme stop functions can overwrite it
  scene.fog.density = 0.03;
  scene.fog.color.set(0x000000);

  // Rotate disco lights around the table — faster, more chaotic
  partyLights.forEach((p, i) => {
    const a = p.angle + partyTime * (1.8 + i * 0.2);
    const r = 3.5 + Math.sin(partyTime * 1.2 + i * 0.7) * 2.0;
    p.light.position.set(
      Math.cos(a) * r,
      2.5 + Math.sin(partyTime * 2.0 + i * 1.1) * 2.0,
      Math.sin(a) * r
    );
    // Fast color cycling
    const colorIdx = Math.floor(partyTime * 3 + i * 1.3) % partyColors.length;
    const nextIdx = (colorIdx + 1) % partyColors.length;
    const t = (partyTime * 3 + i * 1.3) % 1;
    p.light.color.copy(partyColors[colorIdx]).lerp(partyColors[nextIdx], t);
    // Pulsing intensity synced to beat (128 BPM = ~2.13 beats/sec)
    const beatPhase = (partyTime * 2.133) % 1;
    const beatPulse = beatPhase < 0.1 ? 1.0 : Math.exp(-beatPhase * 4);
    p.light.intensity = 2.0 + beatPulse * 4.0 + Math.sin(partyTime * 8 + i * 1.5) * 0.8;
  });

  // Mirror ball follows game ball — faster spin
  if (partyBallMirrorMesh) {
    partyBallMirrorMesh.position.copy(gameState.ballPos);
    partyBallMirrorMesh.rotation.y += dt * 15;
    partyBallMirrorMesh.rotation.x += dt * 9;
    // Pulsing emissive on the mirror ball
    const pulse = Math.sin(partyTime * 8) * 0.5 + 0.5;
    const hue = (partyTime * 0.3) % 1;
    if (!partyBallMirrorMesh.material.emissive) partyBallMirrorMesh.material.emissive = _tmpColor.clone();
    partyBallMirrorMesh.material.emissive.setHSL(hue, 1, 0.15 + pulse * 0.25);
    partyBallMirrorMesh.material.emissiveIntensity = 1.5 + pulse * 1.5;
  }

  // Animate laser beams — sweep across screen (cached beam elements)
  if (_partyLaserBeams) {
    for (let i = 0; i < _partyLaserBeams.length; i++) {
      const beam = _partyLaserBeams[i];
      const idx = i;
      const angle = Math.sin(partyTime * (0.5 + idx * 0.15) + idx * 1.2) * 25;
      const yOff = Math.sin(partyTime * 0.8 + idx * 0.9) * 8;
      beam.style.transform = `rotate(${angle}deg) translateY(${yOff}px)`;
      // Cycle beam colors
      const hue = ((partyTime * 40 + idx * 60) % 360);
      beam.style.background = `linear-gradient(90deg, transparent 0%, hsla(${hue},100%,60%,0) 10%, hsla(${hue},100%,60%,0.7) 50%, hsla(${hue},100%,60%,0) 90%, transparent 100%)`;
      beam.style.boxShadow = `0 0 14px 5px hsla(${hue},100%,50%,0.4), 0 0 35px 10px hsla(${hue},100%,50%,0.15)`;
      // Pulse opacity on beat
      const bPulse = Math.sin(partyTime * 4 + idx) * 0.3 + 0.7;
      beam.style.opacity = bPulse;
    }
  }

  // Neon vignette — heavier pulse to the beat, darker overall
  if (partyNeonVignette) {
    const vBeat = (partyTime * 2.133) % 1;
    const vPump = vBeat < 0.1 ? 0.7 : 0.85 + (1 - Math.exp(-vBeat * 3)) * 0.1;
    partyNeonVignette.style.opacity = vPump.toString();
  }

  // Dance floor tiles — instanced color-cycling checkerboard with beat pulse, edge fade, and manual fog
  if (partyFloorGlow && partyFloorGlow.visible) {
    const beatPhase = (partyTime * 2.133) % 1;
    const beatPump = beatPhase < 0.1 ? 1.0 : Math.exp(-beatPhase * 5);
    const stepBeat = Math.floor(partyTime * 2.133);
    const colorArr = partyFloorGlow.instanceColor.array;
    const fogDensity = scene.fog.density;
    const fogR = scene.fog.color.r, fogG = scene.fog.color.g, fogB = scene.fog.color.b;
    const camX = camera.position.x, camY = camera.position.y, camZ = camera.position.z;
    const tileSize = 1.6;
    const halfX = 12, halfZ = 12; // (tilesX-1)/2, (tilesZ-1)/2
    for (let idx = 0; idx < tileCount; idx++) {
      const tx = _tileTX[idx];
      const tz = _tileTZ[idx];
      const fade = _tileEdgeFade[idx];
      const checker = (tx + tz + stepBeat) % 3;
      const hue = ((idx * 0.04 + partyTime * 0.2 + checker * 0.33) % 1);
      const lit = checker === 0 ? 0.55 + beatPump * 0.3 : (checker === 1 ? 0.2 : 0.05);
      _tmpColor.setHSL(hue, 1, lit * fade);
      // Manual FogExp2 per instance
      const wx = (tx - halfX) * tileSize;
      const wz = (tz - halfZ) * tileSize;
      const dx = wx - camX, dy = -1.99 - camY, dz = wz - camZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const fogFactor = Math.exp(-fogDensity * fogDensity * dist * dist);
      const i3 = idx * 3;
      colorArr[i3] = fogR + (_tmpColor.r - fogR) * fogFactor;
      colorArr[i3 + 1] = fogG + (_tmpColor.g - fogG) * fogFactor;
      colorArr[i3 + 2] = fogB + (_tmpColor.b - fogB) * fogFactor;
    }
    partyFloorGlow.instanceColor.needsUpdate = true;
  }

  // Neon floating particles — instanced drift and pulse
  if (partyNeonIM && partyNeonIM.visible) {
    const nColors = partyNeonIM.instanceColor.array;
    for (let i = 0; i < NEON_PARTICLE_COUNT; i++) {
      const p = _neonData[i];
      p.y = p.baseY + Math.sin(partyTime * p.speed + p.phase) * 1.5;
      p.x += Math.sin(partyTime * 0.5 + p.phase) * p.drift * dt;
      p.z += Math.cos(partyTime * 0.3 + p.phase) * p.drift * dt;
      // Wrap around bounds
      if (p.x > 7) p.x = -7; if (p.x < -7) p.x = 7;
      if (p.z > 7) p.z = -7; if (p.z < -7) p.z = 7;
      // Scale pulse
      const sPulse = 0.8 + Math.sin(partyTime * 8 + i) * 0.4;
      _neonMatrix.identity();
      _neonMatrix.makeTranslation(p.x, p.y, p.z);
      _neonMatrix.scale(_neonPos.set(sPulse, sPulse, sPulse));
      partyNeonIM.setMatrixAt(i, _neonMatrix);
      // Cycle particle color
      const h = (p.hue + partyTime * 0.1) % 1;
      _tmpColor.setHSL(h, 1, 0.55);
      const i3 = i * 3;
      nColors[i3] = _tmpColor.r;
      nColors[i3 + 1] = _tmpColor.g;
      nColors[i3 + 2] = _tmpColor.b;
    }
    partyNeonIM.instanceMatrix.needsUpdate = true;
    partyNeonIM.instanceColor.needsUpdate = true;
    // Pulse opacity
    const pPulse = Math.sin(partyTime * 6) * 0.3 + 0.6;
    partyNeonIM.material.opacity = pPulse;
  }

  // Ultra-dark background with subtle hue shift on beat — keep fog strictly black
  const bgPulse = Math.sin(partyTime * 4.266) * 0.5 + 0.5;
  _tmpColor.setHSL((partyTime * 0.04) % 1, 0.3, 0.001 + bgPulse * 0.003);
  if (scene.background && scene.background.isColor) scene.background.copy(_tmpColor);
  else { _tmpColor.set(0x000000); scene.background = _tmpColor.clone(); }
  scene.fog.color.set(0x000000);

  // Occasionally trigger random tile flash waves (ripple from center) with manual fog
  if (partyFloorGlow && partyFloorGlow.visible && Math.floor(partyTime * 2.133) % 8 === 0) {
    const ripple = (partyTime * 3) % 16;
    const colorArr = partyFloorGlow.instanceColor.array;
    const fogDen = scene.fog.density;
    const fR = scene.fog.color.r, fG = scene.fog.color.g, fB = scene.fog.color.b;
    const cX = camera.position.x, cY = camera.position.y, cZ = camera.position.z;
    const ts = 1.6;
    for (let idx = 0; idx < tileCount; idx++) {
      const tx = _tileTX[idx] - 12;
      const tz = _tileTZ[idx] - 12;
      const dist = Math.sqrt(tx * tx + tz * tz);
      const fade = _tileEdgeFade[idx];
      if (Math.abs(dist - ripple) < 1.5 && fade > 0.05) {
        _tmpColor.setHSL(Math.random(), 1, 0.7 * fade);
        // Manual FogExp2
        const wx = tx * ts, wz = tz * ts;
        const ddx = wx - cX, ddy = -1.99 - cY, ddz = wz - cZ;
        const d = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
        const ff = Math.exp(-fogDen * fogDen * d * d);
        const i3 = idx * 3;
        colorArr[i3] = fR + (_tmpColor.r - fR) * ff;
        colorArr[i3 + 1] = fG + (_tmpColor.g - fG) * ff;
        colorArr[i3 + 2] = fB + (_tmpColor.b - fB) * ff;
      }
    }
    partyFloorGlow.instanceColor.needsUpdate = true;
  }
}

// Party beat sequencer — full drum machine with patterns
let partySeqStep = 0;
let partySeqPattern = 0;
let partySeqTimeout = null;

// Each pattern is 16 steps. 1 = trigger, 0 = silent
// K=kick, S=snare, CH=closed hat, OH=open hat
const partyPatterns = [
  { // Pattern 0: Four-on-the-floor house
    K:  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
    S:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
    CH: [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    OH: [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
  },
  { // Pattern 1: Syncopated funk
    K:  [1,0,0,0, 0,0,1,0, 0,1,0,0, 1,0,0,0],
    S:  [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
    CH: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    OH: [0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0,1,0],
  },
  { // Pattern 2: Driving techno
    K:  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
    S:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
    CH: [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
    OH: [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
  },
  { // Pattern 3: Breakbeat shuffle
    K:  [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
    S:  [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,1,0],
    CH: [1,0,1,1, 0,0,1,0, 1,1,0,1, 0,0,1,0],
    OH: [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,1,0,0],
  },
];

function startPartyBeat() {
  stopPartyBeat();
  partySeqStep = 0;
  partySeqPattern = 0;
  schedulePartyStep();
}

function schedulePartyStep() {
  if (!partyModeActive) return;
  try {
    const ctx = getAudioCtx();
    const bpm = 128;
    const stepMs = (60000 / bpm) / 4; // 16th note
    const t = ctx.currentTime;
    const pat = partyPatterns[partySeqPattern];
    const step = partySeqStep;

    // Kick drum
    if (pat.K[step]) {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(110, t);
      osc.frequency.exponentialRampToValueAtTime(35, t + 0.12);
      g.gain.setValueAtTime(0.28, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.22);
      osc.connect(g); g.connect(ctx.destination);
      osc.start(t); osc.stop(t + 0.22);
      // Kick click layer
      const osc2 = ctx.createOscillator();
      const g2 = ctx.createGain();
      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(1500, t);
      osc2.frequency.exponentialRampToValueAtTime(100, t + 0.015);
      g2.gain.setValueAtTime(0.12, t);
      g2.gain.exponentialRampToValueAtTime(0.001, t + 0.02);
      osc2.connect(g2); g2.connect(ctx.destination);
      osc2.start(t); osc2.stop(t + 0.02);
    }

    // Snare
    if (pat.S[step]) {
      // Tone body
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(220, t);
      osc.frequency.exponentialRampToValueAtTime(120, t + 0.06);
      g.gain.setValueAtTime(0.15, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
      osc.connect(g); g.connect(ctx.destination);
      osc.start(t); osc.stop(t + 0.1);
      // Noise burst
      const ns = ctx.createBufferSource();
      const bufSz = Math.floor(ctx.sampleRate * 0.08);
      const buf = ctx.createBuffer(1, bufSz, ctx.sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < bufSz; i++) ch[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSz * 0.15));
      ns.buffer = buf;
      const ng = ctx.createGain();
      ng.gain.setValueAtTime(0.13, t);
      ng.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass'; bp.frequency.value = 4000; bp.Q.value = 1.2;
      ns.connect(bp); bp.connect(ng); ng.connect(ctx.destination);
      ns.start(t); ns.stop(t + 0.08);
    }

    // Closed hi-hat
    if (pat.CH[step]) {
      const ns = ctx.createBufferSource();
      const bufSz = Math.floor(ctx.sampleRate * 0.03);
      const buf = ctx.createBuffer(1, bufSz, ctx.sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < bufSz; i++) ch[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSz * 0.05));
      ns.buffer = buf;
      const ng = ctx.createGain();
      // Accent on beats 0,4,8,12 for groove
      const accent = (step % 4 === 0) ? 0.06 : 0.035;
      ng.gain.setValueAtTime(accent, t);
      ng.gain.exponentialRampToValueAtTime(0.001, t + 0.03);
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass'; hp.frequency.value = 7000;
      ns.connect(hp); hp.connect(ng); ng.connect(ctx.destination);
      ns.start(t); ns.stop(t + 0.03);
    }

    // Open hi-hat
    if (pat.OH[step]) {
      const ns = ctx.createBufferSource();
      const bufSz = Math.floor(ctx.sampleRate * 0.12);
      const buf = ctx.createBuffer(1, bufSz, ctx.sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < bufSz; i++) ch[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSz * 0.4));
      ns.buffer = buf;
      const ng = ctx.createGain();
      ng.gain.setValueAtTime(0.07, t);
      ng.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass'; hp.frequency.value = 5500;
      ns.connect(hp); hp.connect(ng); ng.connect(ctx.destination);
      ns.start(t); ns.stop(t + 0.12);
    }

    // Advance step
    partySeqStep++;
    if (partySeqStep >= 16) {
      partySeqStep = 0;
      // Switch pattern every 2 bars (32 steps = 2 passes of 16)
      if (Math.random() < 0.5) {
        partySeqPattern = Math.floor(Math.random() * partyPatterns.length);
      }
    }

    partySeqTimeout = setTimeout(schedulePartyStep, stepMs);
  } catch (e) { /* audio not available */ }
}

function stopPartyBeat() {
  if (partyBeatInterval) {
    clearInterval(partyBeatInterval);
    partyBeatInterval = null;
  }
  if (partySeqTimeout) {
    clearTimeout(partySeqTimeout);
    partySeqTimeout = null;
  }
}

// Party-mode paddle hit sound (bass-heavy with synth)
function playPartyHitSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Deep bass hit
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(200, t);
    osc.frequency.exponentialRampToValueAtTime(60, t + 0.15);
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.2);

    // Synth zap
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sawtooth';
    osc2.frequency.setValueAtTime(2000, t);
    osc2.frequency.exponentialRampToValueAtTime(300, t + 0.08);
    gain2.gain.setValueAtTime(0.06, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.1);
  } catch (e) { /* audio not available */ }
}

// Party-mode bounce sound (bass thump + blip)
function playPartyBounceSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(250, t);
    osc.frequency.exponentialRampToValueAtTime(80, t + 0.06);
    gain.gain.setValueAtTime(0.18, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.08);

    // High blip
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(1500 + Math.random() * 800, t);
    gain2.gain.setValueAtTime(0.06, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.05);
  } catch (e) { /* audio not available */ }
}

// --- Retro Mode State ---
let retroModeActive = false;
let retroTime = 0;
let retroScanlineOverlay = null;
let retroCRTOverlay = null;
let retroPixelOverlay = null;
let retroGlitchOverlay = null;
let retroChiptuneInterval = null;
let retroGlitchTimeout = null;
let retroPrevBallMat = null;
let retroBallGeo = null;
let retroBallMesh = null;

// Retro grid lines on the floor (Tron-style)
const retroGridLines = [];

function createRetroOverlays() {
  if (retroScanlineOverlay) return;

  // Heavy scanline overlay — thick CRT lines
  retroScanlineOverlay = document.createElement('div');
  retroScanlineOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9998; opacity: 0;
    transition: opacity 0.5s;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 3px,
      rgba(0, 0, 0, 0.25) 3px,
      rgba(0, 0, 0, 0.25) 6px
    );
    mix-blend-mode: multiply;
  `;
  document.body.appendChild(retroScanlineOverlay);

  // CRT curvature + vignette overlay — heavier
  retroCRTOverlay = document.createElement('div');
  retroCRTOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9997; opacity: 0;
    transition: opacity 0.5s;
    background: radial-gradient(ellipse at center, transparent 40%, rgba(0, 0, 0, 0.7) 100%);
    box-shadow: inset 0 0 200px rgba(0, 30, 0, 0.5);
  `;
  document.body.appendChild(retroCRTOverlay);

  // Pixel grid overlay — tiny grid pattern
  retroPixelOverlay = document.createElement('div');
  retroPixelOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9996; opacity: 0;
    transition: opacity 0.5s;
    background-image:
      linear-gradient(rgba(0,255,60,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,60,0.03) 1px, transparent 1px);
    background-size: 4px 4px;
  `;
  document.body.appendChild(retroPixelOverlay);

  // Glitch flash overlay — random color aberration moments
  retroGlitchOverlay = document.createElement('div');
  retroGlitchOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9999; opacity: 0;
    transition: opacity 0.05s;
    background: transparent;
  `;
  document.body.appendChild(retroGlitchOverlay);
}
createRetroOverlays();

// Retro ambient glow light (dim green)
const retroAmbientLight = new THREE.PointLight(0x33ff66, 0, 20);
retroAmbientLight.name = 'retroAmbientLight';
retroAmbientLight.position.set(0, 6, 0);
scene.add(retroAmbientLight);

// Second retro light for red/green flicker
const retroAccentLight = new THREE.PointLight(0xff3333, 0, 15);
retroAccentLight.name = 'retroAccentLight';
retroAccentLight.position.set(0, 4, -3);
scene.add(retroAccentLight);

// Retro grid floor lines
function createRetroGrid() {
  if (retroGridLines.length > 0) return;
  const gridMat = new THREE.MeshBasicMaterial({ color: 0x33ff66, transparent: true, opacity: 0 });
  // Horizontal lines
  for (let i = -20; i <= 20; i += 2) {
    const geo = new THREE.PlaneGeometry(80, 0.04);
    const line = new THREE.Mesh(geo, gridMat.clone());
    line.name = `retroGridH${i}`;
    line.rotation.x = -Math.PI / 2;
    line.position.set(0, -1.99, i);
    line.visible = false;
    scene.add(line);
    retroGridLines.push(line);
  }
  // Vertical lines
  for (let i = -20; i <= 20; i += 2) {
    const geo = new THREE.PlaneGeometry(0.04, 80);
    const line = new THREE.Mesh(geo, gridMat.clone());
    line.name = `retroGridV${i}`;
    line.rotation.x = -Math.PI / 2;
    line.position.set(i, -1.99, 0);
    line.visible = false;
    scene.add(line);
    retroGridLines.push(line);
  }
}
createRetroGrid();

// Retro-styled puck (flat low-poly octagon)
function createRetroBall() {
  if (retroBallMesh) return;
  retroBallGeo = new THREE.CylinderGeometry(BALL_RADIUS, BALL_RADIUS, 0.08, 8);
  const retroBallMat = new THREE.MeshBasicMaterial({ color: 0x33ff66 });
  retroBallMesh = new THREE.Mesh(retroBallGeo, retroBallMat);
  retroBallMesh.name = 'retroBall';
  retroBallMesh.visible = false;
  scene.add(retroBallMesh);
}
createRetroBall();

// Retro CSS style injection for scoreboard
const retroStyleSheet = document.createElement('style');
retroStyleSheet.id = 'retro-mode-styles';
retroStyleSheet.textContent = `
  .retro-ui * {
    font-family: 'Press Start 2P', monospace !important;
    text-shadow: 0 0 8px rgba(51,255,102,0.8), 0 0 16px rgba(51,255,102,0.4) !important;
  }
  .retro-ui span[style*="Instrument Serif"] {
    font-family: 'Press Start 2P', monospace !important;
  }
  @keyframes retroBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  .retro-scanline-anim {
    animation: retroScanMove 4s linear infinite;
  }
  @keyframes retroScanMove {
    0% { background-position: 0 0; }
    100% { background-position: 0 100vh; }
  }
  @keyframes retroGlitch {
    0% { transform: translate(0,0); }
    20% { transform: translate(-2px, 1px); }
    40% { transform: translate(2px, -1px); }
    60% { transform: translate(-1px, -2px); }
    80% { transform: translate(1px, 2px); }
    100% { transform: translate(0,0); }
  }
`;
document.head.appendChild(retroStyleSheet);

function startRetroMode() {
  if (retroModeActive) return;
  retroModeActive = true;
  retroTime = 0;

  // Show all CRT overlays
  if (retroScanlineOverlay) { retroScanlineOverlay.style.opacity = '1'; retroScanlineOverlay.classList.add('retro-scanline-anim'); }
  if (retroCRTOverlay) retroCRTOverlay.style.opacity = '1';
  if (retroPixelOverlay) retroPixelOverlay.style.opacity = '1';

  // Retro lights
  retroAmbientLight.intensity = 0.8;
  retroAccentLight.intensity = 0.5;

  // Show retro ball, hide normal ball
  if (retroBallMesh) retroBallMesh.visible = true;
  ball.visible = false;

  // Show grid lines with fade-in
  retroGridLines.forEach(line => { line.visible = true; line.material.opacity = 0.15; });

  // Apply retro font to ALL UI
  const uiEls = [uiContainer, statsBar, infoDiv, pointNotif, msgDiv];
  uiEls.forEach(el => el.classList.add('retro-ui'));

  // Make rink lines glow green
  lineMat.color.set(0x33ff66);
  edgeLineMat.color.set(0x33ff66);

  // Start chiptune background loop
  startChiptuneLoop();

  // Start periodic glitch effect
  scheduleGlitch();
}

function stopRetroMode() {
  if (!retroModeActive) return;
  retroModeActive = false;

  // Hide all overlays
  if (retroScanlineOverlay) { retroScanlineOverlay.style.opacity = '0'; retroScanlineOverlay.classList.remove('retro-scanline-anim'); }
  if (retroCRTOverlay) retroCRTOverlay.style.opacity = '0';
  if (retroPixelOverlay) retroPixelOverlay.style.opacity = '0';
  if (retroGlitchOverlay) retroGlitchOverlay.style.opacity = '0';

  // Kill retro lights
  retroAmbientLight.intensity = 0;
  retroAccentLight.intensity = 0;

  // Hide retro ball, show normal ball
  if (retroBallMesh) retroBallMesh.visible = false;
  ball.visible = true;

  // Hide grid lines
  retroGridLines.forEach(line => { line.material.opacity = 0; line.visible = false; });

  // Reset background/fog
  scene.fog.color.set(0x000000);
  if (scene.background && scene.background.isColor) scene.background.set(0x000000);

  // Remove retro UI class
  const uiEls = [uiContainer, statsBar, infoDiv, pointNotif, msgDiv];
  uiEls.forEach(el => el.classList.remove('retro-ui'));

  // Reset table lines to white
  lineMat.color.set(0xffffff);
  edgeLineMat.color.set(0xffffff);

  // Stop chiptune and glitch
  stopChiptuneLoop();
  if (retroGlitchTimeout) { clearTimeout(retroGlitchTimeout); retroGlitchTimeout = null; }

  // Remove canvas filter
  renderer.domElement.style.filter = '';
}

function updateRetroMode(dt) {
  if (!retroModeActive) return;
  retroTime += dt;

  // CRT flicker — random brightness jitter
  const flicker = 0.85 + Math.random() * 0.15;
  const bgBrightness = (0.015 + Math.sin(retroTime * 2.5) * 0.005) * flicker;
  _tmpColor.setRGB(bgBrightness * 0.2, bgBrightness, bgBrightness * 0.2);
  if (scene.background && scene.background.isColor) scene.background.copy(_tmpColor);
  scene.fog.color.copy(_tmpColor);

  // Pulse retro ambient lights
  retroAmbientLight.intensity = 0.6 + Math.sin(retroTime * 3) * 0.3;
  retroAccentLight.intensity = 0.3 + Math.sin(retroTime * 5 + 1.5) * 0.2;
  retroAccentLight.position.x = Math.sin(retroTime * 1.2) * 4;

  // Retro puck follows game puck
  if (retroBallMesh) {
    retroBallMesh.position.copy(gameState.ballPos);
    // Snappy Y-axis spin for flat puck
    retroBallMesh.rotation.y = Math.floor(retroTime * 8) * (Math.PI / 4);
    const pulse = Math.sin(retroTime * 10) > 0 ? 0x33ff66 : 0x66ffaa;
    retroBallMesh.material.color.set(pulse);
  }

  // Grid lines scroll/pulse
  retroGridLines.forEach((line, i) => {
    const wave = Math.sin(retroTime * 2 + i * 0.3) * 0.5 + 0.5;
    line.material.opacity = 0.05 + wave * 0.15;
    line.material.color.setHSL(0.38 + Math.sin(retroTime * 0.5 + i * 0.1) * 0.05, 1, 0.5);
  });

  // Occasional CRT horizontal tear — apply via canvas filter
  if (Math.random() < 0.02) {
    renderer.domElement.style.filter = `hue-rotate(${Math.random() * 20 - 10}deg) brightness(${0.9 + Math.random() * 0.3})`;
    setTimeout(() => { if (retroModeActive) renderer.domElement.style.filter = ''; }, 50);
  }

  // Phosphor afterglow: slightly tint the scanline overlay green
  if (retroScanlineOverlay) {
    const glow = 0.1 + Math.sin(retroTime * 6) * 0.05;
    retroScanlineOverlay.style.boxShadow = `inset 0 0 100px rgba(51, 255, 102, ${glow})`;
  }
}

// Glitch effect — random screen tears
function scheduleGlitch() {
  if (!retroModeActive) return;
  const delay = 2000 + Math.random() * 6000;
  retroGlitchTimeout = setTimeout(() => {
    if (!retroModeActive) return;
    triggerRetroGlitch();
    scheduleGlitch();
  }, delay);
}

function triggerRetroGlitch() {
  if (!retroGlitchOverlay) return;

  // Random color bars
  const bars = Math.floor(3 + Math.random() * 5);
  let bgParts = [];
  for (let i = 0; i < bars; i++) {
    const y = Math.random() * 100;
    const h = 1 + Math.random() * 4;
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    bgParts.push(`linear-gradient(transparent ${y}%, rgba(${r},${g},${b},0.3) ${y}%, rgba(${r},${g},${b},0.3) ${y + h}%, transparent ${y + h}%)`);
  }
  retroGlitchOverlay.style.background = bgParts.join(',');
  retroGlitchOverlay.style.opacity = '1';

  // Also offset the canvas briefly
  renderer.domElement.style.transform = `translateX(${(Math.random() - 0.5) * 8}px)`;

  // Play glitch sound
  playRetroGlitchSound();

  setTimeout(() => {
    if (retroGlitchOverlay) retroGlitchOverlay.style.opacity = '0';
    renderer.domElement.style.transform = '';
  }, 80 + Math.random() * 120);
}

// --- Retro Chiptune Background Loop ---
function startChiptuneLoop() {
  stopChiptuneLoop();
  try {
    const ctx = getAudioCtx();
    const bpm = 140;
    const interval = 60000 / bpm / 2; // 8th notes

    // Simple bass pattern (C minor pentatonic)
    const bassNotes = [65.41, 77.78, 87.31, 98.00, 116.54, 98.00, 87.31, 77.78];
    let step = 0;

    function playStep() {
      if (!retroModeActive) return;
      const t = ctx.currentTime;
      const note = bassNotes[step % bassNotes.length];

      // Bass — triangle wave for warmth
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(note, t);
      gain.gain.setValueAtTime(0.08, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.15);

      // Arp on every other step
      if (step % 2 === 0) {
        const arpNotes = [261.63, 311.13, 392.00, 466.16, 523.25];
        const arpNote = arpNotes[Math.floor(step / 2) % arpNotes.length];
        const osc2 = ctx.createOscillator();
        const gain2 = ctx.createGain();
        osc2.type = 'square';
        osc2.frequency.setValueAtTime(arpNote, t);
        gain2.gain.setValueAtTime(0.035, t);
        gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
        osc2.connect(gain2);
        gain2.connect(ctx.destination);
        osc2.start(t);
        osc2.stop(t + 0.08);
      }

      // Hi-hat on every 4th step
      if (step % 4 === 0) {
        const noise = ctx.createBufferSource();
        const bufSize = Math.floor(ctx.sampleRate * 0.02);
        const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
        const d = buf.getChannelData(0);
        for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.05));
        noise.buffer = buf;
        const nGain = ctx.createGain();
        nGain.gain.setValueAtTime(0.04, t);
        nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.02);
        const hp = ctx.createBiquadFilter();
        hp.type = 'highpass';
        hp.frequency.value = 8000;
        noise.connect(hp);
        hp.connect(nGain);
        nGain.connect(ctx.destination);
        noise.start(t);
        noise.stop(t + 0.02);
      }

      step++;
    }

    playStep();
    retroChiptuneInterval = setInterval(playStep, interval);
  } catch (e) { /* audio not available */ }
}

function stopChiptuneLoop() {
  if (retroChiptuneInterval) {
    clearInterval(retroChiptuneInterval);
    retroChiptuneInterval = null;
  }
}

// Retro paddle hit sound — classic PONG bleep with pitch bend
function playRetroHitSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Main square wave bleep
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, t);
    osc.frequency.exponentialRampToValueAtTime(440, t + 0.06);
    gain.gain.setValueAtTime(0.18, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.1);

    // Harmonics for that crunchy 8-bit feel
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'square';
    osc2.frequency.setValueAtTime(1320, t);
    osc2.frequency.exponentialRampToValueAtTime(660, t + 0.04);
    gain2.gain.setValueAtTime(0.06, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.06);

    // Noise burst for texture
    const noise = ctx.createBufferSource();
    const bufSize = Math.floor(ctx.sampleRate * 0.015);
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.03));
    noise.buffer = buf;
    const nGain = ctx.createGain();
    nGain.gain.setValueAtTime(0.08, t);
    nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.015);
    // Bitcrusher effect via sample rate reduction
    noise.connect(nGain);
    nGain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.015);
  } catch (e) { /* audio not available */ }
}

// Retro bounce sound — deeper blip
function playRetroBounceSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(520, t);
    osc.frequency.exponentialRampToValueAtTime(260, t + 0.05);
    gain.gain.setValueAtTime(0.1, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.07);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.07);

    // Second voice
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(260, t);
    gain2.gain.setValueAtTime(0.06, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.05);
  } catch (e) { /* audio not available */ }
}

// Retro score sound — ascending chiptune fanfare
function playRetroScoreSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;
    const notes = [523, 659, 784, 1047, 1319, 1568];
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(freq, t + i * 0.07);
      gain.gain.setValueAtTime(0.1, t + i * 0.07);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.07 + 0.06);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t + i * 0.07);
      osc.stop(t + i * 0.07 + 0.06);
    });
    // Final chord
    const chordNotes = [523, 784, 1047];
    chordNotes.forEach(freq => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(freq, t + 0.45);
      gain.gain.setValueAtTime(0.06, t + 0.45);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.75);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t + 0.45);
      osc.stop(t + 0.75);
    });
  } catch (e) { /* audio not available */ }
}

// Retro glitch sound — short noise burst
function playRetroGlitchSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;
    const noise = ctx.createBufferSource();
    const bufSize = Math.floor(ctx.sampleRate * 0.06);
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * 0.5;
    noise.buffer = buf;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.07, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
    noise.connect(gain);
    gain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.06);
  } catch (e) { /* audio not available */ }
}

// --- Zen Mode State ---
let zenModeActive = false;
let zenTime = 0;
let zenOverlay = null;
let zenParticles = [];
let zenAmbienceInterval = null;
let zenRipples = [];
let zenBallMesh = null;
let zenBallGeo = null;
let zenBallAura = null;
let zenFogOrigColor = new THREE.Color();
let zenBgOrigColor = new THREE.Color();
let zenPetals = [];
let zenLightRays = null;
let zenCausticLight = null;
let zenOrbParticles = [];

// Zen floating particles (dust motes / fireflies)
const zenFloaters = [];

// Zen ambient light (soft sky blue)
const zenWarmLight = new THREE.PointLight(0x88ccee, 0, 20);
zenWarmLight.name = 'zenWarmLight';
zenWarmLight.position.set(0, 6, 0);
scene.add(zenWarmLight);

const zenCoolLight = new THREE.PointLight(0x66aadd, 0, 15);
zenCoolLight.name = 'zenCoolLight';
zenCoolLight.position.set(-3, 4, 2);
scene.add(zenCoolLight);

const zenAccentLight = new THREE.PointLight(0xaaddff, 0, 12);
zenAccentLight.name = 'zenAccentLight';
zenAccentLight.position.set(3, 3, -2);
scene.add(zenAccentLight);

// Create zen overlays
function createZenOverlays() {
  if (zenOverlay) return;

  // Soft warm vignette overlay
  zenOverlay = document.createElement('div');
  zenOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9996; opacity: 0;
    transition: opacity 1.5s ease-in-out;
    background: radial-gradient(ellipse at center, transparent 30%, rgba(5, 15, 30, 0.55) 100%);
  `;
  document.body.appendChild(zenOverlay);
}
createZenOverlays();

// Zen floating motes — InstancedMesh: 60 motes in 1 draw call
const ZEN_MOTE_COUNT = 60;
let zenMoteIM = null;
const _zenMoteData = [];
const _zenMoteMatrix = new THREE.Matrix4();
function createZenFloaters() {
  if (zenMoteIM) return;
  const moteGeo = new THREE.SphereGeometry(0.015, 6, 6);
  const moteMat = new THREE.MeshBasicMaterial({ color: 0xaaddff, transparent: true, opacity: 0.4 });
  zenMoteIM = new THREE.InstancedMesh(moteGeo, moteMat, ZEN_MOTE_COUNT);
  zenMoteIM.name = 'zenMoteInstanced';
  zenMoteIM.visible = false;
  zenMoteIM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ZEN_MOTE_COUNT * 3), 3);
  for (let i = 0; i < ZEN_MOTE_COUNT; i++) {
    const col = Math.random() > 0.5 ? 0xaaddff : 0x77bbee;
    const baseY = Math.random() * 6 + 0.5;
    const px = (Math.random() - 0.5) * 14;
    const pz = (Math.random() - 0.5) * 14;
    _zenMoteData.push({
      x: px, y: baseY, z: pz,
      baseY,
      speed: 0.2 + Math.random() * 0.4,
      phase: Math.random() * Math.PI * 2,
      drift: (Math.random() - 0.5) * 0.3,
    });
    _zenMoteMatrix.identity();
    _zenMoteMatrix.makeTranslation(px, baseY, pz);
    zenMoteIM.setMatrixAt(i, _zenMoteMatrix);
    _tmpColor.set(col);
    zenMoteIM.setColorAt(i, _tmpColor);
  }
  zenMoteIM.instanceMatrix.needsUpdate = true;
  zenMoteIM.instanceColor.needsUpdate = true;
  scene.add(zenMoteIM);
}
createZenFloaters();

// Zen puck — soft glowing flat disc with aura
function createZenBall() {
  if (zenBallMesh) return;
  zenBallGeo = new THREE.CylinderGeometry(BALL_RADIUS * 1.0, BALL_RADIUS * 1.0, 0.09, 24);
  const zenBallMat = new THREE.MeshBasicMaterial({ color: 0xddeeff });
  zenBallMesh = new THREE.Mesh(zenBallGeo, zenBallMat);
  zenBallMesh.name = 'zenBall';
  zenBallMesh.visible = false;
  scene.add(zenBallMesh);

  // Soft outer aura — flat disc glow
  const auraGeo = new THREE.CylinderGeometry(BALL_RADIUS * 2.5, BALL_RADIUS * 2.5, 0.04, 20);
  const auraMat = new THREE.MeshBasicMaterial({
    color: 0xaaddff, transparent: true, opacity: 0.12,
    side: THREE.DoubleSide, depthWrite: false
  });
  zenBallAura = new THREE.Mesh(auraGeo, auraMat);
  zenBallAura.name = 'zenBallAura';
  zenBallAura.visible = false;
  scene.add(zenBallAura);

  zenCausticLight = new THREE.PointLight(0xaaddff, 0, 8);
  zenCausticLight.name = 'zenCausticLight';
  scene.add(zenCausticLight);
}
createZenBall();

// Zen floating petal particles — InstancedMesh: 50 petals in 1 draw call
const ZEN_PETAL_COUNT = 50;
let zenPetalIM = null;
const _zenPetalData = [];
const _zenPetalMatrix = new THREE.Matrix4();
const _zenPetalQuat = new THREE.Quaternion();
const _zenPetalEuler = new THREE.Euler();
function createZenPetals() {
  if (zenPetalIM) return;
  const petalGeo = new THREE.PlaneGeometry(0.06, 0.04);
  const petalMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false
  });
  zenPetalIM = new THREE.InstancedMesh(petalGeo, petalMat, ZEN_PETAL_COUNT);
  zenPetalIM.name = 'zenPetalInstanced';
  zenPetalIM.visible = false;
  zenPetalIM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ZEN_PETAL_COUNT * 3), 3);
  for (let i = 0; i < ZEN_PETAL_COUNT; i++) {
    const px = (Math.random() - 0.5) * 16;
    const py = Math.random() * 8 + 2;
    const pz = (Math.random() - 0.5) * 16;
    const rx = Math.random() * Math.PI;
    const ry = Math.random() * Math.PI;
    const rz = Math.random() * Math.PI;
    _zenPetalData.push({
      x: px, y: py, z: pz,
      rx: rx, ry: ry, rz: rz,
      baseY: py,
      fallSpeed: 0.2 + Math.random() * 0.3,
      swaySpeed: 0.5 + Math.random() * 0.5,
      swayAmp: 0.3 + Math.random() * 0.5,
      tumbleSpeed: 0.5 + Math.random() * 1.5,
      phase: Math.random() * Math.PI * 2,
    });
    _zenPetalEuler.set(rx, ry, rz);
    _zenPetalQuat.setFromEuler(_zenPetalEuler);
    _zenPetalMatrix.compose(
      _reusableVec.set(px, py, pz),
      _zenPetalQuat,
      _reusableVec2.set(1, 1, 1)
    );
    zenPetalIM.setMatrixAt(i, _zenPetalMatrix);
    _tmpColor.setHSL(0.55 + Math.random() * 0.1, 0.5, 0.75);
    zenPetalIM.setColorAt(i, _tmpColor);
  }
  zenPetalIM.instanceMatrix.needsUpdate = true;
  zenPetalIM.instanceColor.needsUpdate = true;
  scene.add(zenPetalIM);
}
createZenPetals();

// Zen orb particles — InstancedMesh: 30 orbs in 1 draw call
const ZEN_ORB_COUNT = 30;
let zenOrbIM = null;
const _zenOrbData = [];
const _zenOrbMatrix = new THREE.Matrix4();
function createZenOrbs() {
  if (zenOrbIM) return;
  const orbGeo = new THREE.SphereGeometry(0.025, 8, 8);
  const orbMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.3 });
  zenOrbIM = new THREE.InstancedMesh(orbGeo, orbMat, ZEN_ORB_COUNT);
  zenOrbIM.name = 'zenOrbInstanced';
  zenOrbIM.visible = false;
  zenOrbIM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ZEN_ORB_COUNT * 3), 3);
  for (let i = 0; i < ZEN_ORB_COUNT; i++) {
    const hue = 0.55 + Math.random() * 0.08;
    const angle = (i / ZEN_ORB_COUNT) * Math.PI * 2;
    const radius = 3 + Math.random() * 4;
    const height = 0.5 + Math.random() * 3;
    _zenOrbData.push({
      angle, radius, baseY: height,
      orbitSpeed: 0.05 + Math.random() * 0.1,
      bobSpeed: 0.3 + Math.random() * 0.5,
      bobAmp: 0.2 + Math.random() * 0.4,
      phase: Math.random() * Math.PI * 2,
    });
    _zenOrbMatrix.identity();
    _zenOrbMatrix.makeTranslation(Math.cos(angle) * radius, height, Math.sin(angle) * radius);
    zenOrbIM.setMatrixAt(i, _zenOrbMatrix);
    _tmpColor.setHSL(hue, 0.6, 0.8);
    zenOrbIM.setColorAt(i, _tmpColor);
  }
  zenOrbIM.instanceMatrix.needsUpdate = true;
  zenOrbIM.instanceColor.needsUpdate = true;
  scene.add(zenOrbIM);
}
createZenOrbs();

// Zen light rays overlay (CSS based soft beams)
function createZenLightRays() {
  if (zenLightRays) return;
  zenLightRays = document.createElement('div');
  zenLightRays.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9995; opacity: 0;
    transition: opacity 2s ease-in-out;
    background:
      linear-gradient(115deg, transparent 40%, rgba(170,221,255,0.03) 45%, transparent 50%),
      linear-gradient(135deg, transparent 50%, rgba(170,221,255,0.04) 55%, transparent 60%),
      linear-gradient(155deg, transparent 55%, rgba(170,221,255,0.025) 60%, transparent 65%);
  `;
  document.body.appendChild(zenLightRays);
}
createZenLightRays();

// Zen ripple rings (spawned on table bounce) — now spawns multiple concentric rings
function spawnZenRipple(pos) {
  if (!zenModeActive) return;
  // Spawn 3 concentric rings with staggered timing
  for (let r = 0; r < 3; r++) {
    const innerR = 0.05 + r * 0.02;
    const outerR = 0.08 + r * 0.02;
    const ringGeo = new THREE.RingGeometry(innerR, outerR, 16);
    const hue = 0.55 + r * 0.03;
    _tmpColor.setHSL(hue, 0.5, 0.75);
    const ringMat = new THREE.MeshBasicMaterial({
      color: _tmpColor.clone(), transparent: true, opacity: 0.5 - r * 0.1, side: THREE.DoubleSide
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.name = `zenRipple${zenRipples.length}`;
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(pos.x, TABLE_Y + TABLE_HEIGHT / 2 + 0.01, pos.z);
    scene.add(ring);
    zenRipples.push({ mesh: ring, life: 1.0 - r * 0.15, scale: 1 + r * 0.5 });
  }
  // Brief aura flash on bounce
  if (zenBallAura) {
    zenBallAura.material.opacity = 0.3;
  }
}

// Zen stylesheet
const zenStyleSheet = document.createElement('style');
zenStyleSheet.id = 'zen-mode-styles';
zenStyleSheet.textContent = `
  .zen-ui * {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 200 !important;
    letter-spacing: 2px !important;
  }
  .zen-ui span[style*="Instrument Serif"] {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 200 !important;
  }
`;
document.head.appendChild(zenStyleSheet);

function startZenMode() {
  if (zenModeActive) return;
  zenModeActive = true;
  zenTime = 0;

  // Show zen overlay (slow fade in)
  if (zenOverlay) zenOverlay.style.opacity = '1';
  if (zenLightRays) zenLightRays.style.opacity = '1';

  // Activate zen lights (brighter to illuminate table properly)
  zenWarmLight.intensity = 1.2;
  zenCoolLight.intensity = 0.7;
  zenAccentLight.intensity = 0.5;

  // Add subtle blue emissive to table so it glows
  tableMat.emissive.set(0x1a4a6a);
  tableMat.emissiveIntensity = 0.15;

  // Show floating motes
  if (zenMoteIM) { zenMoteIM.visible = true; zenMoteIM.material.opacity = 0.4; }

  // Show petals
  if (zenPetalIM) { zenPetalIM.visible = true; zenPetalIM.material.opacity = 0.35; }

  // Show orbs
  if (zenOrbIM) { zenOrbIM.visible = true; zenOrbIM.material.opacity = 0.3; }

  // Show zen ball + aura, hide normal ball
  if (zenBallMesh) zenBallMesh.visible = true;
  if (zenBallAura) zenBallAura.visible = true;
  if (zenCausticLight) zenCausticLight.intensity = 1.5;
  ball.visible = false;

  // Apply zen font to UI
  const uiEls = [uiContainer, statsBar, infoDiv, pointNotif, msgDiv];
  uiEls.forEach(el => el.classList.add('zen-ui'));

  // Soften table lines to muted baby blue
  lineMat.color.set(0x5599cc);
  edgeLineMat.color.set(0x5599cc);

  // Slightly reduce fog for more depth visibility
  scene.fog.density = 0.04;

  // Start ambient soundscape
  startZenAmbience();
}

function stopZenMode() {
  if (!zenModeActive) return;
  zenModeActive = false;

  // Fade out overlays
  if (zenOverlay) zenOverlay.style.opacity = '0';
  if (zenLightRays) zenLightRays.style.opacity = '0';

  // Kill lights
  zenWarmLight.intensity = 0;
  zenCoolLight.intensity = 0;
  zenAccentLight.intensity = 0;

  // Reset table emissive
  tableMat.emissive.set(0x000000);
  tableMat.emissiveIntensity = 0;

  // Hide floating motes
  if (zenMoteIM) { zenMoteIM.visible = false; }

  // Hide petals
  if (zenPetalIM) { zenPetalIM.visible = false; }

  // Hide orbs
  if (zenOrbIM) { zenOrbIM.visible = false; }

  // Hide zen ball + aura, show normal ball
  if (zenBallMesh) zenBallMesh.visible = false;
  if (zenBallAura) zenBallAura.visible = false;
  if (zenCausticLight) zenCausticLight.intensity = 0;
  ball.visible = true;

  // Clean up ripples
  zenRipples.forEach(r => { scene.remove(r.mesh); r.mesh.geometry.dispose(); r.mesh.material.dispose(); });
  zenRipples = [];

  // Reset background/fog
  scene.fog.color.set(0x000000);
  scene.fog.density = 0.055;
  if (scene.background && scene.background.isColor) scene.background.set(0x000000);

  // Remove zen font
  const uiEls = [uiContainer, statsBar, infoDiv, pointNotif, msgDiv];
  uiEls.forEach(el => el.classList.remove('zen-ui'));

  // Reset table lines
  lineMat.color.set(0xffffff);
  edgeLineMat.color.set(0xffffff);

  // Stop ambience
  stopZenAmbience();

  // Remove canvas filter
  renderer.domElement.style.filter = '';
}

function updateZenMode(dt) {
  if (!zenModeActive) return;
  zenTime += dt;

  // Gentle cool blue background breathing
  const breath = Math.sin(zenTime * 0.3) * 0.5 + 0.5;
  const breath2 = Math.sin(zenTime * 0.17) * 0.5 + 0.5;
  const bgR = 0.008 + breath * 0.004;
  const bgG = 0.02 + breath * 0.015 + breath2 * 0.005;
  const bgB = 0.05 + breath * 0.025 + breath2 * 0.01;
  _tmpColor.setRGB(bgR, bgG, bgB);
  if (scene.background && scene.background.isColor) scene.background.copy(_tmpColor);
  scene.fog.color.copy(_tmpColor);

  // Slow pulsing warm light
  zenWarmLight.intensity = 1.0 + Math.sin(zenTime * 0.5) * 0.3;
  zenWarmLight.position.y = 5 + Math.sin(zenTime * 0.25) * 0.5;

  // Cool light gentle orbit
  zenCoolLight.position.x = Math.sin(zenTime * 0.15) * 4;
  zenCoolLight.position.z = Math.cos(zenTime * 0.15) * 3;
  zenCoolLight.intensity = 0.5 + Math.sin(zenTime * 0.7) * 0.2;

  // Accent light subtle wander
  zenAccentLight.position.x = Math.cos(zenTime * 0.2) * 3;
  zenAccentLight.intensity = 0.4 + Math.sin(zenTime * 0.4 + 1) * 0.15;

  // Subtle table emissive pulse (gentle breathing glow)
  const tableGlow = 0.1 + Math.sin(zenTime * 0.35) * 0.05;
  tableMat.emissiveIntensity = tableGlow;

  // Update floating motes — instanced
  if (zenMoteIM && zenMoteIM.visible) {
    for (let i = 0; i < ZEN_MOTE_COUNT; i++) {
      const f = _zenMoteData[i];
      const t = zenTime * f.speed + f.phase;
      f.y = f.baseY + Math.sin(t) * 0.5;
      f.x += f.drift * dt * 0.3;
      f.z += Math.sin(t * 0.7 + i) * dt * 0.1;

      // Wrap around
      if (f.x > 7) f.x = -7;
      if (f.x < -7) f.x = 7;
      if (f.z > 7) f.z = -7;
      if (f.z < -7) f.z = 7;

      _zenMoteMatrix.identity();
      _zenMoteMatrix.makeTranslation(f.x, f.y, f.z);
      zenMoteIM.setMatrixAt(i, _zenMoteMatrix);
    }
    zenMoteIM.instanceMatrix.needsUpdate = true;
    // Gentle fade in/out breathing via global material opacity
    const moteFade = Math.sin(zenTime * 1.5) * 0.5 + 0.5;
    zenMoteIM.material.opacity = 0.15 + moteFade * 0.5;
  }

  // Update falling petals — instanced drifting, tumbling descent
  if (zenPetalIM && zenPetalIM.visible) {
    for (let i = 0; i < ZEN_PETAL_COUNT; i++) {
      const p = _zenPetalData[i];
      const t = zenTime * p.swaySpeed + p.phase;
      p.y -= p.fallSpeed * dt;
      p.x += Math.sin(t) * p.swayAmp * dt;
      p.z += Math.cos(t * 0.7 + i) * 0.15 * dt;

      // Tumbling rotation
      p.rx += p.tumbleSpeed * dt;
      p.rz += p.tumbleSpeed * 0.7 * dt;
      p.ry += p.tumbleSpeed * 0.3 * dt;

      // Respawn at top when fallen below
      if (p.y < -1) {
        p.y = 8 + Math.random() * 3;
        p.x = (Math.random() - 0.5) * 16;
        p.z = (Math.random() - 0.5) * 16;
      }

      _zenPetalEuler.set(p.rx, p.ry, p.rz);
      _zenPetalQuat.setFromEuler(_zenPetalEuler);
      _zenPetalMatrix.compose(
        _reusableVec.set(p.x, p.y, p.z),
        _zenPetalQuat,
        _reusableVec2.set(1, 1, 1)
      );
      zenPetalIM.setMatrixAt(i, _zenPetalMatrix);
    }
    zenPetalIM.instanceMatrix.needsUpdate = true;
    // Gentle opacity breathing via global material opacity
    const petalFade = Math.sin(zenTime * 0.8) * 0.5 + 0.5;
    zenPetalIM.material.opacity = 0.15 + petalFade * 0.35;
  }

  // Update orbiting luminous dots — instanced
  if (zenOrbIM && zenOrbIM.visible) {
    for (let i = 0; i < ZEN_ORB_COUNT; i++) {
      const o = _zenOrbData[i];
      o.angle += o.orbitSpeed * dt;
      const t = zenTime * o.bobSpeed + o.phase;
      const ox = Math.cos(o.angle) * o.radius;
      const oz = Math.sin(o.angle) * o.radius;
      const oy = o.baseY + Math.sin(t) * o.bobAmp;

      _zenOrbMatrix.identity();
      _zenOrbMatrix.makeTranslation(ox, oy, oz);
      zenOrbIM.setMatrixAt(i, _zenOrbMatrix);
    }
    zenOrbIM.instanceMatrix.needsUpdate = true;
    // Pulsing opacity via global material opacity
    const orbFade = Math.sin(zenTime * 1.3) * 0.5 + 0.5;
    zenOrbIM.material.opacity = 0.1 + orbFade * 0.45;
  }

  // Zen ball follows game ball — pure white glowing orb
  if (zenBallMesh) {
    zenBallMesh.position.copy(gameState.ballPos);
  }
  // Aura follows ball with gentle pulsing scale
  if (zenBallAura) {
    zenBallAura.position.copy(gameState.ballPos);
    const auraPulse = 1.0 + Math.sin(zenTime * 2.0) * 0.15;
    zenBallAura.scale.setScalar(auraPulse);
    zenBallAura.material.opacity = 0.08 + Math.sin(zenTime * 1.5) * 0.04;
  }
  // Caustic light follows ball
  if (zenCausticLight) {
    zenCausticLight.position.copy(gameState.ballPos);
    zenCausticLight.intensity = 1.2 + Math.sin(zenTime * 1.8) * 0.4;
  }

  // Update ripples (expand and fade) — spawn additional ring
  for (let i = zenRipples.length - 1; i >= 0; i--) {
    const r = zenRipples[i];
    r.life -= dt * 0.6;
    r.scale += dt * 3.5;
    r.mesh.scale.set(r.scale, r.scale, 1);
    r.mesh.material.opacity = r.life * 0.5;
    if (r.life <= 0) {
      scene.remove(r.mesh);
      r.mesh.geometry.dispose();
      r.mesh.material.dispose();
      zenRipples.splice(i, 1);
    }
  }

  // Animate light ray overlay opacity subtly
  if (zenLightRays) {
    const rayPulse = 0.6 + Math.sin(zenTime * 0.15) * 0.3;
    zenLightRays.style.opacity = String(rayPulse);
  }

  // Very subtle cool blue color filter on canvas
  const coolShift = 0.95 + Math.sin(zenTime * 0.2) * 0.03;
  renderer.domElement.style.filter = `saturate(1.15) brightness(${coolShift}) hue-rotate(10deg)`;
}

// --- Zen Ambient Soundscape ---
let zenAmbienceNodes = [];

function startZenAmbience() {
  stopZenAmbience();
  try {
    const ctx = getAudioCtx();

    // Soft drone pad — layered sine waves forming a warm chord
    // Each voice breathes independently with slow LFO modulation
    const chordFreqs = [130.81, 196.00, 261.63, 329.63]; // C3, G3, C4, E4
    const breathRates = [0.07, 0.11, 0.05, 0.09]; // Different LFO speeds (Hz) for organic feel
    const baseVol = 0.012; // Much quieter base volume per voice
    chordFreqs.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      // Volume breathing LFO — slow sine that modulates gain
      const lfo = ctx.createOscillator();
      const lfoGain = ctx.createGain();
      lfo.type = 'sine';
      lfo.frequency.setValueAtTime(breathRates[i], ctx.currentTime);
      lfoGain.gain.setValueAtTime(baseVol * 0.7, ctx.currentTime); // LFO depth — sweeps from ~30% to 100% of base
      lfo.connect(lfoGain);
      lfoGain.connect(gain.gain); // Modulate the voice gain

      // Subtle pitch drift LFO for organic detuning
      const pitchLfo = ctx.createOscillator();
      const pitchLfoGain = ctx.createGain();
      pitchLfo.type = 'sine';
      pitchLfo.frequency.setValueAtTime(0.03 + i * 0.01, ctx.currentTime);
      pitchLfoGain.gain.setValueAtTime(freq * 0.003, ctx.currentTime); // ~0.3% pitch drift
      pitchLfo.connect(pitchLfoGain);
      pitchLfoGain.connect(osc.frequency);

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime);

      filter.type = 'lowpass';
      filter.frequency.value = 600;
      filter.Q.value = 0.3;

      // Fade in gently to base volume
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(baseVol, ctx.currentTime + 4);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      lfo.start();
      pitchLfo.start();

      zenAmbienceNodes.push({ osc, gain, filter, lfo, lfoGain, pitchLfo, pitchLfoGain });
    });

    // Periodic gentle chime (every ~4s with randomness)
    function playChime() {
      if (!zenModeActive) return;
      const t = ctx.currentTime;

      // Bell tone — sine with fast decay
      const bellFreqs = [523.25, 659.25, 783.99, 1046.50, 1318.51, 1567.98];
      const freq = bellFreqs[Math.floor(Math.random() * bellFreqs.length)];

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.04, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 2.5);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 2.5);

      // Second harmonic overtone
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(freq * 2.02, t);  // Slightly detuned
      gain2.gain.setValueAtTime(0.015, t);
      gain2.gain.exponentialRampToValueAtTime(0.001, t + 1.8);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(t);
      osc2.stop(t + 1.8);

      // Third partial
      const osc3 = ctx.createOscillator();
      const gain3 = ctx.createGain();
      osc3.type = 'sine';
      osc3.frequency.setValueAtTime(freq * 3.01, t);
      gain3.gain.setValueAtTime(0.008, t);
      gain3.gain.exponentialRampToValueAtTime(0.001, t + 1.2);
      osc3.connect(gain3);
      gain3.connect(ctx.destination);
      osc3.start(t);
      osc3.stop(t + 1.2);
    }

    // Schedule periodic chimes
    zenAmbienceInterval = setInterval(() => {
      if (!zenModeActive) return;
      if (Math.random() < 0.6) playChime();
    }, 4000 + Math.random() * 2000);

    // Play first chime after a short delay
    setTimeout(() => { if (zenModeActive) playChime(); }, 2000);

  } catch (e) { /* audio not available */ }
}

function stopZenAmbience() {
  // Quick fade out drone pads and all LFO nodes
  zenAmbienceNodes.forEach(n => {
    try {
      const ctx = n.osc.context;
      const t = ctx.currentTime;
      n.gain.gain.cancelScheduledValues(t);
      n.gain.gain.setValueAtTime(n.gain.gain.value, t);
      n.gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
      n.osc.stop(t + 0.3);
      if (n.lfo) n.lfo.stop(t + 0.3);
      if (n.pitchLfo) n.pitchLfo.stop(t + 0.3);
    } catch (e) {}
  });
  zenAmbienceNodes = [];

  if (zenAmbienceInterval) {
    clearInterval(zenAmbienceInterval);
    zenAmbienceInterval = null;
  }
}

// Zen paddle hit sound — whisper-soft chime touch
function playZenHitSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Gentle pentatonic chime — randomly pick a note
    const chimeFreqs = [523.25, 587.33, 659.25, 783.99, 880.00];
    const freq = chimeFreqs[Math.floor(Math.random() * chimeFreqs.length)];

    // Primary tone — very quiet sine with long fade
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 1200;
    lp.Q.value = 0.5;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    osc.detune.setValueAtTime(Math.random() * 4 - 2, t);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.025, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 1.2);
    osc.connect(lp);
    lp.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 1.3);

    // Faint harmonic fifth above — barely audible shimmer
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(freq * 1.5, t);
    gain2.gain.setValueAtTime(0, t);
    gain2.gain.linearRampToValueAtTime(0.008, t + 0.04);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 1.5);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 1.6);
  } catch (e) {}
}

// Zen bounce sound — single raindrop on still water
function playZenBounceSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Tiny water drop — very soft, gentle descending pitch
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 800;
    osc.type = 'sine';
    osc.frequency.setValueAtTime(600 + Math.random() * 150, t);
    osc.frequency.exponentialRampToValueAtTime(250, t + 0.15);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.018, t + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
    osc.connect(lp);
    lp.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.55);
  } catch (e) {}
}

// Zen score sound — distant singing bowl, very gentle
function playZenScoreSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Single singing bowl fundamental — soft and long
    const fundamentals = [261.63, 329.63, 392.00];
    const fund = fundamentals[Math.floor(Math.random() * fundamentals.length)];

    // Just 2 gentle partials — fundamental + one harmonic
    const partials = [1, 2.76];
    const amps = [0.035, 0.012];
    const decays = [4.0, 2.5];

    partials.forEach((ratio, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 1500;
      osc.type = 'sine';
      osc.frequency.setValueAtTime(fund * ratio, t);
      osc.detune.setValueAtTime(Math.random() * 4 - 2, t);
      gain.gain.setValueAtTime(0, t);
      // Slow fade in — like the sound appears from silence
      gain.gain.linearRampToValueAtTime(amps[i], t + 0.08);
      gain.gain.exponentialRampToValueAtTime(0.001, t + decays[i]);
      osc.connect(lp);
      lp.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t);
      osc.stop(t + decays[i] + 0.1);
    });
  } catch (e) {}
}

// --- Inferno Mode State ---
let infernoModeActive = false;
let infernoTime = 0;
let infernoOverlay = null;
let infernoHeatOverlay = null;
let infernoEmbers = [];
let infernoFireLight1 = null;
let infernoFireLight2 = null;
let infernoFireLight3 = null;
let infernoBallMesh = null;
let infernoBallGeo = null;
let infernoFireParticles = [];
const infernoFireActiveSet = new Set();
let infernoCrackleInterval = null;
let infernoRumbleNodes = [];

// Create inferno overlays
function createInfernoOverlays() {
  if (infernoOverlay) return;

  // Fiery vignette — deep red/orange edges
  infernoOverlay = document.createElement('div');
  infernoOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9996; opacity: 0;
    transition: opacity 0.8s ease-in-out;
    background: radial-gradient(ellipse at center, transparent 20%, rgba(80,10,0,0.3) 60%, rgba(40,0,0,0.7) 100%);
  `;
  document.body.appendChild(infernoOverlay);

  // Heat distortion shimmer overlay
  infernoHeatOverlay = document.createElement('div');
  infernoHeatOverlay.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9997; opacity: 0;
    transition: opacity 0.8s;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 4px,
      rgba(255,100,0,0.02) 4px,
      rgba(255,100,0,0.02) 8px
    );
    mix-blend-mode: screen;
  `;
  document.body.appendChild(infernoHeatOverlay);
}
createInfernoOverlays();

// Inferno CSS animations
const infernoStyleSheet = document.createElement('style');
infernoStyleSheet.id = 'inferno-mode-styles';
infernoStyleSheet.textContent = `
  @keyframes infernoHeatShimmer {
    0% { background-position: 0 0; }
    100% { background-position: 0 40px; }
  }
  .inferno-heat-anim {
    animation: infernoHeatShimmer 0.6s linear infinite;
  }
`;
document.head.appendChild(infernoStyleSheet);

// Inferno fire lights
infernoFireLight1 = new THREE.PointLight(0xff4400, 0, 18);
infernoFireLight1.name = 'infernoFireLight1';
infernoFireLight1.position.set(-2, 3, -1);
scene.add(infernoFireLight1);

infernoFireLight2 = new THREE.PointLight(0xff8800, 0, 18);
infernoFireLight2.name = 'infernoFireLight2';
infernoFireLight2.position.set(2, 4, 1);
scene.add(infernoFireLight2);

infernoFireLight3 = new THREE.PointLight(0xff2200, 0, 12);
infernoFireLight3.name = 'infernoFireLight3';
infernoFireLight3.position.set(0, 2, 0);
scene.add(infernoFireLight3);

// Ember floaters — InstancedMesh: 80 embers in 1 draw call
const INFERNO_EMBER_COUNT = 80;
let infernoEmberIM = null;
const _infernoEmberData = [];
function createInfernoEmbers() {
  if (infernoEmberIM) return;
  const emberGeo = new THREE.SphereGeometry(0.02, 4, 4);
  const emberMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.7 });
  infernoEmberIM = new THREE.InstancedMesh(emberGeo, emberMat, INFERNO_EMBER_COUNT);
  infernoEmberIM.name = 'infernoEmberInstanced';
  infernoEmberIM.visible = false;
  infernoEmberIM.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(INFERNO_EMBER_COUNT * 3), 3);
  for (let i = 0; i < INFERNO_EMBER_COUNT; i++) {
    const px = (Math.random() - 0.5) * 14;
    const py = Math.random() * -1;
    const pz = (Math.random() - 0.5) * 14;
    const resetY = -2 + Math.random() * -2;
    _infernoEmberData.push({
      x: px, y: py, z: pz,
      speed: 0.5 + Math.random() * 1.5,
      drift: (Math.random() - 0.5) * 0.5,
      phase: Math.random() * Math.PI * 2,
      resetY: resetY,
      maxY: 6 + Math.random() * 4,
    });
    _reusableMatrix.identity();
    _reusableMatrix.makeTranslation(px, py, pz);
    infernoEmberIM.setMatrixAt(i, _reusableMatrix);
    const hue = 0.02 + Math.random() * 0.08;
    _tmpColor.setHSL(hue, 1, 0.5 + Math.random() * 0.3);
    infernoEmberIM.setColorAt(i, _tmpColor);
  }
  infernoEmberIM.instanceMatrix.needsUpdate = true;
  infernoEmberIM.instanceColor.needsUpdate = true;
  scene.add(infernoEmberIM);
}
createInfernoEmbers();

// Inferno puck — flat glowing lava disc
function createInfernoBall() {
  if (infernoBallMesh) return;
  infernoBallGeo = new THREE.CylinderGeometry(BALL_RADIUS * 1.2, BALL_RADIUS * 1.2, 0.1, 16);
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffcc00, emissive: 0xff4400, emissiveIntensity: 2.0,
    roughness: 0.25, metalness: 0.0,
  });
  infernoBallMesh = new THREE.Mesh(infernoBallGeo, mat);
  infernoBallMesh.name = 'infernoBall';
  infernoBallMesh.visible = false;
  scene.add(infernoBallMesh);
}
createInfernoBall();

// Inferno fire trail particles — POOLED for performance
const INFERNO_FIRE_POOL_SIZE = 60;
const infernoFirePool = [];
const infernoFireGeo = new THREE.SphereGeometry(0.05, 4, 4);
const _fireVelDelta = new THREE.Vector3();
for (let fi = 0; fi < INFERNO_FIRE_POOL_SIZE; fi++) {
  const mat = new THREE.MeshBasicMaterial({ color: 0xff6600, transparent: true, opacity: 0 });
  const m = new THREE.Mesh(infernoFireGeo, mat);
  m.name = `infernoFire${fi}`;
  m.visible = false;
  m.frustumCulled = false;
  scene.add(m);
  infernoFirePool.push({ mesh: m, vel: new THREE.Vector3(), life: 0, maxLife: 0, active: false });
}
let infernoFirePoolIdx = 0;

function spawnInfernoFireTrail(pos) {
  if (!infernoModeActive) return;
  const count = 3;
  for (let i = 0; i < count; i++) {
    const p = infernoFirePool[infernoFirePoolIdx];
    infernoFirePoolIdx = (infernoFirePoolIdx + 1) % INFERNO_FIRE_POOL_SIZE;
    const hue = 0.02 + Math.random() * 0.1;
    const lightness = 0.4 + Math.random() * 0.3;
    p.mesh.material.color.setHSL(hue, 1, lightness);
    p.mesh.material.opacity = 0.9;
    p.mesh.position.set(
      pos.x + (Math.random() - 0.5) * 0.15,
      pos.y + (Math.random() - 0.5) * 0.15,
      pos.z + (Math.random() - 0.5) * 0.15
    );
    const size = 0.6 + Math.random() * 1.2;
    p.mesh.scale.setScalar(size);
    p.mesh.visible = true;
    p.vel.set(
      (Math.random() - 0.5) * 0.8,
      1.0 + Math.random() * 2.0,
      (Math.random() - 0.5) * 0.8
    );
    const lifeVal = 0.4 + Math.random() * 0.4;
    p.life = lifeVal;
    p.maxLife = lifeVal;
    p.active = true;
    if (!infernoFireActiveSet.has(p)) { infernoFireActiveSet.add(p); infernoFireParticles.push(p); }
  }
}

function updateInfernoFireParticles(dt) {
  let writeIdx = 0;
  for (let i = 0; i < infernoFireParticles.length; i++) {
    const p = infernoFireParticles[i];
    if (!p.active) { infernoFireActiveSet.delete(p); continue; }
    _fireVelDelta.copy(p.vel).multiplyScalar(dt);
    p.mesh.position.add(_fireVelDelta);
    p.vel.y += dt * 2;
    p.life -= dt;
    const t = p.life / p.maxLife;
    p.mesh.material.opacity = t * 0.8;
    p.mesh.scale.setScalar(0.3 + t * 0.7);
    const hue = THREE.MathUtils.lerp(0.0, 0.1, t);
    p.mesh.material.color.setHSL(hue, 1, THREE.MathUtils.lerp(0.2, 0.6, t));
    if (p.life <= 0) {
      p.mesh.visible = false;
      p.mesh.material.opacity = 0;
      p.active = false;
      infernoFireActiveSet.delete(p);
    } else {
      infernoFireParticles[writeIdx++] = p;
    }
  }
  infernoFireParticles.length = writeIdx;
}

function startInfernoMode() {
  if (infernoModeActive) return;
  infernoModeActive = true;
  infernoTime = 0;

  // Show overlays
  if (infernoOverlay) infernoOverlay.style.opacity = '1';
  if (infernoHeatOverlay) { infernoHeatOverlay.style.opacity = '1'; infernoHeatOverlay.classList.add('inferno-heat-anim'); }

  // Activate fire lights
  infernoFireLight1.intensity = 2.0;
  infernoFireLight2.intensity = 1.5;
  infernoFireLight3.intensity = 1.0;

  // Show instanced embers
  if (infernoEmberIM) {
    infernoEmberIM.visible = true;
    infernoEmberIM.material.opacity = 0.7;
    // Reset positions to starting Y
    for (let i = 0; i < INFERNO_EMBER_COUNT; i++) {
      const e = _infernoEmberData[i];
      e.y = e.resetY;
      e.x = (Math.random() - 0.5) * 14;
      e.z = (Math.random() - 0.5) * 14;
      _reusableMatrix.identity();
      _reusableMatrix.makeTranslation(e.x, e.y, e.z);
      infernoEmberIM.setMatrixAt(i, _reusableMatrix);
    }
    infernoEmberIM.instanceMatrix.needsUpdate = true;
  }

  // Show inferno ball, hide normal
  if (infernoBallMesh) infernoBallMesh.visible = true;
  ball.visible = false;

  // Make table lines fiery
  lineMat.color.set(0xff6600);
  edgeLineMat.color.set(0xff4400);

  // Start fire crackle sound loop
  startInfernoCrackle();

  // Start low rumble
  startInfernoRumble();
}

function stopInfernoMode() {
  if (!infernoModeActive) return;
  infernoModeActive = false;

  // Hide overlays
  if (infernoOverlay) infernoOverlay.style.opacity = '0';
  if (infernoHeatOverlay) { infernoHeatOverlay.style.opacity = '0'; infernoHeatOverlay.classList.remove('inferno-heat-anim'); }

  // Kill fire lights
  infernoFireLight1.intensity = 0;
  infernoFireLight2.intensity = 0;
  infernoFireLight3.intensity = 0;

  // Hide instanced embers
  if (infernoEmberIM) { infernoEmberIM.visible = false; infernoEmberIM.material.opacity = 0; }

  // Hide inferno ball, show normal
  if (infernoBallMesh) infernoBallMesh.visible = false;
  ball.visible = true;

  // Clean up fire trail particles
  infernoFireParticles.forEach(p => {
    p.mesh.visible = false;
    p.mesh.material.opacity = 0;
    p.active = false;
  });
  infernoFireParticles.length = 0;
  infernoFireActiveSet.clear();

  // Reset background/fog
  scene.fog.color.set(0x000000);
  if (scene.background && scene.background.isColor) scene.background.set(0x000000);

  // Reset table lines
  lineMat.color.set(0xffffff);
  edgeLineMat.color.set(0xffffff);

  // Stop sounds
  stopInfernoCrackle();
  stopInfernoRumble();

  // Remove canvas filter
  renderer.domElement.style.filter = '';
}

function updateInfernoMode(dt) {
  if (!infernoModeActive) return;
  infernoTime += dt;

  // Flickering dark red/orange background — like the inside of a furnace
  const flicker = 0.8 + Math.random() * 0.2;
  const pulse = Math.sin(infernoTime * 1.5) * 0.5 + 0.5;
  const bgR = (0.04 + pulse * 0.02) * flicker;
  const bgG = (0.008 + pulse * 0.005) * flicker;
  const bgB = 0.002 * flicker;
  _tmpColor.setRGB(bgR, bgG, bgB);
  if (scene.background && scene.background.isColor) scene.background.copy(_tmpColor);
  scene.fog.color.copy(_tmpColor);

  // Fire lights flicker intensely
  infernoFireLight1.intensity = 1.5 + Math.sin(infernoTime * 12) * 0.8 + Math.random() * 0.5;
  infernoFireLight1.position.x = -2 + Math.sin(infernoTime * 3) * 1;
  infernoFireLight1.position.y = 2.5 + Math.sin(infernoTime * 4) * 0.5;

  infernoFireLight2.intensity = 1.2 + Math.sin(infernoTime * 9 + 1) * 0.6 + Math.random() * 0.4;
  infernoFireLight2.position.x = 2 + Math.cos(infernoTime * 2.5) * 1;
  infernoFireLight2.position.y = 3 + Math.cos(infernoTime * 3.5) * 0.5;

  infernoFireLight3.intensity = 0.8 + Math.sin(infernoTime * 15 + 2) * 0.5 + Math.random() * 0.3;
  infernoFireLight3.position.set(
    Math.sin(infernoTime * 2) * 2,
    1.5 + Math.sin(infernoTime * 5) * 0.5,
    Math.cos(infernoTime * 1.8) * 2
  );

  // Update embers — instanced: float upward, flicker, color shift
  if (infernoEmberIM && infernoEmberIM.visible) {
    const eColors = infernoEmberIM.instanceColor.array;
    for (let i = 0; i < INFERNO_EMBER_COUNT; i++) {
      const e = _infernoEmberData[i];
      const t = infernoTime * e.speed + e.phase;
      e.y += e.speed * dt * 1.5;
      e.x += Math.sin(t * 2 + i) * dt * e.drift;
      e.z += Math.cos(t * 1.5 + i * 0.7) * dt * e.drift;

      // Reset when too high
      if (e.y > e.maxY) {
        e.x = (Math.random() - 0.5) * 14;
        e.y = e.resetY;
        e.z = (Math.random() - 0.5) * 14;
      }

      _reusableMatrix.identity();
      _reusableMatrix.makeTranslation(e.x, e.y, e.z);
      infernoEmberIM.setMatrixAt(i, _reusableMatrix);

      // Shift color: brighter when lower, redder when higher
      const heightT = THREE.MathUtils.clamp((e.y - e.resetY) / (e.maxY - e.resetY), 0, 1);
      const hue = THREE.MathUtils.lerp(0.08, 0.0, heightT);
      const light = THREE.MathUtils.lerp(0.6, 0.3, heightT);
      _tmpColor.setHSL(hue, 1, light);
      eColors[i * 3] = _tmpColor.r;
      eColors[i * 3 + 1] = _tmpColor.g;
      eColors[i * 3 + 2] = _tmpColor.b;
    }
    infernoEmberIM.instanceMatrix.needsUpdate = true;
    infernoEmberIM.instanceColor.needsUpdate = true;
    // Global flicker opacity
    const flickGlobal = Math.sin(infernoTime * 10) * 0.15 + 0.65;
    infernoEmberIM.material.opacity = flickGlobal;
  }

  // Inferno ball follows game ball
  if (infernoBallMesh) {
    infernoBallMesh.position.copy(gameState.ballPos);
    infernoBallMesh.rotation.x += dt * 8;
    infernoBallMesh.rotation.y += dt * 6;

    // Pulsing emissive — like a burning coal
    const emPulse = 1.5 + Math.sin(infernoTime * 10) * 0.5 + Math.random() * 0.3;
    infernoBallMesh.material.emissiveIntensity = emPulse;

    // Cycle emissive between orange and red
    const h = 0.03 + Math.sin(infernoTime * 5) * 0.03;
    infernoBallMesh.material.emissive.setHSL(h, 1, 0.5);

    // Scale pulsation
    const scalePulse = 1.0 + Math.sin(infernoTime * 8) * 0.05;
    infernoBallMesh.scale.setScalar(scalePulse);

    // Spawn fire trail behind the ball while it's moving
    if (!gameState.paused) {
      spawnInfernoFireTrail(gameState.ballPos);
    }
  }

  // Update fire trail particles
  updateInfernoFireParticles(dt);

  // Heat distortion on canvas — subtle wavy filter
  const heatWave = Math.sin(infernoTime * 3) * 0.3;
  renderer.domElement.style.filter = `brightness(${1.0 + Math.random() * 0.04}) contrast(${1.05 + heatWave * 0.02})`;

  // Occasional intense flicker (like a flame burst)
  if (Math.random() < 0.01) {
    renderer.domElement.style.filter = `brightness(1.15) saturate(1.3)`;
    setTimeout(() => {
      if (infernoModeActive) renderer.domElement.style.filter = '';
    }, 60);
  }
}

// --- Inferno Audio ---
function startInfernoCrackle() {
  stopInfernoCrackle();
  try {
    const ctx = getAudioCtx();
    function playCrackle() {
      if (!infernoModeActive) return;
      const t = ctx.currentTime;

      // Fire crackle — short noise bursts
      const noise = ctx.createBufferSource();
      const bufSize = Math.floor(ctx.sampleRate * (0.02 + Math.random() * 0.04));
      const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < bufSize; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.2));
      }
      noise.buffer = buf;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.04 + Math.random() * 0.03, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 2000 + Math.random() * 4000;
      bp.Q.value = 2;
      noise.connect(bp);
      bp.connect(gain);
      gain.connect(ctx.destination);
      noise.start(t);
      noise.stop(t + 0.06);
    }

    // Random crackle timing for natural fire sound
    function scheduleCrackle() {
      if (!infernoModeActive) return;
      playCrackle();
      const delay = 80 + Math.random() * 300;
      infernoCrackleInterval = setTimeout(scheduleCrackle, delay);
    }
    scheduleCrackle();
  } catch (e) {}
}

function stopInfernoCrackle() {
  if (infernoCrackleInterval) {
    clearTimeout(infernoCrackleInterval);
    infernoCrackleInterval = null;
  }
}

function startInfernoRumble() {
  stopInfernoRumble();
  try {
    const ctx = getAudioCtx();
    // Low frequency rumble — like a furnace
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(35, ctx.currentTime);
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 2);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    infernoRumbleNodes.push({ osc, gain });

    // Second harmonic
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(70, ctx.currentTime);
    gain2.gain.setValueAtTime(0, ctx.currentTime);
    gain2.gain.linearRampToValueAtTime(0.03, ctx.currentTime + 2);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start();
    infernoRumbleNodes.push({ osc: osc2, gain: gain2 });
  } catch (e) {}
}

function stopInfernoRumble() {
  infernoRumbleNodes.forEach(n => {
    try {
      const ctx = n.osc.context;
      n.gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);
      n.osc.stop(ctx.currentTime + 0.6);
    } catch (e) {}
  });
  infernoRumbleNodes = [];
}

// Inferno paddle hit sound — explosive fire burst
function playInfernoHitSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Deep explosion bass
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(150, t);
    osc.frequency.exponentialRampToValueAtTime(30, t + 0.2);
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.3);

    // Fire whoosh — filtered noise
    const noise = ctx.createBufferSource();
    const bufSize = Math.floor(ctx.sampleRate * 0.15);
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.3));
    noise.buffer = buf;
    const nGain = ctx.createGain();
    nGain.gain.setValueAtTime(0.15, t);
    nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 1500;
    bp.Q.value = 1;
    noise.connect(bp);
    bp.connect(nGain);
    nGain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.15);

    // High sizzle
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sawtooth';
    osc2.frequency.setValueAtTime(3000, t);
    osc2.frequency.exponentialRampToValueAtTime(500, t + 0.08);
    gain2.gain.setValueAtTime(0.04, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.1);
  } catch (e) {}
}

// Inferno bounce sound — sizzling impact
function playInfernoBounceSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Impact thud
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(200, t);
    osc.frequency.exponentialRampToValueAtTime(60, t + 0.08);
    gain.gain.setValueAtTime(0.2, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.1);

    // Sizzle
    const noise = ctx.createBufferSource();
    const bufSize = Math.floor(ctx.sampleRate * 0.08);
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.15));
    noise.buffer = buf;
    const nGain = ctx.createGain();
    nGain.gain.setValueAtTime(0.08, t);
    nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 4000;
    noise.connect(hp);
    hp.connect(nGain);
    nGain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.08);
  } catch (e) {}
}

// Inferno score sound — fiery explosion fanfare
function playInfernoScoreSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Deep boom
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(80, t);
    osc.frequency.exponentialRampToValueAtTime(20, t + 0.5);
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.6);

    // Rising fire whoosh
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sawtooth';
    osc2.frequency.setValueAtTime(200, t);
    osc2.frequency.exponentialRampToValueAtTime(2000, t + 0.3);
    gain2.gain.setValueAtTime(0.06, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(800, t);
    lp.frequency.exponentialRampToValueAtTime(4000, t + 0.3);
    osc2.connect(lp);
    lp.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.4);

    // Crackle burst
    const noise = ctx.createBufferSource();
    const bufSize = Math.floor(ctx.sampleRate * 0.3);
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufSize * 0.3));
    noise.buffer = buf;
    const nGain = ctx.createGain();
    nGain.gain.setValueAtTime(0.1, t + 0.1);
    nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
    noise.connect(nGain);
    nGain.connect(ctx.destination);
    noise.start(t + 0.1);
    noise.stop(t + 0.4);
  } catch (e) {}
}

function updateThemeUI() {
  // Update scoreboard name colors
  const playerNameEl = document.querySelector('#playerServeDot')?.parentElement?.querySelector('span:nth-child(2)');
  const aiNameEl = document.querySelector('#aiServeDot')?.parentElement?.querySelector('span:nth-child(2)');
  if (playerNameEl) playerNameEl.style.color = activePlayerColor;
  if (aiNameEl) aiNameEl.style.color = activeAiColor;

  // Update serve dots
  const playerDot = document.getElementById('playerServeDot');
  const aiDot = document.getElementById('aiServeDot');
  if (playerDot) playerDot.style.background = activePlayerColor;
  if (aiDot) aiDot.style.background = activeAiColor;

  // Update set pips
  updateSetDisplay();
}

function applyLerpedTheme(t) {
  // Smooth easing
  const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  playerPaddle.children.forEach(child => {
    if (child.name.includes('Head')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.playerHead).lerp(themeLerpTo.playerHead, e));
    else if (child.name.includes('RubberFront')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.playerRubber).lerp(themeLerpTo.playerRubber, e));
    else if (child.name.includes('RubberBack')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.playerBack).lerp(themeLerpTo.playerBack, e));
  });
  aiPaddle.children.forEach(child => {
    if (child.name.includes('Head')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.aiHead).lerp(themeLerpTo.aiHead, e));
    else if (child.name.includes('RubberFront')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.aiRubber).lerp(themeLerpTo.aiRubber, e));
    else if (child.name.includes('RubberBack')) child.material.color.copy(themeLerpCurrent.copy(themeLerpFrom.aiBack).lerp(themeLerpTo.aiBack, e));
  });
  tableMat.color.copy(themeLerpCurrent.copy(themeLerpFrom.tableSurface).lerp(themeLerpTo.tableSurface, e));
  tableMat.clearcoat = THREE.MathUtils.lerp(themeLerpFrom.tableClearcoat, themeLerpTo.tableClearcoat, e);
  tableMat.roughness = THREE.MathUtils.lerp(themeLerpFrom.tableRoughness, themeLerpTo.tableRoughness, e);
  tableMat.metalness = THREE.MathUtils.lerp(themeLerpFrom.tableMetalness, themeLerpTo.tableMetalness, e);

  // Ball color lerp
  ballMat.color.copy(themeLerpCurrent.copy(themeLerpFrom.ballColor).lerp(themeLerpTo.ballColor, e));
  ballMat.emissive.copy(themeLerpCurrent.copy(themeLerpFrom.ballEmissive).lerp(themeLerpTo.ballEmissive, e));
  // Update trail colors to match ball
  trailMeshes.forEach(m => m.material.color.copy(ballMat.color));
}

function applyTheme(idx) {
  currentTheme = idx;
  const theme = themes[idx];

  // Update active colors for particles and UI
  activePlayerColor = theme.player.head;
  activeAiColor = theme.ai.head;
  updateThemeUI();

  // Capture current state as "from"
  captureCurrentColors(themeLerpFrom);
  // Set target state as "to"
  setTargetColors(themeLerpTo, theme);

  // Start lerp
  themeLerpProgress = 0;
  themeLerpActive = true;

  // Stop all inactive modes FIRST so they don't overwrite the new mode's settings
  if (!theme.party) stopPartyMode();
  if (!theme.retro) stopRetroMode();
  if (!theme.zen) stopZenMode();
  if (!theme.inferno) stopInfernoMode();

  // Then start the active mode
  if (theme.party) startPartyMode();
  if (theme.retro) startRetroMode();
  if (theme.zen) startZenMode();
  if (theme.inferno) startInfernoMode();
}

// --- Theme Buttons (always visible) ---

const themesPanel = document.createElement('div');
themesPanel.style.cssText = `
  position: fixed; bottom: 56px; left: 0; width: 100%;
  padding: 0; z-index: 150; pointer-events: none;
  font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 12px;
  display: flex;
  overflow: hidden;
  background: transparent;
`;
document.body.appendChild(themesPanel);

// Theme cards container
const themesBody = document.createElement('div');
themesBody.style.cssText = `
  display: flex; justify-content: center; align-items: center; gap: 8px;
  padding: 12px 24px 14px; overflow-x: auto; width: 100%; pointer-events: none;
`;
themesPanel.appendChild(themesBody);

const themeCards = [];
themes.forEach((theme, idx) => {
  const card = document.createElement('div');
  const isActive = idx === 0;
  card.style.cssText = `
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    cursor: pointer; pointer-events: all; flex-shrink: 0;
    padding: 8px 12px; border-radius: 10px;
    border: 1px solid ${isActive ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.08)'};
    background: ${isActive ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.5)'};
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    transition: border-color 0.25s, background 0.25s, transform 0.15s;
  `;

  // Theme name
  const nameEl = document.createElement('div');
  nameEl.style.cssText = `
    font-family: 'Instrument Serif', serif; font-style: italic; font-size: 11px;
    color: ${isActive ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.3)'};
    letter-spacing: 0.3px; white-space: nowrap; transition: color 0.25s;
  `;
  nameEl.textContent = theme.name;
  card.appendChild(nameEl);

  card.addEventListener('mouseenter', () => {
    card.style.transform = 'scale(1.06)';
    if (idx !== currentTheme) card.style.borderColor = 'rgba(255,255,255,0.15)';
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = 'scale(1)';
    if (idx !== currentTheme) card.style.borderColor = 'rgba(255,255,255,0.05)';
  });

  card.addEventListener('click', () => {
    applyTheme(idx);
    // Update all cards
    themeCards.forEach((c, i) => {
      const active = i === idx;
      c.card.style.borderColor = active ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.08)';
      c.card.style.background = active ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.5)';
      c.nameEl.style.color = active ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.3)';
    });
  });

  themesBody.appendChild(card);
  themeCards.push({ card, nameEl });
});

// Apply Arctic theme immediately on load so mallets start blue/red
{
  const arctic = themes[0];
  applyPaddleSkin(playerPaddle, arctic.player);
  applyPaddleSkin(aiPaddle, arctic.ai);
  applyTableSkin(arctic.table);
  ballMat.color.set(arctic.ball.color);
  ballMat.emissive.set(arctic.ball.emissive);
  trailMeshes.forEach(m => m.material.color.set(arctic.ball.color));
  activePlayerColor = arctic.player.head;
  activeAiColor = arctic.ai.head;
}

// --- Performance section ---
const perfSec = createSection('Performance');
const fpsRow = document.createElement('div');
fpsRow.style.cssText = 'display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;';
const fpsLabel = document.createElement('span');
fpsLabel.style.cssText = 'color: #aaa; font-size: 11px;';
fpsLabel.textContent = 'FPS';
const fpsValue = document.createElement('span');
fpsValue.id = 'settings-fps';
fpsValue.style.cssText = "color: #0f0; font-size: 11px; font-family: 'SF Mono', 'Consolas', monospace; font-variant-numeric: tabular-nums;";
fpsValue.textContent = '--';
fpsRow.appendChild(fpsLabel);
fpsRow.appendChild(fpsValue);
perfSec.appendChild(fpsRow);

// --- Gameplay section ---
const gpSec = createSection('Gameplay');
createSlider(gpSec, 'AI Speed', 2, 12, 0.5, 5.5, (v) => { window._aiSpeed = v; });
createSlider(gpSec, 'Puck Speed', 4, 16, 0.5, 8, (v) => { window._ballSpeedBase = v; });
createSlider(gpSec, 'Friction', -2, -0.1, 0.1, -0.5, (v) => { window._gravity = v; });
window._aiSpeed = 5.5;
window._ballSpeedBase = BALL_SPEED_BASE;
window._gravity = GRAVITY;

// Score display
const scoreDiv = document.createElement('div');
scoreDiv.style.cssText = `
  display: flex; justify-content: center; gap: 48px; padding: 24px;
  font-size: 28px; font-weight: 400; color: #fff; letter-spacing: 0.5px;
`;
scoreDiv.style.cssText = `
  display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
  padding: 24px; font-size: 28px; font-weight: 400; color: #fff; letter-spacing: 0.5px;
  max-width: 600px; margin: 0 auto;
`;
scoreDiv.innerHTML = `
  <div style="display:flex;flex-direction:column;align-items:center;gap:4px;justify-self:end;padding-right:24px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span id="playerServeDot" style="width:8px;height:8px;border-radius:50%;background:#1a88ff;display:none;animation:servePulse 1.2s ease-in-out infinite;flex-shrink:0;"></span>
      <span style="color:#1a88ff;font-family:'Instrument Serif',serif;font-size:22px;font-style:italic;letter-spacing:0.5px;">You</span>
      <span id="playerScore" style="color:#fff;font-family:'Instrument Serif',serif;font-size:48px;line-height:1;min-width:56px;text-align:center;font-variant-numeric:tabular-nums;">0</span>
    </div>
    <div id="playerSets" style="display:flex;gap:6px;"></div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;width:80px;">
    <div style="color:rgba(255,255,255,0.2);font-family:'Instrument Serif',serif;font-size:26px;font-style:italic;">vs</div>
    <div id="setLabel" style="color:rgba(255,255,255,0.2);font-family:'Instrument Serif',serif;font-size:13px;font-style:italic;letter-spacing:0.5px;white-space:nowrap;">Game 1</div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:4px;justify-self:start;padding-left:24px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span id="aiScore" style="color:#fff;font-family:'Instrument Serif',serif;font-size:48px;line-height:1;min-width:56px;text-align:center;font-variant-numeric:tabular-nums;">0</span>
      <span style="color:#ff3355;font-family:'Instrument Serif',serif;font-size:22px;font-style:italic;letter-spacing:0.5px;">CPU</span>
      <span id="aiServeDot" style="width:8px;height:8px;border-radius:50%;background:#ff3355;display:none;animation:servePulse 1.2s ease-in-out infinite;flex-shrink:0;"></span>
    </div>
    <div id="aiSets" style="display:flex;gap:6px;"></div>
  </div>
`;
uiContainer.appendChild(scoreDiv);

// Point notification (below serve indicator)
const pointNotif = document.createElement('div');
pointNotif.style.cssText = `
  text-align: center; padding: 4px 10px 10px; color: rgba(255,255,255,0.9); font-size: 36px;
  font-family: 'Instrument Serif', serif; font-style: italic; letter-spacing: 1px;
  opacity: 0; min-height: 50px; font-weight: 400;
  transform: translateY(12px) scale(0.5); filter: blur(8px);
  transition: opacity 0.35s cubic-bezier(0.16,1,0.3,1), transform 0.5s cubic-bezier(0.16,1,0.3,1), filter 0.4s ease;
`;

// Serve dot pulse animation
const serveDotStyle = document.createElement('style');
serveDotStyle.textContent = `
  @keyframes servePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
  }
`;
document.head.appendChild(serveDotStyle);

// Info display
const infoDiv = document.createElement('div');
infoDiv.style.cssText = `
  text-align: center; padding: 10px; color: rgba(255,255,255,0.35); font-size: 14px;
  font-family: 'Instrument Serif', serif; font-style: italic; letter-spacing: 0.5px;
`;
infoDiv.textContent = 'Space / Tap to drop puck · Mouse / WASD to move';
uiContainer.appendChild(infoDiv);

// Point notification sits below the serve info
uiContainer.appendChild(pointNotif);

// Stats bar at bottom
const statsBar = document.createElement('div');
statsBar.style.cssText = `
  position: fixed; bottom: 0; left: 0; width: 100%; pointer-events: none;
  font-family: 'Instrument Serif', serif; z-index: 10; box-sizing: border-box;
  display: flex; justify-content: center; align-items: flex-end; gap: 32px; padding: 14px 24px;
  background: linear-gradient(transparent, rgba(0,0,0,0.6));
`;

function createStatItem(label, id, initialValue) {
  const item = document.createElement('div');
  item.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 2px;';
  const valueEl = document.createElement('div');
  valueEl.id = id;
  valueEl.style.cssText = `
    color: rgba(255,255,255,0.85); font-size: 20px; font-style: italic;
    line-height: 1; letter-spacing: 0.5px;
  `;
  valueEl.textContent = initialValue;
  const labelEl = document.createElement('div');
  labelEl.style.cssText = `
    color: rgba(255,255,255,0.25); font-size: 11px; font-style: italic;
    letter-spacing: 1px; font-family: 'Inter', sans-serif;
  `;
  labelEl.textContent = label;
  item.appendChild(valueEl);
  item.appendChild(labelEl);
  statsBar.appendChild(item);
  return valueEl;
}

createStatItem('Touches', 'statRally', '0');
createStatItem('Longest', 'statLongest', '0');
createStatItem('Avg Touch', 'statAvg', '—');
createStatItem('Top Speed', 'statSpeed', '0');
createStatItem('Streak', 'statStreak', '0');
createStatItem('Aces', 'statAces', '0 / 0');



document.body.appendChild(statsBar);

function updateStatsDisplay() {
  document.getElementById('statRally').textContent = stats.currentRallyTouches;
  document.getElementById('statLongest').textContent = stats.longestRally;
  const avg = stats.totalRallies > 0 ? (stats.totalTouches / stats.totalRallies).toFixed(1) : '—';
  document.getElementById('statAvg').textContent = avg;
  document.getElementById('statSpeed').textContent = stats.topBallSpeed.toFixed(1);
  const currentStreak = Math.max(stats.playerStreak, stats.aiStreak);
  document.getElementById('statStreak').textContent = currentStreak;
  document.getElementById('statStreak').style.color = 'rgba(255,255,255,0.85)';
  document.getElementById('statAces').textContent = `${stats.playerAces} / ${stats.aiAces}`;
}

// Screen flash overlay for point scored
const screenFlash = document.createElement('div');
screenFlash.style.cssText = `
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none; z-index: 5; opacity: 0;
  transition: opacity 0.08s ease-in;
  background: radial-gradient(ellipse at center, transparent 30%, var(--flash-color, rgba(34,153,255,0.3)) 100%);
`;
document.body.appendChild(screenFlash);

let screenFlashTimeout = null;
function triggerScreenFlash(isPlayer) {
  _tmpColor.set(isPlayer ? activePlayerColor : activeAiColor);
  const r = Math.round(_tmpColor.r*255), g = Math.round(_tmpColor.g*255), b = Math.round(_tmpColor.b*255);
  const color = `rgba(${r},${g},${b},0.35)`;
  screenFlash.style.setProperty('--flash-color', color);
  screenFlash.style.background = `radial-gradient(ellipse at center, rgba(${r},${g},${b},0.08) 0%, ${color} 70%, rgba(${r},${g},${b},0.5) 100%)`;

  // Flash in
  screenFlash.style.transition = 'opacity 0.06s ease-in';
  screenFlash.style.opacity = '1';

  if (screenFlashTimeout) clearTimeout(screenFlashTimeout);
  // Hold briefly then fade out
  screenFlashTimeout = setTimeout(() => {
    screenFlash.style.transition = 'opacity 0.6s ease-out';
    screenFlash.style.opacity = '0';
  }, 100);
}

// Message display
const msgDiv = document.createElement('div');
msgDiv.style.cssText = `
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  color: #fff; font-family: 'Instrument Serif', serif; font-size: 32px; font-weight: 400;
  font-style: italic; pointer-events: none; text-align: center; opacity: 0;
  transition: opacity 0.4s ease; letter-spacing: 0.5px;
`;
document.body.appendChild(msgDiv);

function showMessage(text, duration = 1500) {
  msgDiv.textContent = text;
  msgDiv.style.opacity = '1';
  setTimeout(() => { msgDiv.style.opacity = '0'; }, duration);
}

let pointNotifTimeout = null;
function showPointNotif(text, duration = 2200) {
  // Determine color based on scorer — use active theme colors
  const isPlayer = text.includes('You');
  const accentColor = isPlayer ? activePlayerColor : activeAiColor;
  _tmpColor.set(accentColor);
  const glowColor = `rgba(${Math.round(_tmpColor.r*255)},${Math.round(_tmpColor.g*255)},${Math.round(_tmpColor.b*255)},0.4)`;

  // Reset for re-trigger
  pointNotif.style.transition = 'none';
  pointNotif.style.opacity = '0';
  pointNotif.style.transform = 'translateY(12px) scale(0.5)';
  pointNotif.style.filter = 'blur(8px)';
  pointNotif.style.color = accentColor;
  pointNotif.style.textShadow = `0 0 30px ${glowColor}, 0 0 60px ${glowColor}`;

  // Build inner HTML — single line with dash separator
  const scorerName = isPlayer ? 'You' : 'CPU';
  pointNotif.innerHTML = `<span style="font-size:38px;letter-spacing:1px;">Goal — ${scorerName}</span>`;

  // Trigger entrance
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      pointNotif.style.transition = 'opacity 0.35s cubic-bezier(0.16,1,0.3,1), transform 0.5s cubic-bezier(0.16,1,0.3,1), filter 0.4s ease';
      pointNotif.style.opacity = '1';
      pointNotif.style.transform = 'translateY(0) scale(1)';
      pointNotif.style.filter = 'blur(0px)';
    });
  });

  // Exit animation
  if (pointNotifTimeout) clearTimeout(pointNotifTimeout);
  pointNotifTimeout = setTimeout(() => {
    pointNotif.style.transition = 'opacity 0.5s ease, transform 0.5s ease, filter 0.4s ease';
    pointNotif.style.opacity = '0';
    pointNotif.style.transform = 'translateY(-8px) scale(0.95)';
    pointNotif.style.filter = 'blur(4px)';
  }, duration);
}

function updateServeDots() {
  const playerDot = document.getElementById('playerServeDot');
  const aiDot = document.getElementById('aiServeDot');
  if (playerDot && aiDot) {
    playerDot.style.display = gameState.serverIsPlayer ? 'inline-block' : 'none';
    aiDot.style.display = gameState.serverIsPlayer ? 'none' : 'inline-block';
  }
}

function updateScoreDisplay() {
  document.getElementById('playerScore').textContent = gameState.playerScore;
  document.getElementById('aiScore').textContent = gameState.aiScore;
}

function updateSetDisplay() {
  const setLabel = document.getElementById('setLabel');
  if (gameState.matchOver) {
    setLabel.textContent = 'Series Over';
  } else {
    setLabel.textContent = `Game ${gameState.currentSet}`;
  }

  // Render set pips for each side
  ['player', 'ai'].forEach(side => {
    const container = document.getElementById(side + 'Sets');
    container.innerHTML = '';
    const totalSets = SETS_TO_WIN * 2 - 1; // max possible sets
    const won = side === 'player' ? gameState.playerSets : gameState.aiSets;
    const color = side === 'player' ? activePlayerColor : activeAiColor;
    for (let i = 0; i < SETS_TO_WIN; i++) {
      const pip = document.createElement('div');
      pip.style.cssText = `
        width: 8px; height: 8px; border-radius: 50%;
        background: ${i < won ? color : 'rgba(255,255,255,0.1)'};
        transition: background 0.3s ease;
      `;
      container.appendChild(pip);
    }
  });
}

function startNextSet() {
  gameState.playerScore = 0;
  gameState.aiScore = 0;
  gameState.gameOver = false;
  // Alternate first server each set
  gameState.serverIsPlayer = gameState.currentSet % 2 === 1;
  updateScoreDisplay();
  updateSetDisplay();
  updateServeIndicator();
  if (gameState.serverIsPlayer) {
    gameState.paused = true;
    gameState.waitingForPlayerServe = true;
    positionBallOnPaddle();
    showMessage(`Game ${gameState.currentSet} — Your drop`, 2000);
  } else {
    gameState.paused = true;
    showMessage(`Game ${gameState.currentSet} — CPU drops`, 2000);
    setTimeout(() => serve(), 1500);
  }
}

// Point scoring
function scorePoint(scorer) {
  // Finalize rally stats before scoring
  if (stats.currentRallyTouches > 0) {
    stats.totalRallies++;
    stats.totalTouches += stats.currentRallyTouches;
    if (stats.currentRallyTouches > stats.longestRally) {
      stats.longestRally = stats.currentRallyTouches;
    }
  }

  // Ace detection: point scored with only 1 touch (the serve)
  if (stats.currentRallyTouches <= 1) {
    if (scorer === 'player') stats.playerAces++;
    else stats.aiAces++;
  }

  stats.currentRallyTouches = 0;

  if (scorer === 'player') {
    gameState.playerScore++;
    stats.playerWins++;
    stats.playerStreak++;
    stats.aiStreak = 0;
    if (stats.playerStreak > stats.bestStreak) stats.bestStreak = stats.playerStreak;
    showPointNotif('Goal — You');
    triggerScreenFlash(true);
  } else {
    gameState.aiScore++;
    stats.aiWins++;
    stats.aiStreak++;
    stats.playerStreak = 0;
    if (stats.aiStreak > stats.bestStreak) stats.bestStreak = stats.aiStreak;
    showPointNotif('Goal — CPU');
    triggerScreenFlash(false);
  }
  updateScoreDisplay();
  updateStatsDisplay();

  // Check set win (first to 11, lead by 2)
  const ps = gameState.playerScore;
  const as = gameState.aiScore;
  if ((ps >= POINTS_TO_WIN_SET || as >= POINTS_TO_WIN_SET) && Math.abs(ps - as) >= 2) {
    gameState.gameOver = true;

    // Record set result
    gameState.setHistory.push({ player: ps, ai: as });
    if (ps > as) {
      gameState.playerSets++;
    } else {
      gameState.aiSets++;
    }
    updateSetDisplay();

    // Check match win
    if (gameState.playerSets >= SETS_TO_WIN) {
      gameState.matchOver = true;
      showMessage('You Win the Series!', 4000);
      updateSetDisplay();
      setTimeout(() => {
        infoDiv.textContent = 'Press space or tap for a new series';
      }, 500);
    } else if (gameState.aiSets >= SETS_TO_WIN) {
      gameState.matchOver = true;
      showMessage('CPU Wins the Series!', 4000);
      updateSetDisplay();
      setTimeout(() => {
        infoDiv.textContent = 'Press space or tap for a new series';
      }, 500);
    } else {
      // Set won but match continues
      const setWinner = ps > as ? 'You' : 'CPU';
      gameState.currentSet++;
      showMessage(`${setWinner} win${ps > as ? '' : 's'} Game ${gameState.currentSet - 1}!`, 2500);
      setTimeout(() => {
        infoDiv.textContent = 'Press space or tap for next game';
      }, 500);
    }
  }

  if (!gameState.gameOver) {
    // Alternate server every 2 points
    const totalPoints = ps + as;
    gameState.serverIsPlayer = Math.floor(totalPoints / 2) % 2 === 0;

    gameState.paused = true;
    if (gameState.serverIsPlayer) {
      gameState.waitingForPlayerServe = true;
      updateServeIndicator();
      updateServeDots();
    } else {
      updateServeIndicator();
      updateServeDots();
      setTimeout(() => serve(), 1200);
    }
  } else {
    gameState.paused = true;
  }
}

// Ball physics
const tableTop = TABLE_Y + TABLE_HEIGHT / 2;
const tableMinX = -TABLE_WIDTH / 2;
const tableMaxX = TABLE_WIDTH / 2;
const tableMinZ = -TABLE_LENGTH / 2;
const tableMaxZ = TABLE_LENGTH / 2;

let prevBallPos = new THREE.Vector3();
let trailTimer = 0;
let aiTargetX = 0;
let aiReactionDelay = 0;

function updatePhysics(dt) {
  if (gameState.paused) return;

  const bp = gameState.ballPos;
  const bv = gameState.ballVel;

  prevBallPos.copy(bp);

  // Apply gravity
  bv.y += (window._gravity ?? GRAVITY) * dt;

  // Move ball
  bp.x += bv.x * dt;
  bp.y += bv.y * dt;
  bp.z += bv.z * dt;

  // Puck stays flat on the table surface — snap Y position
  bp.y = tableTop + 0.04; // puck height above table
  bv.y = 0; // no vertical movement

  // Side rail bounces — reflect X velocity
  if (bp.x <= tableMinX + BALL_RADIUS) {
    bp.x = tableMinX + BALL_RADIUS;
    bv.x = Math.abs(bv.x) * BOUNCE_DAMPING;
  } else if (bp.x >= tableMaxX - BALL_RADIUS) {
    bp.x = tableMaxX - BALL_RADIUS;
    bv.x = -Math.abs(bv.x) * BOUNCE_DAMPING;
  }

  // Track "bounced on opponent side" whenever puck crosses the centerline in the right direction
  if (bp.z > 0 && gameState.lastHit === 'ai') {
    gameState.bouncedOnOpponentSide = true;
  }
  if (bp.z < 0 && gameState.lastHit === 'player') {
    gameState.bouncedOnOpponentSide = true;
  }

  // Minimum speed — puck never stops
  const hSpeed2 = Math.sqrt(bv.x * bv.x + bv.z * bv.z);
  if (hSpeed2 < 2.0 && hSpeed2 > 0.01) {
    const scale = 2.0 / hSpeed2;
    bv.x *= scale;
    bv.z *= scale;
  }

  // Player paddle collision - use racket head radius
  const pp = playerPaddle.position;
  const racketRadius = PADDLE_WIDTH / 2;
  const hitZoneDepth = 0.8;
  const distXZ_x = bp.x - pp.x;
  const distXZ_y = bp.y - pp.y;
  const distFromCenter = Math.sqrt(distXZ_x * distXZ_x + distXZ_y * distXZ_y);

  if (bp.z > pp.z - hitZoneDepth / 2 - BALL_RADIUS && bp.z < pp.z + hitZoneDepth / 2 + BALL_RADIUS &&
      distFromCenter < racketRadius + BALL_RADIUS + 0.2) {

    const hitX = THREE.MathUtils.clamp(distXZ_x / racketRadius, -1, 1);

    // Air hockey: flat horizontal hit — no Y component
    const MIN_RETURN_SPEED = 9;
    const incomingSpeed = Math.sqrt(bv.x * bv.x + bv.z * bv.z);
    const returnSpeed = Math.max(incomingSpeed * 1.05, MIN_RETURN_SPEED);

    bv.z = -returnSpeed * 0.95;
    bv.x = hitX * returnSpeed * 0.5;
    bv.y = 0;

    bp.z = pp.z - hitZoneDepth / 2 - BALL_RADIUS - 0.12;

    gameState.lastHit = 'player';
    gameState.bouncedOnOpponentSide = false;
    gameState.playerSideBounces = 0;
    gameState.aiSideBounces = 0;
    stats.currentRallyTouches++;
    if (gameState.serving) {
      gameState.serving = false;
      gameState.rallying = true;
    }

    spawnParticles(bp, _tmpColor.set(activePlayerColor).getHex());
    if (partyModeActive) {
      for (let pb = 0; pb < 3; pb++) {
        const pColor = partyColors[Math.floor(Math.random() * partyColors.length)];
        spawnParticles(bp, pColor.getHex());
      }
      playPartyHitSound();
    } else if (retroModeActive) playRetroHitSound(); else if (zenModeActive) playZenHitSound(); else if (infernoModeActive) playInfernoHitSound(); else playPaddleHitSound();
  }

  // AI paddle collision - use racket head radius
  const ap = aiPaddle.position;
  const aiRacketRadius = PADDLE_WIDTH / 2;
  const aiHitZoneDepth = 0.8;
  const aiDistX = bp.x - ap.x;
  const aiDistY = bp.y - ap.y;
  const aiDistFromCenter = Math.sqrt(aiDistX * aiDistX + aiDistY * aiDistY);

  if (bp.z < ap.z + aiHitZoneDepth / 2 + BALL_RADIUS && bp.z > ap.z - aiHitZoneDepth / 2 - BALL_RADIUS &&
      aiDistFromCenter < aiRacketRadius + BALL_RADIUS + 0.2) {

    const hitX = THREE.MathUtils.clamp(aiDistX / aiRacketRadius, -1, 1);

    // Air hockey: flat horizontal hit from AI mallet
    const MIN_RETURN_SPEED = 9;
    const incomingSpeed = Math.sqrt(bv.x * bv.x + bv.z * bv.z);
    const returnSpeed = Math.max(incomingSpeed * 1.05, MIN_RETURN_SPEED);

    bv.z = returnSpeed * 0.95;
    bv.x = hitX * returnSpeed * 0.5;
    bv.y = 0;

    bp.z = ap.z + aiHitZoneDepth / 2 + BALL_RADIUS + 0.12;

    gameState.lastHit = 'ai';
    gameState.bouncedOnOpponentSide = false;
    gameState.playerSideBounces = 0;
    gameState.aiSideBounces = 0;
    stats.currentRallyTouches++;
    if (gameState.serving) {
      gameState.serving = false;
      gameState.rallying = true;
    }

    spawnParticles(bp, _tmpColor.set(activeAiColor).getHex());
    if (partyModeActive) playPartyHitSound(); else if (retroModeActive) playRetroHitSound(); else if (zenModeActive) playZenHitSound(); else if (infernoModeActive) playInfernoHitSound(); else playPaddleHitSound();
  }

  // Out of bounds - scoring
  // Rule: after a paddle hit, the ball MUST bounce on the opponent's side of the table.
  // If it goes out (past the end, off the side, or falls to the floor) without having
  // bounced on the opponent's side, the HITTER loses the point (shot went out).
  // If it DID bounce on the opponent's side and then goes out, the RECEIVER loses
  // the point (they failed to return it).
  // Air hockey goals — puck must pass through goal slot (center zone only)
  const goalHalfWidth = 0.7; // half of 1.4 goal width
  if (bp.z > TABLE_LENGTH / 2 - 0.1) {
    // Puck reached player's goal end
    if (Math.abs(bp.x) < goalHalfWidth) {
      // Puck in goal slot — AI scores
      scorePoint('ai');
    } else {
      // Puck hit the end rail outside goal — bounce back
      bv.z = -Math.abs(bv.z) * BOUNCE_DAMPING;
      bp.z = TABLE_LENGTH / 2 - 0.15;
    }
  } else if (bp.z < -(TABLE_LENGTH / 2 - 0.1)) {
    // Puck reached AI's goal end
    if (Math.abs(bp.x) < goalHalfWidth) {
      // Puck in goal slot — player scores
      scorePoint('player');
    } else {
      // Bounce off end rail
      bv.z = Math.abs(bv.z) * BOUNCE_DAMPING;
      bp.z = -(TABLE_LENGTH / 2 - 0.15);
    }
  }
}

// AI logic
let aiSmoothedX = 0;
let aiSmoothedZ = -(TABLE_LENGTH / 2 - 0.6);
const AI_REST_X = 0;
const AI_REST_Z = -(TABLE_LENGTH / 2 - 0.6);
const MALLET_Y = TABLE_Y + TABLE_HEIGHT / 2 + PADDLE_DEPTH * 0.75;

function updateAI(dt) {
  if (gameState.paused) return;

  const bp = gameState.ballPos;
  const bv = gameState.ballVel;
  const ap = aiPaddle.position;

  let targetX, targetZ;
  let lerpSpeed;

  // Flat puck — AI tracks in X and Z only
  if (bv.z < 0) {
    // Puck coming toward AI — intercept
    const timeToReach = Math.abs((bp.z - AI_REST_Z) / (bv.z + 0.001));
    targetX = bp.x + bv.x * timeToReach;
    targetX = THREE.MathUtils.clamp(targetX, tableMinX + PADDLE_WIDTH / 2, tableMaxX - PADDLE_WIDTH / 2);
    // Move forward to meet puck, but stay in own half
    targetZ = THREE.MathUtils.clamp(bp.z - 0.5, -TABLE_LENGTH / 2 + 0.4, -0.4);
    lerpSpeed = 0.14;
  } else {
    // Puck going away — return to goal-line position
    targetX = AI_REST_X;
    targetZ = AI_REST_Z;
    lerpSpeed = 0.04;
  }

  aiSmoothedX += (targetX - aiSmoothedX) * lerpSpeed;
  aiSmoothedZ += (targetZ - aiSmoothedZ) * lerpSpeed;

  const aiSpeed = window._aiSpeed ?? 5.5;
  const diffX = aiSmoothedX - ap.x;
  const moveX = Math.sign(diffX) * Math.min(Math.abs(diffX), aiSpeed * dt);
  ap.x += moveX;
  ap.x = THREE.MathUtils.clamp(ap.x, tableMinX + PADDLE_WIDTH / 2, tableMaxX - PADDLE_WIDTH / 2);

  const diffZ = aiSmoothedZ - ap.z;
  const moveZ = Math.sign(diffZ) * Math.min(Math.abs(diffZ), aiSpeed * dt);
  ap.z += moveZ;
  ap.z = THREE.MathUtils.clamp(ap.z, -TABLE_LENGTH / 2 + 0.3, -0.3);

  // Keep AI mallet at table surface level
  ap.y = MALLET_Y;
}

// Player mallet control — flat on table surface
function updatePlayer(dt) {
  const pp = playerPaddle.position;
  const moveSpeed = 9;

  if (useMouseControl) {
    // X: table width
    const targetX = mouseX * (TABLE_WIDTH / 2);
    pp.x += (targetX - pp.x) * 0.45;
    // Z: map mouse Y to player's half (z = 0.3 near center, z = TABLE_LENGTH/2-0.3 at back)
    const targetZ = THREE.MathUtils.lerp(0.35, TABLE_LENGTH / 2 - 0.3, (mouseY + 1) / 2);
    pp.z += (targetZ - pp.z) * 0.45;
  } else {
    if (keys.a) pp.x -= moveSpeed * dt;
    if (keys.d) pp.x += moveSpeed * dt;
    if (keys.w) pp.z -= moveSpeed * dt;
    if (keys.s) pp.z += moveSpeed * dt;
  }

  pp.x = THREE.MathUtils.clamp(pp.x, tableMinX + PADDLE_WIDTH / 2, tableMaxX - PADDLE_WIDTH / 2);
  pp.z = THREE.MathUtils.clamp(pp.z, 0.3, TABLE_LENGTH / 2 - 0.25);

  // Scroll wheel — minor Z nudge
  pp.z += scrollDelta * 0.2;
  scrollDelta *= 0.8;
  if (Math.abs(scrollDelta) < 0.001) scrollDelta = 0;

  // Keep mallet flat on table
  pp.y = MALLET_Y;
}

// Trail update
function updateTrail() {
  trailTimer++;
  if (trailTimer % 2 === 0) {
    for (let i = trailCount - 1; i > 0; i--) {
      trailPositions[i].copy(trailPositions[i - 1]);
    }
    trailPositions[0].copy(gameState.ballPos);
  }

  for (let i = 0; i < trailCount; i++) {
    trailMeshes[i].position.copy(trailPositions[i]);
    trailMeshes[i].visible = !gameState.paused && trailPositions[i].y > -5;
    const s = 1 - i / trailCount;
    if (infernoModeActive) {
      // Bigger, brighter fireball trail
      trailMeshes[i].scale.setScalar(s * 2.0);
      const t = i / trailCount;
      const hue = THREE.MathUtils.lerp(0.1, 0.0, t); // yellow -> red
      const lightness = THREE.MathUtils.lerp(0.6, 0.3, t);
      trailMeshes[i].material.color.setHSL(hue, 1, lightness);
      trailMeshes[i].material.opacity = (1 - t) * 0.7;
    } else if (zenModeActive) {
      // Soft white-to-blue ethereal trail
      const t = i / trailCount;
      trailMeshes[i].scale.setScalar(s * 0.8);
      const hue = THREE.MathUtils.lerp(0.0, 0.58, t); // white -> baby blue
      const sat = THREE.MathUtils.lerp(0.0, 0.5, t);
      const light = THREE.MathUtils.lerp(1.0, 0.7, t);
      trailMeshes[i].material.color.setHSL(hue, sat, light);
      trailMeshes[i].material.opacity = (1 - t) * 0.4;
    } else {
      trailMeshes[i].scale.setScalar(s);
    }
  }
}

// Audio context for edge/corner sound cues
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

function playBounceSound() {
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(800, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.04);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.05);

    // Add a noise burst for the "tick" texture
    const bufferSize = Math.floor(ctx.sampleRate * 0.03);
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.15));
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.08, ctx.currentTime);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.03);
    const filter = ctx.createBiquadFilter();
    filter.type = 'highpass';
    filter.frequency.value = 2000;
    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(ctx.destination);
    noise.start();
    noise.stop(ctx.currentTime + 0.03);
  } catch (e) { /* audio not available */ }
}

function playPaddleHitSound() {
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;

    // Low woody thump — sine body
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(900, t);
    osc1.frequency.exponentialRampToValueAtTime(500, t + 0.04);
    gain1.gain.setValueAtTime(0.2, t);
    gain1.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(t);
    osc1.stop(t + 0.05);

    // High harmonic for crisp click
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(1400, t);
    osc2.frequency.exponentialRampToValueAtTime(700, t + 0.03);
    gain2.gain.setValueAtTime(0.1, t);
    gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(t);
    osc2.stop(t + 0.04);

    // Highpass-filtered noise for thin click texture
    const bufferSize = Math.floor(ctx.sampleRate * 0.03);
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.1));
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.1, t);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 0.035);
    const bp = ctx.createBiquadFilter();
    bp.type = 'highpass';
    bp.frequency.value = 2500;
    bp.Q.value = 1.0;
    noise.connect(bp);
    bp.connect(noiseGain);
    noiseGain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.035);
  } catch (e) { /* audio not available */ }
}

// Particles on bounce — POOLED for performance (no alloc per spawn)
const PARTICLE_POOL_SIZE = 80;
const particlePool = [];
const particlePoolGeo = new THREE.SphereGeometry(0.03, 4, 4);
const _pVelDelta = new THREE.Vector3(); // reusable scratch vector
for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
  const mat = new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0 });
  const m = new THREE.Mesh(particlePoolGeo, mat);
  m.name = `particle${i}`;
  m.visible = false;
  m.frustumCulled = false;
  scene.add(m);
  particlePool.push({ mesh: m, vel: new THREE.Vector3(), life: 0, active: false });
}
let particlePoolIdx = 0;

const particles = []; // active references
const particleActiveSet = new Set(); // O(1) lookup instead of includes()
function spawnParticles(pos, color = 0xffaa00) {
  for (let i = 0; i < 8; i++) {
    const p = particlePool[particlePoolIdx];
    particlePoolIdx = (particlePoolIdx + 1) % PARTICLE_POOL_SIZE;
    p.mesh.material.color.set(color);
    p.mesh.material.opacity = 1;
    p.mesh.position.copy(pos);
    p.mesh.visible = true;
    p.mesh.scale.setScalar(1);
    p.vel.set(
      (Math.random() - 0.5) * 3,
      Math.random() * 3,
      (Math.random() - 0.5) * 3
    );
    p.life = 1;
    p.active = true;
    if (!particleActiveSet.has(p)) { particleActiveSet.add(p); particles.push(p); }
  }
}

function updateParticles(dt) {
  let writeIdx = 0;
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    if (!p.active) { particleActiveSet.delete(p); continue; }
    p.vel.y -= 8 * dt;
    _pVelDelta.copy(p.vel).multiplyScalar(dt);
    p.mesh.position.add(_pVelDelta);
    p.life -= dt * 2;
    if (p.life <= 0) {
      p.mesh.visible = false;
      p.mesh.material.opacity = 0;
      p.active = false;
      particleActiveSet.delete(p);
    } else {
      p.mesh.material.opacity = p.life;
      particles[writeIdx++] = p;
    }
  }
  particles.length = writeIdx;
}

// Watch for bounces to trigger particles
let lastBallVelY = 0;

// Main loop
const clock = new THREE.Clock();

function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);

  updatePlayer(dt);
  updateAI(dt);

  const prevVelY = gameState.ballVel.y;
  updatePhysics(dt);

  // Detect puck bouncing off side rails (X velocity sign flip)
  const prevBvX = gameState.ballVel.x;
  const bpNow = gameState.ballPos;
  const didRailBounce = (prevBallPos.x - bpNow.x) * prevBvX < -0.01 && !gameState.paused &&
    (bpNow.x <= tableMinX + BALL_RADIUS + 0.05 || bpNow.x >= tableMaxX - BALL_RADIUS - 0.05);

  if (didRailBounce) {
    if (partyModeActive) {
      for (let pb = 0; pb < 3; pb++) {
        const pColor = partyColors[Math.floor(Math.random() * partyColors.length)];
        spawnParticles(bpNow, pColor.getHex());
      }
      playPartyBounceSound();
    } else if (retroModeActive) {
      spawnParticles(bpNow, 0x33ff66);
      playRetroBounceSound();
    } else if (zenModeActive) {
      const zbc = [0xaaddff, 0x88ccee, 0xccddff][Math.floor(Math.random() * 3)];
      spawnParticles(bpNow, zbc);
      playZenBounceSound();
      spawnZenRipple(bpNow);
    } else if (infernoModeActive) {
      const fColor = [0xff4400, 0xff6600, 0xff8800][Math.floor(Math.random() * 3)];
      spawnParticles(bpNow, fColor);
      playInfernoBounceSound();
      for (let fi = 0; fi < 3; fi++) spawnInfernoFireTrail(bpNow);
    } else {
      spawnParticles(bpNow, 0x00ccff);
      playBounceSound();
    }
  }

  updateParticles(dt);

  // While waiting for serve, keep ball on the server's paddle
  if (gameState.waitingForPlayerServe || (gameState.paused && !gameState.gameOver)) {
    positionBallOnPaddle();
  }

  // Update puck mesh — flat, spins on Y axis only
  ball.position.copy(gameState.ballPos);
  ball.rotation.y += gameState.ballVel.length() * dt * 1.5;

  // Theme color lerp animation
  if (themeLerpActive) {
    themeLerpProgress += dt / THEME_LERP_DURATION;
    if (themeLerpProgress >= 1) {
      themeLerpProgress = 1;
      themeLerpActive = false;
    }
    applyLerpedTheme(themeLerpProgress);
  }

  // Party mode update
  updatePartyMode(dt);

  // Retro mode update
  updateRetroMode(dt);

  // Zen mode update
  updateZenMode(dt);

  // Inferno mode update
  updateInfernoMode(dt);

  if (window._trailEnabled !== false) updateTrail();
  else trailMeshes.forEach(m => m.visible = false);

  // Mallet rotation — subtle Y-axis wobble based on movement speed
  const rawMouseVelX = mouseX - prevMouseX;
  const rawMouseVelY = mouseY - prevMouseY;
  prevMouseX = mouseX;
  prevMouseY = mouseY;

  const smoothFactor = 0.15;
  smoothMouseVelX += (rawMouseVelX - smoothMouseVelX) * smoothFactor;
  smoothMouseVelY += (rawMouseVelY - smoothMouseVelY) * smoothFactor;

  // Mallets stay flat — only subtle Y spin on fast lateral movement
  playerPaddle.rotation.y += (smoothMouseVelX * 2.0 - playerPaddle.rotation.y) * 0.12;
  playerPaddle.rotation.x = 0;
  playerPaddle.rotation.z = 0;

  aiPaddle.rotation.y += ((aiPaddle.position.x - prevAiX) * 1.5 - aiPaddle.rotation.y) * 0.12;
  aiPaddle.rotation.x = 0;
  aiPaddle.rotation.z = 0;

  prevPlayerX = playerPaddle.position.x;
  prevAiX = aiPaddle.position.x;
  prevPlayerZ = aiPaddle.position.z;

  // Ball glow effect + track top speed
  const speed = gameState.ballVel.length();
  ballMat.emissiveIntensity = 0.2 + speed * 0.05;
  if (speed > stats.topBallSpeed && !gameState.paused) {
    stats.topBallSpeed = speed;
  }

  // Live rally counter update
  if (!gameState.paused) {
    const rallyEl = document.getElementById('statRally');
    const speedEl = document.getElementById('statSpeed');
    if (rallyEl) rallyEl.textContent = stats.currentRallyTouches;
    if (speedEl) speedEl.textContent = stats.topBallSpeed.toFixed(1);
  }

  // Camera subtle movement (skip when in free orbit mode)
  if (!freeOrbitMode) {
    camera.position.x += (mouseX * 1.5 - camera.position.x) * 0.02;
    camera.lookAt(0, 0.5, 0);
  } else {
    orbitControls.update();
  }

  // IMPORTANT: Update camera matrices every frame for correct SSR
  projMatU.value.copy(camera.projectionMatrix);
  projInvMatU.value.copy(camera.projectionMatrixInverse);
  viewMatInvU.value.copy(camera.matrixWorld);



  postProcessing.render();
}

let prevPlayerX = 0;
let prevAiX = 0;
let prevPlayerZ = 0;
let prevMouseX = 0;
let prevMouseY = 0;
let smoothMouseVelX = 0;
let smoothMouseVelY = 0;

// FPS counter
const fpsEl = document.getElementById('fps-counter');
const settingsFpsEl = document.getElementById('settings-fps');
let fpsFrames = 0, fpsLastTime = performance.now();
function updateFPS() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsLastTime >= 500) {
    const fps = Math.round(fpsFrames / ((now - fpsLastTime) / 1000));
    fpsEl.textContent = `FPS: ${fps}`;
    if (settingsFpsEl) {
      settingsFpsEl.textContent = fps;
      settingsFpsEl.style.color = fps >= 55 ? '#0f0' : fps >= 30 ? '#ff0' : '#f33';
    }
    fpsFrames = 0;
    fpsLastTime = now;
  }
}

const _origAnimate = animate;
function animateWithFPS(time) {
  _origAnimate(time);
  updateFPS();
}
renderer.setAnimationLoop(animateWithFPS);

// Handle resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Update serve indicator text
function updateServeIndicator() {
  if (gameState.gameOver) return;
  if (gameState.serverIsPlayer) {
    infoDiv.textContent = 'Your drop — click or press space';
  } else {
    infoDiv.textContent = 'CPU dropping puck…';
  }
  updateServeDots();
}

// Start — show ball resting on player paddle, waiting for input
gameState.paused = true;
gameState.waitingForPlayerServe = true;
gameState.serverIsPlayer = true;

function positionBallOnPaddle() {
  const side = gameState.serverIsPlayer ? 1 : -1;
  const puckY = TABLE_Y + TABLE_HEIGHT / 2 + 0.04;
  gameState.ballPos.set(0, puckY, side * (TABLE_LENGTH / 2 - 1.8));
  gameState.ballVel.set(0, 0, 0);
  ball.position.copy(gameState.ballPos);
}

positionBallOnPaddle();
updateSetDisplay();
showMessage('Click or tap to drop the puck', 3000);
updateServeIndicator();
updateServeDots();