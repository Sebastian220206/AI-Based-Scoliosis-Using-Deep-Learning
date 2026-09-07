import * as THREE from "https://unpkg.com/three@0.152.2/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.152.2/examples/jsm/controls/OrbitControls.js?module";
import { GLTFLoader } from "https://unpkg.com/three@0.152.2/examples/jsm/loaders/GLTFLoader.js?module";



let scene, camera, renderer, controls;
let uploadedImage = null;
let xraySelected = false;
window.loadDataset = function () {

  fetch("http://127.0.0.1:5000/api/xrays")
    .then(res => res.json())
    .then(data => {
      const list = document.getElementById("xray-list");
      list.innerHTML = "";

      data.forEach(item => {
        const li = document.createElement("li");

        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.name;

        img.onclick = () => selectXray(item.url);

        li.appendChild(img);
        list.appendChild(li);
      });
    })
    .catch(err => console.error(err));
}

let uploadedXray = null;
window.selectXray = function (url) {

  console.log("IMAGE URL FROM BACKEND:", url);

  uploadedImage = new Image();
  uploadedImage.src = url;

  uploadedImage.onload = () => {
    xraySelected = true;
    console.log("IMAGE LOADED SUCCESSFULLY");
  };

  const preview = document.getElementById("xray-preview");
  preview.src = url;
  preview.style.display = "block";

  document.getElementById("xray-status").innerText =
    "✅ X-ray selected from dataset";
}


window.handleXrayUpload = function (event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const preview = document.getElementById("xray-preview");
    preview.src = e.target.result;
    preview.style.display = "block";

    preview.onload = () => {
      xraySelected = true;
      console.log("X-ray uploaded and ready");
    };
  };
  reader.readAsDataURL(file);
}



window.show3D = function () {
  if (!xraySelected) {
    alert("Please upload or select an X-ray first");
    return;
  }

  const status = document.getElementById("ai-status");
  const loader = document.getElementById("loader");

  status.style.display = "block";
  loader.style.display = "block";

  setTimeout(() => {
    status.innerHTML = "🔍 Extracting spinal curvature features...";
  }, 1000);

  setTimeout(() => {
    status.innerHTML = "📊 Generating 3D reconstruction...";
  }, 2000);

  setTimeout(() => {
    status.style.display = "none";
    loader.style.display = "none";

    initScene();
    loadSpineModel();
  }, 3000);
};

function initScene() {
  const container = document.getElementById("three-container");
  container.innerHTML = "";
  container.style.display = "block";

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1220); // Medical deep blue

  camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / 450,
    0.1,
    2000
  );

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, 450);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  // --- OrbitControls ---
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);
  controls.update();

  // --- Lighting (Robust Clinical Setup) ---
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
  keyLight.position.set(5, 10, 7);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x88ccff, 1.0);
  fillLight.position.set(-5, 5, 5);
  scene.add(fillLight);

  animate();
}

function loadSpineModel() {
  const loader = new GLTFLoader();

  loader.load("http://127.0.0.1:5000/models/spine.glb", function (gltf) {
    const model = gltf.scene;

    // Compute bounding box
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    // Move model so center = (0,0,0)
    model.position.sub(center);

    scene.add(model);

    // Set orbit center
    controls.target.set(0, 0, 0);
    controls.update();

    // Position camera based on model size
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(0, maxDim * 0.5, maxDim * 2);

    console.log("✅ Spine GLB loaded and centered at origin");
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}


// function extractCurvatureFromXray() {
//   if (!uploadedImage) return 0.2;

//   const ratio = uploadedImage.width / uploadedImage.height;

//   if (ratio > 1.4) return 0.45; 
//   if (ratio > 1.2) return 0.35; 
//   return 0.15;                 
// }
function openDataset() {
  document.getElementById("dataset-panel").classList.add("open");
  loadDataset();
}
function toggleDataset() {
  const panel = document.getElementById("dataset-panel");
  panel.classList.toggle("open");

  if (panel.classList.contains("open")) {
    loadDataset();
  }
}
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("openDatasetBtn");
  if (btn) {
    btn.addEventListener("click", openDataset);
  }
});
function extractCurvatureFromXray() {

  if (!uploadedImage) return 0.2;


  const ratio = uploadedImage.width / uploadedImage.height;

  if (ratio > 1.4) return 0.45;
  if (ratio > 1.0) return 0.35;
  return 0.25;
}
