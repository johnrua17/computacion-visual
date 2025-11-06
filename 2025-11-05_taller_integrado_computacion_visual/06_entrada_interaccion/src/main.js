import { KeyboardInput } from './input/keyboard.js';
import { MouseInput } from './input/mouse.js';
import { TouchInput } from './input/touch.js';
import { CollisionSystem } from './physics/collisions.js';

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 8;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.getElementById('container').appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0x00ffff, 1, 100);
pointLight.position.set(5, 5, 5);
scene.add(pointLight);

// Main object
const mainGeometry = new THREE.IcosahedronGeometry(1.5, 1);
const mainMaterial = new THREE.MeshStandardMaterial({
    color: 0xff00ff,
    emissive: 0xff00ff,
    emissiveIntensity: 0.2,
    metalness: 0.7,
    roughness: 0.3
});
const mainMesh = new THREE.Mesh(mainGeometry, mainMaterial);
scene.add(mainMesh);

// Collision objects
const collisionObjects = [];
const collisionGeometry = new THREE.SphereGeometry(0.5, 16, 16);

for (let i = 0; i < 8; i++) {
    const material = new THREE.MeshStandardMaterial({
        color: Math.random() * 0xffffff,
        metalness: 0.5,
        roughness: 0.5
    });
    const mesh = new THREE.Mesh(collisionGeometry, material);
    
    const angle = (i / 8) * Math.PI * 2;
    const radius = 4;
    mesh.position.x = Math.cos(angle) * radius;
    mesh.position.y = Math.sin(angle) * radius;
    mesh.position.z = (Math.random() - 0.5) * 2;
    
    mesh.userData = { 
        id: i, 
        originalColor: material.color.clone(),
        velocity: new THREE.Vector3(
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02
        )
    };
    
    scene.add(mesh);
    collisionObjects.push(mesh);
}

// Input systems
const keyboard = new KeyboardInput(mainMesh);
const mouse = new MouseInput(camera, collisionObjects, renderer.domElement);
const touch = new TouchInput(camera, mainMesh, renderer.domElement);
const collisionSystem = new CollisionSystem(mainMesh, collisionObjects);

// UI controls
let collisionCount = 0;

document.getElementById('color-picker').addEventListener('input', (e) => {
    mainMaterial.color.setStyle(e.target.value);
    mainMaterial.emissive.setStyle(e.target.value);
});

document.getElementById('scale-slider').addEventListener('input', (e) => {
    const scale = parseFloat(e.target.value);
    mainMesh.scale.setScalar(scale);
    document.getElementById('scale-value').textContent = scale.toFixed(2);
});

document.getElementById('reset-btn').addEventListener('click', () => {
    mainMesh.position.set(0, 0, 0);
    mainMesh.rotation.set(0, 0, 0);
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Update input status display
    document.getElementById('mouse-pos').textContent = `(${mouse.position.x}, ${mouse.position.y})`;
    document.getElementById('keys-pressed').textContent = keyboard.getActiveKeys() || 'None';
    document.getElementById('touch-status').textContent = touch.isActive ? 'Active' : 'Inactive';
    
    // Rotate main mesh
    mainMesh.rotation.x += 0.01;
    mainMesh.rotation.y += 0.01;
    
    // Move collision objects
    collisionObjects.forEach(obj => {
        obj.position.add(obj.userData.velocity);
        
        ['x', 'y', 'z'].forEach(axis => {
            if (Math.abs(obj.position[axis]) > 6) {
                obj.userData.velocity[axis] *= -1;
            }
        });
        
        obj.rotation.x += 0.01;
        obj.rotation.y += 0.01;
    });
    
    // Check collisions
    const collisions = collisionSystem.check();
    if (collisions.length > 0) {
        collisionCount += collisions.length;
        document.getElementById('collision-count').textContent = collisionCount;
        document.getElementById('last-collision').textContent = `Object ${collisions[0].id}`;
    }
    
    renderer.render(scene, camera);
}

// Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();