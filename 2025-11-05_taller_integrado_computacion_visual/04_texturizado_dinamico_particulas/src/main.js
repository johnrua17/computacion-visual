import { ParticleSystem } from './particles/particleSystem.js';

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.getElementById('container').appendChild(renderer.domElement);

// Dynamic material shader
const vertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    uniform float uTime;
    
    float noise(vec3 p) {
        return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
    }
    
    void main() {
        vUv = uv;
        vPosition = position;
        vNormal = normal;
        
        vec3 pos = position;
        float n = noise(position * 2.0 + uTime * 0.5);
        pos += normal * n * 0.1;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
`;

const fragmentShader = `
    uniform float uTime;
    uniform float uNoiseSpeed;
    uniform float uEmissiveIntensity;
    uniform vec3 uColorA;
    uniform vec3 uColorB;
    
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    
    float smoothNoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        
        float a = noise(i);
        float b = noise(i + vec2(1.0, 0.0));
        float c = noise(i + vec2(0.0, 1.0));
        float d = noise(i + vec2(1.0, 1.0));
        
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    
    void main() {
        vec2 uv = vUv;
        uv += vec2(sin(uTime * 0.5), cos(uTime * 0.3)) * 0.1;
        
        float n1 = smoothNoise(uv * 5.0 + uTime * uNoiseSpeed);
        float n2 = smoothNoise(uv * 10.0 - uTime * uNoiseSpeed * 0.5);
        float n3 = smoothNoise(uv * 20.0 + uTime * uNoiseSpeed * 0.25);
        
        float noiseValue = (n1 + n2 * 0.5 + n3 * 0.25) / 1.75;
        
        vec3 color = mix(uColorA, uColorB, noiseValue);
        
        float emissive = sin(vPosition.y * 3.0 + uTime * 2.0) * 0.5 + 0.5;
        color += emissive * uEmissiveIntensity * 0.5;
        
        float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
        color += fresnel * uColorA * 0.3;
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

const dynamicMaterial = new THREE.ShaderMaterial({
    uniforms: {
        uTime: { value: 0 },
        uNoiseSpeed: { value: 1.0 },
        uEmissiveIntensity: { value: 1.0 },
        uColorA: { value: new THREE.Color(0x00ffff) },
        uColorB: { value: new THREE.Color(0xff00ff) }
    },
    vertexShader: vertexShader,
    fragmentShader: fragmentShader,
    side: THREE.DoubleSide
});

// Main mesh
const geometry = new THREE.IcosahedronGeometry(2, 4);
const mesh = new THREE.Mesh(geometry, dynamicMaterial);
scene.add(mesh);

// Particle system
const particleSystem = new ParticleSystem(scene, 1000);

// Controls
let isPlaying = true;
let time = 0;

document.getElementById('emissive-slider').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    dynamicMaterial.uniforms.uEmissiveIntensity.value = value;
    document.getElementById('emissive-value').textContent = value.toFixed(2);
});

document.getElementById('noise-slider').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    dynamicMaterial.uniforms.uNoiseSpeed.value = value;
    document.getElementById('noise-value').textContent = value.toFixed(2);
});

document.getElementById('play-pause').addEventListener('click', () => {
    isPlaying = !isPlaying;
    document.getElementById('play-pause').textContent = isPlaying ? 'Pause' : 'Play';
});

document.getElementById('reset').addEventListener('click', () => {
    dynamicMaterial.uniforms.uEmissiveIntensity.value = 1.0;
    dynamicMaterial.uniforms.uNoiseSpeed.value = 1.0;
    document.getElementById('emissive-slider').value = 1.0;
    document.getElementById('noise-slider').value = 1.0;
    document.getElementById('emissive-value').textContent = '1.0';
    document.getElementById('noise-value').textContent = '1.0';
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    if (isPlaying) {
        time += 0.01;
        dynamicMaterial.uniforms.uTime.value = time;
        
        mesh.rotation.x += 0.003;
        mesh.rotation.y += 0.005;
        
        particleSystem.update(time);
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