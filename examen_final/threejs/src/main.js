import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================
// CONFIGURACIÓN INICIAL
// ============================================

let scene, camera, renderer, controls;
let geometricShapes = [];
let currentCameraPosition = 1;

// Posiciones predefinidas de cámara
const cameraPositions = {
    1: { position: new THREE.Vector3(15, 10, 15), target: new THREE.Vector3(0, 0, 0) },
    2: { position: new THREE.Vector3(-10, 15, 10), target: new THREE.Vector3(0, 0, 0) }
};

// ============================================
// INICIALIZACIÓN
// ============================================

function init() {
    // Crear escena
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    scene.fog = new THREE.Fog(0x1a1a2e, 20, 50);

    // Configurar cámara de perspectiva
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.copy(cameraPositions[1].position);
    camera.lookAt(cameraPositions[1].target);

    // Configurar renderizador
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    const container = document.getElementById('canvas-container');
    container.appendChild(renderer.domElement);

    // Configurar OrbitControls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controls.target.copy(cameraPositions[1].target);

    // Agregar luces
    setupLights();

    // Cargar texturas y crear formas
    loadTexturesAndCreateShapes();

    // Crear el piso
    createFloor();

    // Event listeners
    setupEventListeners();

    // Ocultar pantalla de carga
    document.getElementById('loading').style.display = 'none';

    // Iniciar animación
    animate();
}

// ============================================
// ILUMINACIÓN
// ============================================

function setupLights() {
    // Luz 1: Luz direccional (simula el sol)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.camera.left = -20;
    directionalLight.shadow.camera.right = 20;
    directionalLight.shadow.camera.top = 20;
    directionalLight.shadow.camera.bottom = -20;
    scene.add(directionalLight);

    // Luz 2: Luz puntual (iluminación de acento)
    const pointLight = new THREE.PointLight(0x64ffda, 2, 50);
    pointLight.position.set(-10, 10, -10);
    pointLight.castShadow = true;
    scene.add(pointLight);

    // Agregar esfera visible para la luz puntual
    const sphereGeometry = new THREE.SphereGeometry(0.3, 16, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0x64ffda });
    const lightSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    lightSphere.position.copy(pointLight.position);
    scene.add(lightSphere);

    // Luz ambiental suave
    const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
    scene.add(ambientLight);
}

// ============================================
// TEXTURAS Y FORMAS GEOMÉTRICAS
// ============================================

function loadTexturesAndCreateShapes() {
    const textureLoader = new THREE.TextureLoader();

    // Textura 1: Patrón de tablero de ajedrez (generada proceduralmente)
    const checkerTexture = createCheckerTexture();
    checkerTexture.wrapS = THREE.RepeatWrapping;
    checkerTexture.wrapT = THREE.RepeatWrapping;
    checkerTexture.repeat.set(2, 2);

    // Textura 2: Patrón de ladrillos (generada proceduralmente)
    const brickTexture = createBrickTexture();
    brickTexture.wrapS = THREE.RepeatWrapping;
    brickTexture.wrapT = THREE.RepeatWrapping;
    brickTexture.repeat.set(1, 1);

    // Crear formas geométricas
    createGeometricShapes(checkerTexture, brickTexture);
}

// Función para crear textura de tablero de ajedrez
function createCheckerTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');

    const size = 64;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            context.fillStyle = (i + j) % 2 === 0 ? '#ffffff' : '#333333';
            context.fillRect(i * size, j * size, size, size);
        }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

// Función para crear textura de ladrillos
function createBrickTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');

    // Fondo
    context.fillStyle = '#8B4513';
    context.fillRect(0, 0, 512, 512);

    // Ladrillos
    const brickWidth = 128;
    const brickHeight = 64;
    context.fillStyle = '#A0522D';
    context.strokeStyle = '#654321';
    context.lineWidth = 4;

    for (let y = 0; y < 512; y += brickHeight) {
        const offset = (y / brickHeight) % 2 === 0 ? 0 : brickWidth / 2;
        for (let x = -brickWidth / 2; x < 512; x += brickWidth) {
            context.fillRect(x + offset, y, brickWidth - 4, brickHeight - 4);
            context.strokeRect(x + offset, y, brickWidth - 4, brickHeight - 4);
        }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

function createGeometricShapes(texture1, texture2) {
    // 1. Cubo con textura de tablero
    const cube1 = new THREE.Mesh(
        new THREE.BoxGeometry(2, 2, 2),
        new THREE.MeshStandardMaterial({ map: texture1 })
    );
    cube1.position.set(-5, 1, 0);
    cube1.castShadow = true;
    cube1.receiveShadow = true;
    cube1.userData = { rotationSpeed: { x: 0.01, y: 0.02, z: 0 } };
    scene.add(cube1);
    geometricShapes.push(cube1);

    // 2. Esfera con textura de ladrillos
    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(1.5, 32, 32),
        new THREE.MeshStandardMaterial({ map: texture2 })
    );
    sphere.position.set(0, 1.5, -5);
    sphere.castShadow = true;
    sphere.receiveShadow = true;
    sphere.userData = { rotationSpeed: { x: 0, y: 0.015, z: 0.01 } };
    scene.add(sphere);
    geometricShapes.push(sphere);

    // 3. Cono con material colorido
    const cone = new THREE.Mesh(
        new THREE.ConeGeometry(1, 3, 32),
        new THREE.MeshStandardMaterial({ 
            color: 0xff6b6b,
            roughness: 0.3,
            metalness: 0.5
        })
    );
    cone.position.set(5, 1.5, 0);
    cone.castShadow = true;
    cone.receiveShadow = true;
    cone.userData = { rotationSpeed: { x: 0, y: 0.02, z: 0 } };
    scene.add(cone);
    geometricShapes.push(cone);

    // 4. Cilindro con textura de tablero
    const cylinder = new THREE.Mesh(
        new THREE.CylinderGeometry(1, 1, 3, 32),
        new THREE.MeshStandardMaterial({ map: texture1 })
    );
    cylinder.position.set(0, 1.5, 5);
    cylinder.castShadow = true;
    cylinder.receiveShadow = true;
    cylinder.userData = { rotationSpeed: { x: 0.01, y: 0, z: 0.02 } };
    scene.add(cylinder);
    geometricShapes.push(cylinder);

    // 5. Toro (donut) con material metálico
    const torus = new THREE.Mesh(
        new THREE.TorusGeometry(1.2, 0.4, 16, 100),
        new THREE.MeshStandardMaterial({ 
            color: 0x4ecdc4,
            roughness: 0.2,
            metalness: 0.8
        })
    );
    torus.position.set(-5, 1.5, 5);
    torus.castShadow = true;
    torus.receiveShadow = true;
    torus.userData = { rotationSpeed: { x: 0.02, y: 0.01, z: 0 } };
    scene.add(torus);
    geometricShapes.push(torus);

    // 6. Cubo adicional con textura de ladrillos
    const cube2 = new THREE.Mesh(
        new THREE.BoxGeometry(1.5, 1.5, 1.5),
        new THREE.MeshStandardMaterial({ map: texture2 })
    );
    cube2.position.set(5, 0.75, -5);
    cube2.castShadow = true;
    cube2.receiveShadow = true;
    cube2.userData = { rotationSpeed: { x: 0.015, y: 0.015, z: 0.015 } };
    scene.add(cube2);
    geometricShapes.push(cube2);

    // 7. Octaedro (forma adicional)
    const octahedron = new THREE.Mesh(
        new THREE.OctahedronGeometry(1.5),
        new THREE.MeshStandardMaterial({ 
            color: 0xf9ca24,
            roughness: 0.4,
            metalness: 0.6
        })
    );
    octahedron.position.set(-5, 2, -5);
    octahedron.castShadow = true;
    octahedron.receiveShadow = true;
    octahedron.userData = { rotationSpeed: { x: 0.01, y: 0.02, z: 0.01 } };
    scene.add(octahedron);
    geometricShapes.push(octahedron);
}

// ============================================
// PISO
// ============================================

function createFloor() {
    // Crear textura para el piso (grid pattern)
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');

    // Fondo
    context.fillStyle = '#2d3436';
    context.fillRect(0, 0, 512, 512);

    // Grid
    context.strokeStyle = '#636e72';
    context.lineWidth = 2;
    const gridSize = 64;

    for (let i = 0; i <= 512; i += gridSize) {
        context.beginPath();
        context.moveTo(i, 0);
        context.lineTo(i, 512);
        context.stroke();

        context.beginPath();
        context.moveTo(0, i);
        context.lineTo(512, i);
        context.stroke();
    }

    const floorTexture = new THREE.CanvasTexture(canvas);
    floorTexture.wrapS = THREE.RepeatWrapping;
    floorTexture.wrapT = THREE.RepeatWrapping;
    floorTexture.repeat.set(4, 4);

    const floorGeometry = new THREE.PlaneGeometry(50, 50);
    const floorMaterial = new THREE.MeshStandardMaterial({ 
        map: floorTexture,
        roughness: 0.8,
        metalness: 0.2
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    floor.receiveShadow = true;
    scene.add(floor);
}

// ============================================
// ANIMACIÓN
// ============================================

function animate() {
    requestAnimationFrame(animate);

    // Animar formas geométricas
    geometricShapes.forEach(shape => {
        if (shape.userData.rotationSpeed) {
            shape.rotation.x += shape.userData.rotationSpeed.x;
            shape.rotation.y += shape.userData.rotationSpeed.y;
            shape.rotation.z += shape.userData.rotationSpeed.z;
        }
    });

    // Actualizar controles
    controls.update();

    // Renderizar escena
    renderer.render(scene, camera);
}

// ============================================
// CAMBIO DE PERSPECTIVA
// ============================================

function switchCameraPerspective(perspectiveNumber) {
    if (cameraPositions[perspectiveNumber]) {
        currentCameraPosition = perspectiveNumber;
        
        const targetPos = cameraPositions[perspectiveNumber];
        
        // Animar transición de cámara
        const startPos = camera.position.clone();
        const startTarget = controls.target.clone();
        const duration = 1000; // 1 segundo
        const startTime = Date.now();

        function animateCamera() {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease in-out
            const eased = progress < 0.5 
                ? 2 * progress * progress 
                : -1 + (4 - 2 * progress) * progress;

            camera.position.lerpVectors(startPos, targetPos.position, eased);
            controls.target.lerpVectors(startTarget, targetPos.target, eased);

            if (progress < 1) {
                requestAnimationFrame(animateCamera);
            }
        }

        animateCamera();
    }
}

// ============================================
// EVENT LISTENERS
// ============================================

function setupEventListeners() {
    // Responsive
    window.addEventListener('resize', onWindowResize);

    // Botones de perspectiva
    document.getElementById('btn-perspective1').addEventListener('click', () => {
        switchCameraPerspective(1);
    });

    document.getElementById('btn-perspective2').addEventListener('click', () => {
        switchCameraPerspective(2);
    });

    // Teclas para cambiar perspectiva
    window.addEventListener('keydown', (event) => {
        if (event.key === '1') {
            switchCameraPerspective(1);
        } else if (event.key === '2') {
            switchCameraPerspective(2);
        }
    });
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ============================================
// INICIAR APLICACIÓN
// ============================================

init();
