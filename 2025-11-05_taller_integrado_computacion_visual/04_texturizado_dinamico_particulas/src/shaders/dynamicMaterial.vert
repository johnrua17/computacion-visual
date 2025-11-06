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
    
    // Vertex displacement with noise
    vec3 pos = position;
    float n = noise(position * 2.0 + uTime * 0.5);
    pos += normal * n * 0.1;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}