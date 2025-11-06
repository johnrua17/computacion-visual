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
    // Animated UV offset
    vec2 uv = vUv;
    uv += vec2(sin(uTime * 0.5), cos(uTime * 0.3)) * 0.1;
    
    // Multi-layered noise
    float n1 = smoothNoise(uv * 5.0 + uTime * uNoiseSpeed);
    float n2 = smoothNoise(uv * 10.0 - uTime * uNoiseSpeed * 0.5);
    float n3 = smoothNoise(uv * 20.0 + uTime * uNoiseSpeed * 0.25);
    
    float noiseValue = (n1 + n2 * 0.5 + n3 * 0.25) / 1.75;
    
    // Color mixing based on noise
    vec3 color = mix(uColorA, uColorB, noiseValue);
    
    // Emissive effect based on time and position
    float emissive = sin(vPosition.y * 3.0 + uTime * 2.0) * 0.5 + 0.5;
    color += emissive * uEmissiveIntensity * 0.5;
    
    // Edge glow (Fresnel effect)
    float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
    color += fresnel * uColorA * 0.3;
    
    gl_FragColor = vec4(color, 1.0);
}