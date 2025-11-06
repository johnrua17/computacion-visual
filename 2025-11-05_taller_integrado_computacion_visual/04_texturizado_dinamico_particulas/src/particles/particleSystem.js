export class ParticleSystem {
    constructor(scene, count = 1000) {
        this.scene = scene;
        this.count = count;
        this.init();
    }

    init() {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.count * 3);
        const colors = new Float32Array(this.count * 3);
        const velocities = [];

        for (let i = 0; i < this.count; i++) {
            const i3 = i * 3;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            const radius = 3 + Math.random() * 2;

            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);

            velocities.push(new THREE.Vector3(
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02
            ));

            const color = new THREE.Color();
            color.setHSL(Math.random(), 1.0, 0.5);
            colors[i3] = color.r;
            colors[i3 + 1] = color.g;
            colors[i3 + 2] = color.b;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.8
        });

        this.particles = new THREE.Points(geometry, material);
        this.velocities = velocities;
        this.scene.add(this.particles);
    }

    update(time) {
        const positions = this.particles.geometry.attributes.position.array;
        const colors = this.particles.geometry.attributes.color.array;

        for (let i = 0; i < this.count; i++) {
            const i3 = i * 3;

            positions[i3] += this.velocities[i].x;
            positions[i3 + 1] += this.velocities[i].y;
            positions[i3 + 2] += this.velocities[i].z;

            const distance = Math.sqrt(
                positions[i3] ** 2 + 
                positions[i3 + 1] ** 2 + 
                positions[i3 + 2] ** 2
            );

            if (distance > 6 || distance < 2) {
                this.velocities[i].multiplyScalar(-1);
            }

            const hue = (time * 0.1 + i * 0.001) % 1;
            const color = new THREE.Color();
            color.setHSL(hue, 1.0, 0.5);
            colors[i3] = color.r;
            colors[i3 + 1] = color.g;
            colors[i3 + 2] = color.b;
        }

        this.particles.geometry.attributes.position.needsUpdate = true;
        this.particles.geometry.attributes.color.needsUpdate = true;
        this.particles.rotation.y += 0.001;
    }
}