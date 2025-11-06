export class MouseInput {
    constructor(camera, objects, domElement) {
        this.camera = camera;
        this.objects = objects;
        this.domElement = domElement;
        this.position = { x: 0, y: 0 };
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.init();
    }

    init() {
        this.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.domElement.addEventListener('click', (e) => this.onClick(e));
    }

    onMouseMove(e) {
        this.position.x = e.clientX;
        this.position.y = e.clientY;
        
        this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.objects);

        this.objects.forEach(obj => {
            obj.material.color.copy(obj.userData.originalColor);
            obj.material.emissive.setHex(0x000000);
        });

        if (intersects.length > 0) {
            intersects[0].object.material.emissive.setHex(0xffff00);
            this.domElement.style.cursor = 'pointer';
        } else {
            this.domElement.style.cursor = 'default';
        }
    }

    onClick(e) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.objects);

        if (intersects.length > 0) {
            const obj = intersects[0].object;
            obj.material.color.setHex(Math.random() * 0xffffff);
            obj.userData.originalColor = obj.material.color.clone();
            
            const scale = { value: 1 };
            const animate = () => {
                scale.value += 0.1;
                obj.scale.setScalar(scale.value);
                if (scale.value < 1.5) {
                    requestAnimationFrame(animate);
                } else {
                    obj.scale.setScalar(1);
                }
            };
            animate();
        }
    }
}