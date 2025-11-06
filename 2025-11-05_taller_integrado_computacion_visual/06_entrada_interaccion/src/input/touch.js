export class TouchInput {
    constructor(camera, targetObject, domElement) {
        this.camera = camera;
        this.target = targetObject;
        this.domElement = domElement;
        this.isActive = false;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.init();
    }

    init() {
        this.domElement.addEventListener('touchstart', (e) => this.onTouchStart(e));
        this.domElement.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
        this.domElement.addEventListener('touchend', () => this.onTouchEnd());
    }

    onTouchStart(e) {
        this.isActive = true;
        const touch = e.touches[0];
        this.updateMousePosition(touch.clientX, touch.clientY);
    }

    onTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.updateMousePosition(touch.clientX, touch.clientY);
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        const intersection = new THREE.Vector3();
        this.raycaster.ray.intersectPlane(plane, intersection);
        
        if (intersection) {
            this.target.position.x = intersection.x;
            this.target.position.y = intersection.y;
        }
    }

    onTouchEnd() {
        this.isActive = false;
    }

    updateMousePosition(x, y) {
        this.mouse.x = (x / window.innerWidth) * 2 - 1;
        this.mouse.y = -(y / window.innerHeight) * 2 + 1;
    }
}