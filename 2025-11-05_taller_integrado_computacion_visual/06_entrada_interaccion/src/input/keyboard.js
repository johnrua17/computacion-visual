export class KeyboardInput {
    constructor(targetObject) {
        this.target = targetObject;
        this.keysPressed = new Set();
        this.speed = 0.3;
        
        this.init();
    }

    init() {
        window.addEventListener('keydown', (e) => this.onKeyDown(e));
        window.addEventListener('keyup', (e) => this.onKeyUp(e));
    }

    onKeyDown(e) {
        const key = e.key.toLowerCase();
        this.keysPressed.add(key);

        switch(key) {
            case 'w':
            case 'arrowup':
                this.target.position.y += this.speed;
                break;
            case 's':
            case 'arrowdown':
                this.target.position.y -= this.speed;
                break;
            case 'a':
            case 'arrowleft':
                this.target.position.x -= this.speed;
                break;
            case 'd':
            case 'arrowright':
                this.target.position.x += this.speed;
                break;
            case ' ':
                this.target.position.set(0, 0, 0);
                break;
            case 'r':
                this.target.rotation.x += 0.5;
                this.target.rotation.y += 0.5;
                break;
        }
    }

    onKeyUp(e) {
        this.keysPressed.delete(e.key.toLowerCase());
    }

    getActiveKeys() {
        return this.keysPressed.size > 0 ? Array.from(this.keysPressed).join(', ') : '';
    }
}