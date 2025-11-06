export class CollisionSystem {
    constructor(mainObject, collisionObjects) {
        this.mainObject = mainObject;
        this.collisionObjects = collisionObjects;
    }

    check() {
        const collisions = [];
        const mainBox = new THREE.Box3().setFromObject(this.mainObject);
        
        this.collisionObjects.forEach(obj => {
            const objBox = new THREE.Box3().setFromObject(obj);
            
            if (mainBox.intersectsBox(objBox)) {
                obj.material.emissive.setHex(0xff0000);
                collisions.push(obj);
            }
        });
        
        return collisions;
    }
}