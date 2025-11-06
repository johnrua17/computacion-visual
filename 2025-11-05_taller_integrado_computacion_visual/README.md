# Taller Integral de Computación Visual

## Concepto del Proyecto

Este proyecto implementa dos módulos fundamentales del pipeline gráfico interactivo: **Texturizado Dinámico y Partículas** (Módulo 4) y **Entrada e Interacción** (Módulo 6). El objetivo es crear experiencias visuales interactivas que demuestren la capacidad de combinar shaders personalizados, sistemas de partículas y detección multimodal de entrada del usuario.

El proyecto combina:
- **Shaders personalizados** (GLSL) con efectos visuales dinámicos
- **Texturas procedimentales** generadas en tiempo real
- **Sistemas de partículas** sincronizados con materiales
- **Interacción multimodal** (teclado, mouse, touch)
- **Colisiones físicas** y detección de eventos en tiempo real
- **Interfaz de usuario** reactiva con controles interactivos

---

## Herramientas y Entorno

### Tecnologías Principales

- **Three.js r128**: Biblioteca WebGL para renderizado 3D en el navegador
- **ES6 Modules**: Organización modular del código
- **WebGL**: Renderizado acelerado por hardware
- **GLSL**: Shaders personalizados (vertex y fragment)
- **HTML5 Canvas**: Interfaz de usuario y controles
- **CSS3**: Estilos y diseño de la interfaz

---

## Módulos Implementados

### 4. Texturizado Dinámico y Partículas

**Descripción**: Implementación de materiales reactivos que cambian en tiempo real basados en shaders personalizados, junto con un sistema de partículas sincronizado.

**Características**:
- Material con shader personalizado (vertex y fragment)
- Texturas dinámicas generadas proceduralmente usando ruido
- Efectos de emisión y fresnel para iluminación de bordes
- Sistema de partículas con 1000 partículas animadas
- Controles interactivos para ajustar intensidad de emisión y velocidad del ruido
- Animación procedural del objeto principal (icosaedro)

**Archivos principales**:
- `src/main.js`: Configuración de escena y material dinámico
- `src/particles/particleSystem.js`: Sistema de partículas con física simple
- `src/shaders/dynamicMaterial.vert`: Vertex shader con desplazamiento por ruido
- `src/shaders/dynamicMaterial.frag`: Fragment shader con múltiples capas de ruido

**Efectos visuales**:
- Multi-layered noise para texturas complejas
- Vertex displacement basado en funciones de ruido
- Color mixing dinámico entre dos colores
- Efectos de emisión sincronizados con la posición del objeto

### 6. Entrada e Interacción (UI, Input y Colisiones)

**Descripción**: Sistema completo de captura de entrada multimodal (teclado, mouse, touch) con detección de colisiones físicas y una interfaz de usuario reactiva.

**Características**:
- **Input de teclado**: Sistema WASD/Arrow keys para movimiento
- **Input de mouse**: Hover detection y click en objetos
- **Input táctil**: Soporte para dispositivos móviles con drag
- **Sistema de colisiones**: Detección en tiempo real entre objetos
- **UI Canvas**: Panel de control con color picker y sliders
- **Feedback visual**: Indicadores de estado en tiempo real

**Archivos principales**:
- `src/main.js`: Escena principal y loop de animación
- `src/input/keyboard.js`: Manejador de eventos de teclado
- `src/input/mouse.js`: Manejador de eventos de mouse
- `src/input/touch.js`: Manejador de eventos táctiles
- `src/physics/collisions.js`: Sistema de detección de colisiones

**Interacciones**:
- Movimiento del objeto principal con teclado
- Rotación y escala mediante UI
- Colisiones visuales con cambio de color
- Contador de colisiones en tiempo real

---

## Estructura del Proyecto

```
2025-11-05_taller_integrado_computacion_visual/
├── 04_texturizado_dinamico_particulas/
│   ├── assets/
│   │   └── styles.css
│   ├── src/
│   │   ├── main.js
│   │   ├── particles/
│   │   │   └── particleSystem.js
│   │   └── shaders/
│   │       ├── dynamicMaterial.vert
│   │       └── dynamicMaterial.frag
│   └── index.html
├── 06_entrada_interaccion/
│   ├── assets/
│   │   └── styles.css
│   ├── src/
│   │   ├── main.js
│   │   ├── input/
│   │   │   ├── keyboard.js
│   │   │   ├── mouse.js
│   │   │   └── touch.js
│   │   ├── physics/
│   │   │   └── collisions.js
│   │   └── ui/
│   │       └── controls.js
│   └── index.html
├── renders/              # Evidencias visuales (GIFs, imágenes)
├── taller_3.md           # Especificaciones del taller
└── README.md             # Este archivo
```

---

## Instrucciones de Uso

### Requisitos Previos

- Navegador web moderno con soporte para WebGL (Chrome, Firefox, Edge, Safari)
- Servidor web local (opcional, pero recomendado para evitar problemas CORS)

### Ejecución Local

#### Opción 1: Servidor HTTP Simple (Python)
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

#### Opción 2: Servidor HTTP Simple (Node.js)
```bash
npx http-server
```

#### Opción 3: Live Server (VS Code)
- Instalar extensión "Live Server"
- Click derecho en `index.html` → "Open with Live Server"

### Acceder a los Módulos

1. **Texturizado Dinámico y Partículas**:
   - Abrir `04_texturizado_dinamico_particulas/index.html`
   - Usar los controles para ajustar:
     - Emissive Intensity (0-3)
     - Noise Speed (0-3)
   - Botones: Pause/Play y Reset

2. **Entrada e Interacción**:
   - Abrir `06_entrada_interaccion/index.html`
   - **Controles de teclado**:
     - WASD / Arrow Keys: Mover objeto
     - Space: Reset posición
     - R: Rotar objeto
   - **Mouse**: Hover sobre esferas para efectos
   - **Touch**: Arrastrar en dispositivos móviles
   - **UI**: Usar color picker y slider de escala

---

## Evidencias Visuales

### Módulo 4: Texturizado Dinámico y Partículas
- [ ] `renders/04_texture_animated.gif`: Animación del material dinámico
- [ ] `renders/04_particles_closeup.png`: Vista detallada del sistema de partículas
- [ ] `renders/04_shader_variations.png`: Diferentes variaciones del shader

### Módulo 6: Entrada e Interacción
- [ ] `renders/06_keyboard_interaction.gif`: Interacción con teclado
- [ ] `renders/06_collision_detection.gif`: Detección de colisiones
- [ ] `renders/06_ui_controls.png`: Panel de control y UI

### Video General
- [ ] `renders/demo_video.mp4`: Video completo de ambos módulos (30-60 segundos)

---

## Código Relevante

### Ejemplo: Shader Dinámico (Fragment)
```glsl
uniform float uTime;
uniform float uNoiseSpeed;
uniform float uEmissiveIntensity;

void main() {
    // Multi-layered noise
    float n1 = smoothNoise(uv * 5.0 + uTime * uNoiseSpeed);
    float n2 = smoothNoise(uv * 10.0 - uTime * uNoiseSpeed * 0.5);
    float noiseValue = (n1 + n2 * 0.5) / 1.5;
    
    // Color mixing
    vec3 color = mix(uColorA, uColorB, noiseValue);
    
    // Emissive effect
    float emissive = sin(vPosition.y * 3.0 + uTime * 2.0) * 0.5 + 0.5;
    color += emissive * uEmissiveIntensity * 0.5;
    
    gl_FragColor = vec4(color, 1.0);
}
```

### Ejemplo: Sistema de Colisiones
```javascript
check() {
    const collisions = [];
    const mainPos = this.mainObject.position;
    
    this.objects.forEach(obj => {
        const distance = mainPos.distanceTo(obj.position);
        if (distance < this.threshold) {
            collisions.push({
                id: obj.userData.id,
                distance: distance
            });
            // Visual feedback
            obj.material.color.setHex(0xff0000);
        }
    });
    
    return collisions;
}
```

---

## Reflexión y Aprendizajes

### Retos Técnicos Enfrentados

1. **Compatibilidad de Módulos ES6 con Three.js CDN**:
   - Problema: THREE.js cargado desde CDN no estaba disponible en el contexto de módulos ES6
   - Solución: Implementación de un sistema de espera asíncrona que verifica la disponibilidad de THREE antes de importar módulos

2. **Sincronización de Partículas con Shaders**:
   - Desafío: Coordinar la animación de partículas con los efectos del material dinámico
   - Solución: Sistema de tiempo unificado (`uTime` uniform) compartido entre shader y partículas

3. **Detección de Colisiones en Tiempo Real**:
   - Reto: Optimizar la detección de colisiones para múltiples objetos sin afectar el rendimiento
   - Implementación: Sistema de threshold distance con actualización eficiente de geometrías

### Mejoras Futuras

1. **Optimización de Rendimiento**:
   - Implementar instancing para partículas
   - Usar Web Workers para cálculos pesados
   - Implementar frustum culling para objetos fuera de vista

2. **Expansión de Interacciones**:
   - Mejorar la detección de colisiones con bounding boxes
   - Añadir más tipos de input (gamepad, gestos)
   - Implementar sistema de eventos más robusto

3. **Mejoras Visuales**:
   - Añadir post-processing effects (bloom, motion blur)
   - Implementar sistema de sombras más avanzado
   - Mejorar los shaders con efectos más complejos

4. **Arquitectura**:
   - Implementar un sistema de eventos más robusto
   - Mejorar la organización modular del código
   - Añadir sistema de configuración para parámetros

### Aprendizajes Clave

- **Pipeline Gráfico**: Comprensión profunda del flujo de datos desde geometría hasta píxel en pantalla
- **Shaders GLSL**: Dominio de técnicas de programación de GPU para efectos visuales avanzados
- **Interacción Multimodal**: Diseño de sistemas que responden a múltiples tipos de entrada simultáneamente
- **Arquitectura Modular**: Organización de código en módulos reutilizables y mantenibles

---

## Criterios de Evaluación

| Criterio                                | Estado | Notas |
| --------------------------------------- | ------ | ----- |
| Organización                            | ✅     | Estructura clara y README completo |
| Shaders y texturizado dinámico          | ✅     | Módulo 4: Shaders personalizados con ruido procedural |
| Interacción multimodal                  | ✅     | Módulo 6: Teclado, mouse y touch implementados |
| Animaciones y partículas                | ✅     | Módulo 4: Sistema de 1000 partículas sincronizado |
| UI e interacción                        | ✅     | Módulo 6: Panel de control y detección de colisiones |
| Evidencias visuales                     | ⏳     | Pendiente generar GIFs y videos |
| Código y documentación                  | ✅     | Código comentado, estructura modular |

---

## Referencias

- [Three.js Documentation](https://threejs.org/docs/)
- [WebGL Fundamentals](https://webglfundamentals.org/)
- [GLSL Reference](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
- [WebGL Shader Tutorial](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html)
- [Three.js ShaderMaterial](https://threejs.org/docs/#api/en/materials/ShaderMaterial)

---

## Licencia

Este proyecto es parte de un taller académico de Computación Visual.

---

## Conclusión

Este proyecto demuestra la integración exitosa de shaders personalizados, sistemas de partículas y detección multimodal de entrada del usuario. Los módulos 4 y 6 trabajan en conjunto para crear experiencias visuales interactivas que combinan técnicas avanzadas de renderizado con interacción intuitiva.

**Proyecto desarrollado como parte del Taller Integral de Computación Visual.** 🎨✨

