# Script de Inicio Rápido - Examen Final

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  EXAMEN FINAL - COMPUTACIÓN VISUAL" -ForegroundColor Cyan
Write-Host "  Script de Inicio Rápido" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "c:\Users\johnr\OneDrive\Documentos\GitHub\computacion-visual\examen_final"

function Show-Menu {
    Write-Host ""
    Write-Host "Selecciona una opción:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  [1] Verificar dependencias de Python" -ForegroundColor Green
    Write-Host "  [2] Instalar dependencias de Python" -ForegroundColor Green
    Write-Host "  [3] Abrir notebook de Python (Jupyter)" -ForegroundColor Green
    Write-Host "  [4] Iniciar servidor para Three.js" -ForegroundColor Blue
    Write-Host "  [5] Abrir README del proyecto" -ForegroundColor Magenta
    Write-Host "  [6] Verificar estructura del proyecto" -ForegroundColor Cyan
    Write-Host "  [7] Abrir instrucciones completas" -ForegroundColor Magenta
    Write-Host "  [0] Salir" -ForegroundColor Red
    Write-Host ""
}

function Test-PythonPackages {
    Write-Host "`nVerificando paquetes de Python..." -ForegroundColor Cyan
    
    $packages = @("cv2", "numpy", "matplotlib", "PIL")
    $missing = @()
    
    foreach ($package in $packages) {
        $result = python -c "import $package" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $package instalado" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $package NO instalado" -ForegroundColor Red
            $missing += $package
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Host "`n⚠️  Faltan paquetes. Ejecuta la opción 2 para instalarlos." -ForegroundColor Yellow
    } else {
        Write-Host "`n✓ Todos los paquetes están instalados correctamente!" -ForegroundColor Green
    }
}

function Install-PythonPackages {
    Write-Host "`nInstalando dependencias de Python..." -ForegroundColor Cyan
    
    $packages = @(
        "opencv-python",
        "numpy",
        "matplotlib",
        "pillow",
        "jupyter"
    )
    
    foreach ($package in $packages) {
        Write-Host "`nInstalando $package..." -ForegroundColor Yellow
        pip install $package
    }
    
    Write-Host "`n✓ Instalación completada!" -ForegroundColor Green
}

function Start-JupyterNotebook {
    Write-Host "`nIniciando Jupyter Notebook..." -ForegroundColor Cyan
    Set-Location "$projectPath\python"
    
    Write-Host "Abriendo: examen_final_python.ipynb" -ForegroundColor Green
    Write-Host "Presiona Ctrl+C en esta ventana para detener Jupyter" -ForegroundColor Yellow
    
    jupyter notebook examen_final_python.ipynb
}

function Start-ThreeJsServer {
    Write-Host "`nIniciando servidor HTTP para Three.js..." -ForegroundColor Cyan
    Set-Location "$projectPath\threejs"
    
    Write-Host "`n✓ Servidor iniciado en: http://localhost:8000" -ForegroundColor Green
    Write-Host "✓ Abre esta URL en tu navegador" -ForegroundColor Green
    Write-Host "✓ Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
    Write-Host ""
    
    # Intentar abrir el navegador automáticamente
    Start-Process "http://localhost:8000"
    
    python -m http.server 8000
}

function Open-README {
    Write-Host "`nAbriendo README.md..." -ForegroundColor Cyan
    Start-Process "$projectPath\README.md"
}

function Test-ProjectStructure {
    Write-Host "`nVerificando estructura del proyecto..." -ForegroundColor Cyan
    Write-Host ""
    
    $files = @{
        "Python Notebook" = "$projectPath\python\examen_final_python.ipynb"
        "Three.js HTML" = "$projectPath\threejs\index.html"
        "Three.js JS" = "$projectPath\threejs\src\main.js"
        "README principal" = "$projectPath\README.md"
        "Instrucciones" = "$projectPath\INSTRUCCIONES_COMPLETAR.md"
    }
    
    $folders = @{
        "Python data" = "$projectPath\python\data"
        "Python gifs" = "$projectPath\python\gifs"
        "Three.js src" = "$projectPath\threejs\src"
        "Three.js gifs" = "$projectPath\threejs\gifs"
        "Three.js textures" = "$projectPath\threejs\textures"
    }
    
    Write-Host "Archivos:" -ForegroundColor Yellow
    foreach ($name in $files.Keys) {
        if (Test-Path $files[$name]) {
            Write-Host "  ✓ $name" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $name (FALTA)" -ForegroundColor Red
        }
    }
    
    Write-Host "`nCarpetas:" -ForegroundColor Yellow
    foreach ($name in $folders.Keys) {
        if (Test-Path $folders[$name]) {
            Write-Host "  ✓ $name" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $name (FALTA)" -ForegroundColor Red
        }
    }
    
    Write-Host "`nImagen de animal:" -ForegroundColor Yellow
    if (Test-Path "$projectPath\python\data\animal_extincion.jpg") {
        Write-Host "  ✓ Imagen encontrada" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Imagen NO encontrada" -ForegroundColor Yellow
        Write-Host "     Descarga una imagen y guárdala como:" -ForegroundColor Yellow
        Write-Host "     python\data\animal_extincion.jpg" -ForegroundColor Cyan
    }
}

function Open-Instructions {
    Write-Host "`nAbriendo instrucciones completas..." -ForegroundColor Cyan
    Start-Process "$projectPath\INSTRUCCIONES_COMPLETAR.md"
}

# Menú principal
do {
    Show-Menu
    $choice = Read-Host "Ingresa tu opción"
    
    switch ($choice) {
        "1" { Test-PythonPackages; Pause }
        "2" { Install-PythonPackages; Pause }
        "3" { Start-JupyterNotebook }
        "4" { Start-ThreeJsServer }
        "5" { Open-README; Pause }
        "6" { Test-ProjectStructure; Pause }
        "7" { Open-Instructions; Pause }
        "0" { 
            Write-Host "`n¡Hasta luego!" -ForegroundColor Green
            break
        }
        default { 
            Write-Host "`n⚠️  Opción no válida. Intenta de nuevo." -ForegroundColor Red
            Pause
        }
    }
} while ($choice -ne "0")
