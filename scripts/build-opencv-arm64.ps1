<#
Builds OpenCV natively for Windows on ARM (ARM64) and installs cv2 into a venv.

PyPI has no ARM64 Windows wheel for opencv-python, so the beta/windows-arm64
branch compiles it once. Only the modules camera.py needs are built, statically,
into a single cv2.pyd (about 15-30 min on a Snapdragon X).

Needs: Visual Studio 2022 Build Tools with "Desktop development with C++",
"MSVC ARM64 build tools" and a Windows 11 SDK, plus git and an ARM64 Python.

Usage (from the repo root):
    py -3.12-arm64 -m venv .venv-arm64
    .\scripts\build-opencv-arm64.ps1
#>
param(
    [string]$Venv = ".venv-arm64",
    [string]$Version = "5.0.0",
    [string]$WorkDir = "..\opencv-arm64"
)
$ErrorActionPreference = "Stop"

$python = Resolve-Path (Join-Path $Venv "Scripts\python.exe")
if ((& $python -c "import platform; print(platform.machine())") -ne "ARM64") {
    throw "$python is not an ARM64 Python. Create the venv with: py -3.12-arm64 -m venv $Venv"
}
& $python -m pip install --quiet --upgrade numpy cmake ninja
$scripts = Split-Path $python
$env:PATH = "$scripts;$env:PATH"

# Load the MSVC ARM64 compiler environment into this PowerShell session.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
if (-not $vs) { throw "Visual Studio Build Tools with the ARM64 C++ tools are not installed." }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvarsall.bat`" arm64 >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] }
}

New-Item -ItemType Directory -Force $WorkDir | Out-Null
$src = Join-Path $WorkDir "opencv"
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $Version https://github.com/opencv/opencv $src
}
$build = Join-Path $WorkDir "build"
# Paths handed to CMake use forward slashes: it reads "\U" in "C:\Users" as an escape.
$sitePackages = (& $python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])") -replace "\\", "/"
# Point CMake at this Python's own headers and import library; otherwise it may find
# an x64 Python from the registry and the ARM64 cv2.pyd fails to link.
$pyBase = (& $python -c "import sys; print(sys.base_prefix)") -replace "\\", "/"
$pyLib = & $python -c "import sys; print('python%d%d.lib' % sys.version_info[:2])"

cmake --fresh -S $src -B $build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_LIST="core,imgproc,imgcodecs,videoio,highgui,python3" `
    -DBUILD_SHARED_LIBS=OFF `
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF `
    -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF `
    -DWITH_IPP=OFF -DWITH_FFMPEG=OFF -DWITH_MSMF=ON -DWITH_DSHOW=ON -DWITH_OPENCL=ON `
    -DPYTHON3_EXECUTABLE="$python" `
    -DPYTHON3_INCLUDE_DIR="$pyBase/include" `
    -DPYTHON3_LIBRARY="$pyBase/libs/$pyLib" `
    -DOPENCV_PYTHON3_INSTALL_PATH="$sitePackages" `
    -DCMAKE_INSTALL_PREFIX="$(Join-Path $WorkDir 'install')"
if ($LASTEXITCODE) { throw "CMake configure failed." }

cmake --build $build --config Release
if ($LASTEXITCODE) { throw "Build failed." }
cmake --install $build --config Release
if ($LASTEXITCODE) { throw "Install failed." }

& $python -c "import cv2, platform; print('cv2', cv2.__version__, 'on', platform.machine())"
