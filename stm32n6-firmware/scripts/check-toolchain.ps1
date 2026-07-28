<#
.SYNOPSIS
    Checks that everything document 1 asks you to install is actually installed,
    findable, and talking to the board.

.DESCRIPTION
    Every check is one you would otherwise make the hard way -- by hitting a confusing
    failure ten minutes into a build or, worse, at the moment of flashing. Run this
    after installing, and again whenever something stops working: it is much faster to
    rule the toolchain out than to suspect it.

    Nothing here modifies your system. It only looks.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check-toolchain.ps1
#>

$ErrorActionPreference = "Continue"

$script:Failures = 0
$script:Warnings = 0

function Write-Ok      ($m) { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Write-Fail    ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:Failures++ }
function Write-Warn    ($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:Warnings++ }
function Write-Section ($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }

function Test-Tool {
    param(
        [string]$Name,
        [string]$Exe,
        [string]$VersionArg = "--version",
        [string]$Hint
    )
    $cmd = Get-Command $Exe -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Fail "$Name : '$Exe' not on PATH. $Hint"
        return $null
    }
    $ver = & $Exe $VersionArg 2>&1 | Select-Object -First 1
    Write-Ok "$Name : $ver"
    return $cmd.Source
}

Write-Host "EMaGer / STM32N6 toolchain check" -ForegroundColor White

# --- STM32CubeProgrammer -----------------------------------------------------
Write-Section "STM32CubeProgrammer"

$progDir = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
$progCli = Join-Path $progDir "STM32_Programmer_CLI.exe"
$signCli = Join-Path $progDir "STM32_SigningTool_CLI.exe"

if (Test-Path $progCli) {
    $v = & $progCli --version 2>&1 | Select-String -Pattern "version" | Select-Object -First 1
    Write-Ok "STM32_Programmer_CLI found ($v)"
} else {
    Write-Fail "STM32_Programmer_CLI not found at $progCli -- install STM32CubeProgrammer (doc 1, step 1)"
}

# The signing tool is the one people discover is missing at the worst moment: the
# STM32N6 will not boot an unsigned image, and nothing says so until the board is
# silent.
if (Test-Path $signCli) {
    Write-Ok "STM32_SigningTool_CLI found (required -- the N6 refuses unsigned images)"
} else {
    Write-Fail "STM32_SigningTool_CLI not found. It ships with STM32CubeProgrammer; a partial install will bite you at flash time."
}

if (Get-Command "STM32_Programmer_CLI" -ErrorAction SilentlyContinue) {
    Write-Ok "STM32_Programmer_CLI is on PATH"
} else {
    Write-Warn "STM32_Programmer_CLI is not on PATH -- the flash scripts fall back to the default install path, but adding it (doc 1) is cleaner"
}

# --- external loaders --------------------------------------------------------
Write-Section "External loaders (OSPI flash)"

$elDir = Join-Path $progDir "ExternalLoader"
if (Test-Path $elDir) {
    $loaders = Get-ChildItem -Path $elDir -Filter "*N6*.stldr" -ErrorAction SilentlyContinue
    if ($loaders) {
        Write-Ok "$($loaders.Count) N6 loader(s) available:"
        $loaders | ForEach-Object { Write-Host "         $($_.Name)" }
        Write-Host "         (the NUCLEO-N657X0-Q uses the one naming that board -- do not guess, match the name)"
    } else {
        Write-Fail "No *N6*.stldr in $elDir -- your STM32CubeProgrammer predates STM32N6 support. Update it."
    }
} else {
    Write-Fail "No ExternalLoader directory at $elDir"
}

# --- build toolchain ---------------------------------------------------------
Write-Section "Build toolchain (STM32CubeCLT)"

Test-Tool -Name "arm-none-eabi-gcc" -Exe "arm-none-eabi-gcc" `
          -Hint "Install STM32CubeCLT and accept its 'add to PATH' option (doc 1, step 3). Open a NEW terminal afterwards." | Out-Null
Test-Tool -Name "CMake" -Exe "cmake" `
          -Hint "Comes with STM32CubeCLT." | Out-Null
Test-Tool -Name "Ninja" -Exe "ninja" `
          -Hint "Comes with STM32CubeCLT." | Out-Null

$cltDirs = Get-ChildItem -Path "C:\ST" -Directory -Filter "STM32CubeCLT*" -ErrorAction SilentlyContinue
if ($cltDirs) {
    Write-Ok "STM32CubeCLT installed: $($cltDirs[-1].FullName)"
} else {
    Write-Warn "No C:\ST\STM32CubeCLT* directory -- fine if you installed it elsewhere, but VS Code will need the path (doc 2)"
}

# --- CubeMX ------------------------------------------------------------------
Write-Section "STM32CubeMX"

$mx = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeMX\STM32CubeMX.exe"
if (Test-Path $mx) { Write-Ok "STM32CubeMX found" }
else { Write-Warn "STM32CubeMX not at the default path -- fine if installed elsewhere" }

# --- X-CUBE-AI / stedgeai ----------------------------------------------------
Write-Section "X-CUBE-AI / ST Edge AI Core"

# The version directory changes with every update, which is exactly why hard-coded
# paths to stedgeai rot. Search instead.
$packRoot = Join-Path $env:USERPROFILE "STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI"
$stedgeai = $null
if (Test-Path $packRoot) {
    $stedgeai = Get-ChildItem -Path $packRoot -Recurse -Filter "stedgeai.exe" -ErrorAction SilentlyContinue |
                Sort-Object FullName | Select-Object -Last 1
}
if ($null -eq $stedgeai) {
    $stedgeai = Get-Command "stedgeai" -ErrorAction SilentlyContinue |
                ForEach-Object { Get-Item $_.Source }
}

if ($stedgeai) {
    Write-Ok "stedgeai : $($stedgeai.FullName)"
    $v = & $stedgeai.FullName --version 2>&1 | Select-Object -First 1
    Write-Host "         $v"
    Write-Host "         (set STEDGEAI_EXE to this path to skip the search in other scripts)"
} else {
    Write-Fail "stedgeai not found. Install X-CUBE-AI >= 10.0 from CubeMX: Help -> Manage embedded software packages (doc 1, step 4)."
}

# --- Python / EMaGerLib ------------------------------------------------------
Write-Section "Python (model export)"

$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) {
    $pv = & python --version 2>&1
    Write-Ok "Python : $pv"
    & python -c "import torch, onnx, tensorflow" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Export dependencies present (torch, onnx, tensorflow)"
    } else {
        Write-Warn "Export dependencies missing -- run 'pip install -e `".[deploy]`"' in your EMaGerLib clone. Only needed to (re)generate the model."
    }
} else {
    Write-Warn "Python not on PATH -- only needed to (re)generate the model from EMaGerLib"
}

# --- the board ---------------------------------------------------------------
Write-Section "Board"

if (Test-Path $progCli) {
    Write-Host "  Connecting (SWD, hotplug)..."
    $out = & $progCli -c port=SWD mode=HOTPLUG 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Board responds"
        $out | Select-String -Pattern "Device ID|ST-LINK FW|Board" | ForEach-Object { Write-Host "         $_" }
    } else {
        Write-Fail "No connection. In order of likelihood: boot jumpers not in Development boot (then RESET), poor/long USB-C cable, wrong USB connector, outdated ST-LINK firmware. See doc 6."
    }
} else {
    Write-Warn "Skipped -- STM32_Programmer_CLI not available"
}

# --- verdict -----------------------------------------------------------------
Write-Host ""
if ($script:Failures -eq 0 -and $script:Warnings -eq 0) {
    Write-Host "All checks passed. Continue with doc 2 (VS Code)." -ForegroundColor Green
} elseif ($script:Failures -eq 0) {
    Write-Host "$($script:Warnings) warning(s), nothing blocking. Continue with doc 2 (VS Code)." -ForegroundColor Yellow
} else {
    Write-Host "$($script:Failures) blocking problem(s), $($script:Warnings) warning(s)." -ForegroundColor Red
    Write-Host "Fix the [FAIL] lines before going further -- each one stops the build or the flash later on." -ForegroundColor Red
    exit 1
}
