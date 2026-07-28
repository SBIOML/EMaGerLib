<#
.SYNOPSIS
    Signs the application binary and programs it into the STM32N6 external OSPI flash.

.DESCRIPTION
    The N6 refuses unsigned images, silently -- the board simply does not start, with no
    error anywhere. So signing is not an optional extra step, and it must happen on the
    binary you just built rather than on whatever was signed last time.

    That last point is the reason this script exists rather than two commands in the
    README: reflashing a stale signed image after rebuilding is the single most common
    way to spend an hour debugging a change that was never on the board. The script
    refuses to flash if the signed file is older than the binary.

.PARAMETER Bin
    The freshly built .bin. Default: build\emager-n6.bin

.PARAMETER Address
    Where to program it. Default 0x70100000 (the application slot; the FSBL lives at
    0x70000000 and is programmed once, separately).

.PARAMETER Loader
    External loader .stldr. Auto-detected if omitted.

.PARAMETER SignOnly
    Sign but do not flash.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sign-and-flash.ps1
    powershell -ExecutionPolicy Bypass -File scripts\sign-and-flash.ps1 -Bin build\other.bin -SignOnly
#>

param(
    [string]$Bin      = "build\emager-n6.bin",
    [string]$Address  = "0x70100000",
    [string]$Loader   = "",
    [switch]$SignOnly
)

$ErrorActionPreference = "Stop"

$progDir = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
$progCli = if (Get-Command "STM32_Programmer_CLI" -ErrorAction SilentlyContinue) {
    "STM32_Programmer_CLI"
} else {
    Join-Path $progDir "STM32_Programmer_CLI.exe"
}
$signCli = if (Get-Command "STM32_SigningTool_CLI" -ErrorAction SilentlyContinue) {
    "STM32_SigningTool_CLI"
} else {
    Join-Path $progDir "STM32_SigningTool_CLI.exe"
}

if (-not (Test-Path $Bin)) {
    throw "Binary not found: $Bin`nBuild first (Ctrl+Shift+B in VS Code, or 'cmake --build build')."
}
foreach ($t in @($progCli, $signCli)) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
        throw "Not found: $t`nInstall STM32CubeProgrammer (doc 1, step 1)."
    }
}

$signed = [IO.Path]::ChangeExtension($Bin, $null) + "_signed.bin"

# --- sign --------------------------------------------------------------------
# -nk  : no key. Development signature. A real product needs the N6 secure boot chain
#        with actual ECC keys -- a separate subject, and not one to discover at the end
#        of a project.
# -t ssbl : the image is a second stage boot loader, i.e. the application.
# -hv 2.3 : header version used by current ST examples. If the tool rejects it, run
#        STM32_SigningTool_CLI --help for the versions your build accepts.
Write-Host "Signing $Bin ..." -ForegroundColor Cyan
& $signCli -bin $Bin -nk -t ssbl -hv 2.3 -o $signed
if ($LASTEXITCODE -ne 0) { throw "Signing failed (exit $LASTEXITCODE)." }
Write-Host "  -> $signed" -ForegroundColor Green

# The guard. Signing just ran, so this should never trip -- unless the signing tool
# reported success without writing, which it can do on a permission problem.
$binTime    = (Get-Item $Bin).LastWriteTime
$signedTime = (Get-Item $signed).LastWriteTime
if ($signedTime -lt $binTime) {
    throw "The signed image is older than the binary. Refusing to flash a stale image."
}

if ($SignOnly) {
    Write-Host "Done (sign only)." -ForegroundColor Green
    exit 0
}

# --- external loader ---------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Loader)) {
    $elDir = Join-Path $progDir "ExternalLoader"
    $candidates = @(Get-ChildItem -Path $elDir -Filter "*N6*NUCLEO*.stldr" -ErrorAction SilentlyContinue)
    if ($candidates.Count -eq 0) {
        $candidates = @(Get-ChildItem -Path $elDir -Filter "*N6*.stldr" -ErrorAction SilentlyContinue)
    }
    if ($candidates.Count -eq 0) {
        throw "No N6 external loader found in $elDir. Update STM32CubeProgrammer, or pass -Loader explicitly."
    }
    if ($candidates.Count -gt 1) {
        # Do not silently pick one: the loaders are board-specific and the wrong one
        # fails in confusing ways part-way through an erase.
        Write-Host "Several N6 loaders found:" -ForegroundColor Yellow
        $candidates | ForEach-Object { Write-Host "    $($_.Name)" }
        Write-Host "Using $($candidates[0].Name). If your board is not that one, pass -Loader." -ForegroundColor Yellow
    }
    $Loader = $candidates[0].FullName
}
Write-Host "Loader: $(Split-Path $Loader -Leaf)" -ForegroundColor Cyan

# --- flash -------------------------------------------------------------------
Write-Host "Programming $signed at $Address ..." -ForegroundColor Cyan
& $progCli -c port=SWD mode=HOTPLUG -el "$Loader" -hardRst -w $signed $Address
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Programming failed. Check, in this order:" -ForegroundColor Red
    Write-Host "  1. Boot jumpers in DEVELOPMENT boot, and RESET pressed after changing them"
    Write-Host "  2. USB-C cable (short, good quality) and the ST-LINK connector"
    Write-Host "  3. ST-LINK firmware up to date (STM32CubeProgrammer GUI -> Firmware upgrade)"
    Write-Host "See doc 6."
    exit 1
}

Write-Host ""
Write-Host "Flashed." -ForegroundColor Green
Write-Host "To run it standalone: set the jumpers to BOOT FROM FLASH, then press RESET." -ForegroundColor Yellow
Write-Host "A firmware that works under the debugger can still fail to boot from flash -- verify it." -ForegroundColor Yellow
