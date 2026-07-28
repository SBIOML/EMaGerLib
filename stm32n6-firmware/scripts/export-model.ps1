<#
.SYNOPSIS
    Regenerates the model artifacts from EMaGerLib and copies them into model\.

.DESCRIPTION
    Runs export_model.py and make_test_vectors.py, then copies everything the firmware
    needs into model\. Keeping the two in one script matters: the test vectors are only
    valid for the exact export that produced them, so generating one without the other
    leaves the firmware checking itself against a model it is not running.

.PARAMETER EMaGerLib
    Path to your EMaGerLib clone. Default: ..\EMaGerLib

.PARAMETER Model
    Model class to export. Default: EmagerCNNProtoRingStrided (the deployment target --
    its prototypes can be recomputed on the device without any backprop).

.PARAMETER Checkpoint
    Reuse existing FP32 weights instead of training fresh. Strongly recommended once you
    have a model you like: training is not seeded, so omitting this gives DIFFERENT
    weights every run and invalidates anything already flashed.

.PARAMETER NumVectors
    How many test vectors to embed. Default 32.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\export-model.ps1
    powershell -ExecutionPolicy Bypass -File scripts\export-model.ps1 -Checkpoint model\model_fp32.pt
#>

param(
    [string]$EMaGerLib  = "..\EMaGerLib",
    [string]$Model      = "EmagerCNNProtoRingStrided",
    [string]$Checkpoint = "",
    [int]   $NumVectors = 32
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EMaGerLib)) {
    throw "EMaGerLib not found at $EMaGerLib -- pass -EMaGerLib <path>."
}
$lib = (Resolve-Path $EMaGerLib).Path
$exportScript = Join-Path $lib "examples\deployment\export_model.py"
if (-not (Test-Path $exportScript)) {
    throw "$exportScript not found -- is $lib really an EMaGerLib clone?"
}

$outDir = Join-Path $PSScriptRoot "..\model"
$outDir = [IO.Path]::GetFullPath($outDir)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Push-Location $lib
try {
    # Not $args: that is an automatic variable in PowerShell and assigning to it inside
    # a script is a good way to produce behaviour nobody can explain later.
    $pyArgs = @("examples\deployment\export_model.py", "--model", $Model)
    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $ck = (Resolve-Path $Checkpoint).Path
        $pyArgs += @("--checkpoint", $ck)
        Write-Host "Exporting $Model from $ck ..." -ForegroundColor Cyan
    } else {
        # Worth saying out loud: an unseeded retrain means the artifacts about to be
        # produced have nothing to do with whatever is on the board right now.
        Write-Host "Exporting $Model (training fresh -- weights will DIFFER from any previous export) ..." -ForegroundColor Yellow
    }
    & python @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "export_model.py failed (exit $LASTEXITCODE)." }

    Write-Host "Generating test vectors ..." -ForegroundColor Cyan
    & python "examples\deployment\make_test_vectors.py" --model $Model --num $NumVectors
    if ($LASTEXITCODE -ne 0) { throw "make_test_vectors.py failed (exit $LASTEXITCODE)." }

    $src = Join-Path $lib "examples\deployment\exported\$Model"
    if (-not (Test-Path $src)) { throw "Export directory not found: $src" }

    $wanted = @(
        "model_int8.tflite",
        "emager_model_params.h",
        "emager_prototypes.h",
        "emager_proto.h", "emager_proto.c",
        "emager_selftest.h", "emager_selftest.c",
        "emager_test_vectors.h",
        "model_fp32.pt",          # keep the weights WITH the firmware or they are lost
        "EXPORT_REPORT.md"
    )
    foreach ($f in $wanted) {
        $p = Join-Path $src $f
        if (Test-Path $p) {
            Copy-Item $p $outDir -Force
            Write-Host "  copied $f"
        } else {
            Write-Host "  MISSING $f" -ForegroundColor Yellow
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Artifacts in $outDir" -ForegroundColor Green
Write-Host "Next: compile model_int8.tflite for the NPU (doc 4), then rebuild the firmware." -ForegroundColor Yellow
Write-Host "Both matter -- a regenerated model with a stale generated network produces" -ForegroundColor Yellow
Write-Host "wrong embeddings, and the self-test is what will tell you." -ForegroundColor Yellow
