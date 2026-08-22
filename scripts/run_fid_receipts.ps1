[CmdletBinding()]
param(
    [string]$Python,
    [ValidateRange(2, 10000)]
    [int]$SampleCount = 2048,
    [switch]$AllowDirty
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $repo

function Resolve-PythonInterpreter {
    param(
        [Parameter(Mandatory)]
        [string]$Candidate,
        [Parameter(Mandatory)]
        [bool]$IsDefault
    )

    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    $command = Get-Command -Name $Candidate -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $command -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
        return $command.Source
    }

    if ($IsDefault) {
        $message = @"
Default Python interpreter not found at '$Candidate'.
Create the project environment from '$repo':
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Then rerun this script, or pass -Python <path-to-python.exe>.
"@
        throw $message.Trim()
    }

    throw "Python interpreter supplied with -Python was not found: '$Candidate'. Pass an existing python.exe path or an executable available on PATH."
}

function Invoke-GitText {
    param(
        [Parameter(Mandatory)]
        [string[]]$GitArguments
    )

    $output = @(& git -C $repo @GitArguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $detail = ($output -join [Environment]::NewLine).Trim()
        throw "git $($GitArguments -join ' ') failed with exit code $exitCode. $detail"
    }
    return ($output -join [Environment]::NewLine).Trim()
}

function Get-RepositorySnapshot {
    $head = Invoke-GitText -GitArguments @("rev-parse", "--verify", "HEAD")
    if ($head -notmatch "^[0-9a-fA-F]{40}$") {
        throw "git rev-parse returned an invalid commit id: '$head'"
    }

    $status = Invoke-GitText -GitArguments @(
        "status",
        "--porcelain=v1",
        "--untracked-files=all"
    )
    return [pscustomobject]@{
        Commit = $head
        Dirty = -not [string]::IsNullOrWhiteSpace($status)
        Status = $status
    }
}

function Assert-RepositoryState {
    param(
        [Parameter(Mandatory)]
        [string]$ExpectedCommit,
        [Parameter(Mandatory)]
        [string]$Stage
    )

    $snapshot = Get-RepositorySnapshot
    if ($snapshot.Commit -cne $ExpectedCommit) {
        throw "$Stage changed HEAD from $ExpectedCommit to $($snapshot.Commit); receipts are not publishable."
    }
    if ($snapshot.Dirty) {
        throw "$Stage left the worktree dirty; receipts are not publishable. git status: $($snapshot.Status)"
    }
}

function Get-FileFingerprint {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    $item = Get-Item -LiteralPath $Path
    return [pscustomobject]@{
        Path = $item.FullName
        Bytes = [long]$item.Length
        Sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName).Hash.ToLowerInvariant()
    }
}

function Get-SourceFingerprint {
    $fingerprint = [ordered]@{}
    foreach ($name in @("fid.py", "diffusion.py", "model.py", "data_utils.py")) {
        $fingerprint[$name] = Get-FileFingerprint -Path (Join-Path $repo $name)
    }
    return ,$fingerprint
}

function Assert-LocalFileFingerprint {
    param(
        [Parameter(Mandatory)]
        [psobject]$Expected,
        [Parameter(Mandatory)]
        [string]$Label
    )

    $actual = Get-FileFingerprint -Path $Expected.Path
    if ($actual.Bytes -ne $Expected.Bytes -or $actual.Sha256 -cne $Expected.Sha256) {
        throw "$Label changed during the run; expected $($Expected.Sha256), found $($actual.Sha256)."
    }
}

function Assert-LocalSourceFingerprint {
    param(
        [Parameter(Mandatory)]
        [System.Collections.IDictionary]$Expected
    )

    foreach ($name in $Expected.Keys) {
        Assert-LocalFileFingerprint -Expected $Expected[$name] -Label "source file $name"
    }
}

function Get-RequiredProperty {
    param(
        [AllowNull()]
        [object]$Object,
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($null -eq $Object) {
        throw "$Context is missing."
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Context is missing required property '$Name'."
    }
    return $property.Value
}

function Assert-ExactPropertyNames {
    param(
        [AllowNull()]
        [object]$Object,
        [Parameter(Mandatory)]
        [string[]]$ExpectedNames,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($null -eq $Object) {
        throw "$Context is missing."
    }
    $actual = @($Object.PSObject.Properties.Name | Sort-Object)
    $expected = @($ExpectedNames | Sort-Object)
    $actualText = [string]::Join(", ", [string[]]$actual)
    $expectedText = [string]::Join(", ", [string[]]$expected)
    if ($actual.Count -ne $expected.Count -or $actualText -cne $expectedText) {
        throw "$Context keys must be exactly [$expectedText]; found [$actualText]."
    }
}

function Resolve-RecordedPath {
    param(
        [AllowNull()]
        [object]$Value,
        [Parameter(Mandatory)]
        [string]$Label
    )

    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace($Value)) {
        throw "$Label must be a non-empty path string."
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repo $Value))
}

function Assert-ExpectedValue {
    param(
        [AllowNull()]
        [object]$Actual,
        [AllowNull()]
        [object]$Expected,
        [Parameter(Mandatory)]
        [string]$Label
    )

    if ($null -eq $Expected) {
        if ($null -ne $Actual) {
            throw "$Label must be null; found '$Actual'."
        }
        return
    }
    if ($null -eq $Actual) {
        throw "$Label must be '$Expected'; found null."
    }
    if ($Expected -is [bool]) {
        if ($Actual -isnot [bool] -or $Actual -ne $Expected) {
            throw "$Label must be '$Expected'; found '$Actual'."
        }
        return
    }
    if ($Expected -is [string]) {
        if ($Actual -isnot [string] -or $Actual -cne $Expected) {
            throw "$Label must be '$Expected'; found '$Actual'."
        }
        return
    }
    try {
        $actualNumber = [double]$Actual
        $expectedNumber = [double]$Expected
    }
    catch {
        throw "$Label must be numeric '$Expected'; found '$Actual'."
    }
    if ($actualNumber -ne $expectedNumber) {
        throw "$Label must be '$Expected'; found '$Actual'."
    }
}

function ConvertTo-FiniteDouble {
    param(
        [AllowNull()]
        [object]$Value,
        [Parameter(Mandatory)]
        [string]$Label
    )

    if ($null -eq $Value) {
        throw "$Label is missing."
    }
    try {
        $number = [double]$Value
    }
    catch {
        throw "$Label is not numeric: '$Value'."
    }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "$Label must be finite; found '$Value'."
    }
    return $number
}

function Assert-Sha256Value {
    param(
        [AllowNull()]
        [object]$Value,
        [Parameter(Mandatory)]
        [string]$Label
    )

    if ($Value -isnot [string] -or $Value -cnotmatch "^[0-9a-f]{64}$") {
        throw "$Label must be a lowercase 64-character SHA-256 digest."
    }
}

function Assert-Receipt {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$ReceiptPath,
        [Parameter(Mandatory)]
        [string]$ExpectedCommit,
        [Parameter(Mandatory)]
        [psobject]$ExpectedCheckpoint,
        [Parameter(Mandatory)]
        [System.Collections.IDictionary]$ExpectedSource,
        [Parameter(Mandatory)]
        [System.Collections.IDictionary]$ExpectedEvaluation
    )

    if (-not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) {
        throw "$Name did not produce the requested receipt: $ReceiptPath"
    }
    try {
        $data = Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json
    }
    catch {
        throw "$Name produced invalid receipt JSON at '$ReceiptPath': $($_.Exception.Message)"
    }

    Assert-ExpectedValue -Actual (Get-RequiredProperty $data "schema_version" "$Name receipt") -Expected 2 -Label "$Name schema_version"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $data "metric_id" "$Name receipt") -Expected "pico-diffusion-fid-style-v1" -Label "$Name metric_id"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $data "status" "$Name receipt") -Expected "current-harness-run" -Label "$Name status"

    $repository = Get-RequiredProperty $data "repository" "$Name receipt"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $repository "commit" "$Name repository") -Expected $ExpectedCommit -Label "$Name repository.commit"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $repository "dirty" "$Name repository") -Expected $false -Label "$Name repository.dirty"

    $harness = Get-RequiredProperty $data "harness" "$Name receipt"
    $recordedHarnessPath = Resolve-RecordedPath -Value (Get-RequiredProperty $harness "path" "$Name harness") -Label "$Name harness.path"
    if (-not [string]::Equals(
        $recordedHarnessPath,
        $ExpectedSource["fid.py"].Path,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Name harness.path does not match the evaluated local fid.py."
    }
    Assert-ExpectedValue -Actual (Get-RequiredProperty $harness "sha256" "$Name harness") -Expected $ExpectedSource["fid.py"].Sha256 -Label "$Name harness.sha256"

    $recordedSources = Get-RequiredProperty $harness "source_files" "$Name harness"
    Assert-ExactPropertyNames -Object $recordedSources -ExpectedNames @($ExpectedSource.Keys) -Context "$Name source_files"
    foreach ($sourceName in $ExpectedSource.Keys) {
        $record = Get-RequiredProperty $recordedSources $sourceName "$Name source_files"
        Assert-ExpectedValue -Actual (Get-RequiredProperty $record "bytes" "$Name source_files.$sourceName") -Expected $ExpectedSource[$sourceName].Bytes -Label "$Name source_files.$sourceName.bytes"
        Assert-ExpectedValue -Actual (Get-RequiredProperty $record "sha256" "$Name source_files.$sourceName") -Expected $ExpectedSource[$sourceName].Sha256 -Label "$Name source_files.$sourceName.sha256"
    }
    Assert-LocalSourceFingerprint -Expected $ExpectedSource

    $checkpoint = Get-RequiredProperty $data "checkpoint" "$Name receipt"
    $recordedCheckpointPath = Resolve-RecordedPath -Value (Get-RequiredProperty $checkpoint "path" "$Name checkpoint") -Label "$Name checkpoint.path"
    if (-not [string]::Equals(
        $recordedCheckpointPath,
        $ExpectedCheckpoint.Path,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Name checkpoint.path points to '$recordedCheckpointPath', expected '$($ExpectedCheckpoint.Path)'."
    }
    Assert-ExpectedValue -Actual (Get-RequiredProperty $checkpoint "bytes" "$Name checkpoint") -Expected $ExpectedCheckpoint.Bytes -Label "$Name checkpoint.bytes"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $checkpoint "sha256" "$Name checkpoint") -Expected $ExpectedCheckpoint.Sha256 -Label "$Name checkpoint.sha256"
    Assert-LocalFileFingerprint -Expected $ExpectedCheckpoint -Label "$Name checkpoint"

    $evaluation = Get-RequiredProperty $data "evaluation" "$Name receipt"
    foreach ($entry in $ExpectedEvaluation.GetEnumerator()) {
        $actual = Get-RequiredProperty $evaluation ([string]$entry.Key) "$Name evaluation"
        Assert-ExpectedValue -Actual $actual -Expected $entry.Value -Label "$Name evaluation.$($entry.Key)"
    }

    $realDataset = Get-RequiredProperty $evaluation "real_dataset" "$Name evaluation"
    $datasetKeys = @(
        "canonical_backend",
        "source_backend",
        "split",
        "dataset_samples",
        "dataset_sha256",
        "subset_seed",
        "subset_samples",
        "subset_sha256",
        "selected_sample_sha256",
        "selection"
    )
    Assert-ExactPropertyNames -Object $realDataset -ExpectedNames $datasetKeys -Context "$Name evaluation.real_dataset"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "canonical_backend" "$Name real_dataset") -Expected "pico-cifar10-rgb-content-v1" -Label "$Name real_dataset.canonical_backend"
    $sourceBackend = Get-RequiredProperty $realDataset "source_backend" "$Name real_dataset"
    if ($sourceBackend -cnotin @(
        "torchvision.datasets.ImageFolder",
        "torchvision.datasets.CIFAR10"
    )) {
        throw "$Name real_dataset.source_backend is unsupported: '$sourceBackend'."
    }
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "split" "$Name real_dataset") -Expected "test" -Label "$Name real_dataset.split"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "dataset_samples" "$Name real_dataset") -Expected 10000 -Label "$Name real_dataset.dataset_samples"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "subset_seed" "$Name real_dataset") -Expected 0 -Label "$Name real_dataset.subset_seed"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "subset_samples" "$Name real_dataset") -Expected $SampleCount -Label "$Name real_dataset.subset_samples"
    Assert-ExpectedValue -Actual (Get-RequiredProperty $realDataset "selection" "$Name real_dataset") -Expected "numpy-pcg64-choice-over-content-hash-order-v1" -Label "$Name real_dataset.selection"
    Assert-Sha256Value -Value (Get-RequiredProperty $realDataset "dataset_sha256" "$Name real_dataset") -Label "$Name real_dataset.dataset_sha256"
    Assert-Sha256Value -Value (Get-RequiredProperty $realDataset "subset_sha256" "$Name real_dataset") -Label "$Name real_dataset.subset_sha256"
    $selectedHashes = @(Get-RequiredProperty $realDataset "selected_sample_sha256" "$Name real_dataset")
    if ($selectedHashes.Count -ne $SampleCount) {
        throw "$Name real_dataset.selected_sample_sha256 must contain $SampleCount hashes; found $($selectedHashes.Count)."
    }
    for ($index = 0; $index -lt $selectedHashes.Count; $index++) {
        Assert-Sha256Value -Value $selectedHashes[$index] -Label "$Name real_dataset.selected_sample_sha256[$index]"
    }
    $datasetIdentityJson = $realDataset | ConvertTo-Json -Compress -Depth 8
    if ($null -eq $script:ExpectedRealDatasetIdentityJson) {
        $script:ExpectedRealDatasetIdentityJson = $datasetIdentityJson
    }
    elseif ($datasetIdentityJson -cne $script:ExpectedRealDatasetIdentityJson) {
        throw "$Name real_dataset identity differs from the first sequential evaluation."
    }

    $artifacts = Get-RequiredProperty $data "artifacts" "$Name receipt"
    $artifactNames = @(
        "generated_images",
        "real_images",
        "generated_activations",
        "real_activations"
    )
    Assert-ExactPropertyNames -Object $artifacts -ExpectedNames $artifactNames -Context "$Name artifacts"
    foreach ($artifactName in $artifactNames) {
        $artifact = Get-RequiredProperty $artifacts $artifactName "$Name artifacts"
        Assert-ExpectedValue -Actual (Get-RequiredProperty $artifact "cache_hit" "$Name artifacts.$artifactName") -Expected $false -Label "$Name artifacts.$artifactName.cache_hit"

        $artifactPath = Resolve-RecordedPath -Value (Get-RequiredProperty $artifact "path" "$Name artifacts.$artifactName") -Label "$Name artifacts.$artifactName.path"
        $artifactFingerprint = Get-FileFingerprint -Path $artifactPath
        Assert-ExpectedValue -Actual (Get-RequiredProperty $artifact "bytes" "$Name artifacts.$artifactName") -Expected $artifactFingerprint.Bytes -Label "$Name artifacts.$artifactName.bytes"
        Assert-ExpectedValue -Actual (Get-RequiredProperty $artifact "sha256" "$Name artifacts.$artifactName") -Expected $artifactFingerprint.Sha256 -Label "$Name artifacts.$artifactName.sha256"
    }

    $results = Get-RequiredProperty $data "results" "$Name receipt"
    $fidScore = ConvertTo-FiniteDouble -Value (Get-RequiredProperty $results "internal_fid_style_score" "$Name results") -Label "$Name internal FID-style score"
    if ($fidScore -lt 0.0) {
        throw "$Name internal FID-style score must be non-negative; found $fidScore."
    }
    $sanityScore = ConvertTo-FiniteDouble -Value (Get-RequiredProperty $results "real_vs_real_sanity_score" "$Name results") -Label "$Name real-vs-real sanity score"
    if ([math]::Abs($sanityScore) -gt 1e-6) {
        throw "$Name real-vs-real sanity score is not effectively zero: $sanityScore."
    }
}

$usingDefaultPython = [string]::IsNullOrWhiteSpace($Python)
if ($usingDefaultPython) {
    $Python = Join-Path $repo ".venv\Scripts\python.exe"
}
$Python = Resolve-PythonInterpreter -Candidate $Python -IsDefault $usingDefaultPython

if ($AllowDirty) {
    throw "-AllowDirty is intentionally rejected: a dirty run cannot produce current/publishable FID evidence. Commit or stash changes, then rerun without -AllowDirty."
}

$required = @(
    "fid.py",
    "diffusion.py",
    "model.py",
    "data_utils.py",
    "out_cifar\ckpt.pt",
    "out_cifar_cond\ckpt.pt"
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $repo $path) -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

$initialRepository = Get-RepositorySnapshot
if ($initialRepository.Dirty) {
    throw "Publishable FID receipts require a clean worktree. Commit or stash changes before running. git status: $($initialRepository.Status)"
}
$commit = $initialRepository.Commit
$sourceFingerprint = Get-SourceFingerprint
$script:ExpectedRealDatasetIdentityJson = $null
$unconditionalCheckpoint = Get-FileFingerprint -Path (Join-Path $repo "out_cifar\ckpt.pt")
$conditionalCheckpoint = Get-FileFingerprint -Path (Join-Path $repo "out_cifar_cond\ckpt.pt")

function Assert-EvaluationInputsStable {
    Assert-LocalSourceFingerprint -Expected $sourceFingerprint
    Assert-LocalFileFingerprint -Expected $unconditionalCheckpoint -Label "unconditional checkpoint"
    Assert-LocalFileFingerprint -Expected $conditionalCheckpoint -Label "conditional checkpoint"
}

Write-Output "Commit: $commit"
Write-Output "Python: $Python"
Write-Output "Sample count: $SampleCount"
Write-Output "Unconditional checkpoint SHA-256: $($unconditionalCheckpoint.Sha256)"
Write-Output "Conditional checkpoint SHA-256: $($conditionalCheckpoint.Sha256)"

$environmentProbe = @'
import numpy, platform, torch, torchvision
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
print(
    'Python {} | NumPy {} | PyTorch {} | torchvision {} | CUDA {} | GPU {}'.format(
        platform.python_version(), numpy.__version__, torch.__version__,
        torchvision.__version__, torch.version.cuda, gpu
    )
)
'@
& $Python -c $environmentProbe
$environmentExitCode = $LASTEXITCODE
Assert-RepositoryState -ExpectedCommit $commit -Stage "Environment probe"
Assert-EvaluationInputsStable
if ($environmentExitCode -ne 0) {
    throw "Environment probe failed with exit code $environmentExitCode. Verify the -Python environment and installed requirements."
}

& $Python -m pytest -q
$testExitCode = $LASTEXITCODE
Assert-RepositoryState -ExpectedCommit $commit -Stage "Test suite"
Assert-EvaluationInputsStable
if ($testExitCode -ne 0) {
    throw "Tests failed with exit code $testExitCode."
}

$runStamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss'Z'")
$runDir = Join-Path $repo (Join-Path "evaluation_runs" $runStamp)
New-Item -ItemType Directory -Path $runDir | Out-Null

function Invoke-FidRun {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string[]]$Arguments,
        [Parameter(Mandatory)]
        [psobject]$ExpectedCheckpoint,
        [Parameter(Mandatory)]
        [System.Collections.IDictionary]$ExpectedEvaluation
    )

    Assert-RepositoryState -ExpectedCommit $commit -Stage "Before $Name"
    Assert-EvaluationInputsStable

    $receipt = Join-Path $runDir "$Name.json"
    $log = Join-Path $runDir "$Name.log"
    $fullArguments = @(
        "fid.py"
    ) + $Arguments + @(
        "--n", $SampleCount,
        "--channels", 3,
        "--image-size", 32,
        "--steps", 50,
        "--sampler", "ddim",
        "--order", 2,
        "--batch-size", 64,
        "--sample-batch-size", 64,
        "--seed", 0,
        "--no-cache",
        "--receipt", $receipt
    )

    Write-Output "Running $Name ..."
    & $Python @fullArguments 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    Assert-RepositoryState -ExpectedCommit $commit -Stage "$Name evaluation"
    Assert-EvaluationInputsStable
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. Partial output remains in '$runDir' for diagnosis only."
    }

    Assert-Receipt -Name $Name -ReceiptPath $receipt -ExpectedCommit $commit -ExpectedCheckpoint $ExpectedCheckpoint -ExpectedSource $sourceFingerprint -ExpectedEvaluation $ExpectedEvaluation
    Assert-RepositoryState -ExpectedCommit $commit -Stage "$Name receipt validation"
    Assert-EvaluationInputsStable

    Get-FileHash -Algorithm SHA256 -LiteralPath $receipt
    Get-FileHash -Algorithm SHA256 -LiteralPath $log
}

$unconditionalEvaluation = [ordered]@{
    n = $SampleCount
    seed = 0
    channels = 3
    image_size = 32
    steps = 50
    sampler = "ddim"
    order = 2
    sample_batch_size = 64
    feature_batch_size = 64
    num_classes = $null
    cfg_scale = 0.0
    real_subset_seed = 0
    no_cache_requested = $true
}
$conditionalEvaluation = [ordered]@{
    n = $SampleCount
    seed = 0
    channels = 3
    image_size = 32
    steps = 50
    sampler = "ddim"
    order = 2
    sample_batch_size = 64
    feature_batch_size = 64
    num_classes = 10
    cfg_scale = 2.0
    real_subset_seed = 0
    no_cache_requested = $true
}

Invoke-FidRun -Name "fid-unconditional-ddim50-n$SampleCount" -Arguments @(
    "--ckpt", "out_cifar\ckpt.pt"
) -ExpectedCheckpoint $unconditionalCheckpoint -ExpectedEvaluation $unconditionalEvaluation

Invoke-FidRun -Name "fid-conditional-cfg2-ddim50-n$SampleCount" -Arguments @(
    "--ckpt", "out_cifar_cond\ckpt.pt",
    "--num-classes", 10,
    "--cfg-scale", 2.0
) -ExpectedCheckpoint $conditionalCheckpoint -ExpectedEvaluation $conditionalEvaluation

Assert-RepositoryState -ExpectedCommit $commit -Stage "Completed evaluations"
Assert-EvaluationInputsStable
Write-Output "Completed sequential publishable evaluations. Receipts and logs: $runDir"
