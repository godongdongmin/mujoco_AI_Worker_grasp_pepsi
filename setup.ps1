$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$modelCommit = 'd8344c0dbe7a00208d0301111523dde65efc174a'
$modelRepo = Join-Path $PSScriptRoot 'vendor/robotis_mujoco_menagerie'
function Assert-CommandSucceeded {
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    Assert-CommandSucceeded
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
Assert-CommandSucceeded
if (-not (Test-Path -LiteralPath $modelRepo)) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git $modelRepo
    Assert-CommandSucceeded
    git -C $modelRepo fetch --depth 1 origin $modelCommit
    Assert-CommandSucceeded
    git -C $modelRepo checkout --detach $modelCommit
    Assert-CommandSucceeded
    git -C $modelRepo sparse-checkout set robotis_ffw
    Assert-CommandSucceeded
} else {
    $installedCommit = git -C $modelRepo rev-parse HEAD
    Assert-CommandSucceeded
    if ($installedCommit -ne $modelCommit) {
        throw 'Existing model checkout has a different revision. It has not been changed.'
    }
}
& '.venv/Scripts/python.exe' validate_ik.py
Assert-CommandSucceeded
& '.venv/Scripts/python.exe' validate_ik_panel.py
Assert-CommandSucceeded
& '.venv/Scripts/python.exe' validate_joint_control.py
Assert-CommandSucceeded
& '.venv/Scripts/python.exe' validate_table_scene.py
Assert-CommandSucceeded
& '.venv/Scripts/python.exe' validate_thumb_c.py
Assert-CommandSucceeded
Write-Output 'Ready. Run run_ik.cmd (manual) or run_ik_demo.cmd (demo).'
