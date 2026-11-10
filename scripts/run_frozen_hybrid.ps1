# Energy prior / energy mask on frozen GaitSet, then GaitBase: embed (resumable) and evaluate.
# Usage (from the project root): powershell -File scripts\run_frozen_hybrid.ps1
$ErrorActionPreference = 'Continue'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONPATH = 'src'
$py = '.\.venv\Scripts\python.exe'
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Tee-Object -FilePath logs\frozen_hybrid.log -Append }

foreach ($bb in 'gaitset', 'gaitbase') {
    Log "embed $bb"
    & $py -W ignore -m gaitlab.frozen_hybrid embed $bb *>> logs\frozen_hybrid_$bb.log
    if ($LASTEXITCODE -ne 0) { Log "embed $bb FAILED"; exit 1 }
    $done = & $py -c "import json,os,sys; p='results/p5_plugin/frozen_hybrid.json'; print(int(os.path.exists(p) and len(json.load(open(p)).get(sys.argv[1], {})) == 2))" $bb
    if ($done -eq '1') { Log "eval $bb already done"; continue }
    Log "eval $bb"
    & $py -W ignore -m gaitlab.frozen_hybrid eval $bb *>> logs\frozen_hybrid_$bb.log
    if ($LASTEXITCODE -ne 0) { Log "eval $bb FAILED"; exit 1 }
}
Log 'all done'
