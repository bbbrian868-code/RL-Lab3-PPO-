# [說明] 訓練監督腳本：執行 main.py --resume，若程序意外結束 (沒有 DONE 標記) 就自動從 last.pt 續訓
# [說明] 用法：powershell -ExecutionPolicy Bypass -File src\run_train.ps1 -ExpName Enduro_release -EnvId ALE/Enduro-v5 -Steps 20000000
param(
	[string]$ExpName = "Enduro_release",
	[string]$EnvId = "ALE/Enduro-v5",
	[string]$Steps = "20000000",
	[int]$EvalIntervalSteps = 500000,
	[int]$CheckpointIntervalSteps = 200000,
	[int]$MaxRestarts = 50,
	[string]$LearningRate = "2.5e-4",
	[string]$InitFrom = ""
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ExpDir = Join-Path $Root "runs\$ExpName"
$Log = Join-Path $ExpDir "train.log"
$Done = Join-Path $ExpDir "checkpoints\DONE"
New-Item -ItemType Directory -Force $ExpDir | Out-Null
$env:PYTHONWARNINGS = "ignore"
# [說明] -InitFrom：從別的實驗的 checkpoint 分支 (只在本實驗還沒有 last.pt 時生效，之後重啟都從自己的 last.pt 續訓)
$InitArg = ""
if ($InitFrom -ne "") { $InitArg = "--init-from `"$InitFrom`"" }

for ($i = 0; $i -le $MaxRestarts; $i++) {
	if (Test-Path $Done) { break }
	Add-Content -Path $Log -Value "### [supervisor] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') launch #$i" -Encoding ascii
	# [說明] 透過 cmd 轉向輸出，避免 PowerShell 5.1 把 stderr 包成 ErrorRecord
	cmd.exe /c "cd /d `"$PSScriptRoot`" && `"$Python`" -u main.py --exp-name $ExpName --env-id $EnvId --training-steps $Steps --eval-interval-steps $EvalIntervalSteps --checkpoint-interval-steps $CheckpointIntervalSteps --learning-rate $LearningRate $InitArg --resume >>`"$Log`" 2>&1"
	Add-Content -Path $Log -Value "### [supervisor] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') python exited with code $LASTEXITCODE" -Encoding ascii
	if (-not (Test-Path $Done)) { Start-Sleep -Seconds 30 }
}
