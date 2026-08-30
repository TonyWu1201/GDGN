# =============================================================================
# GDGN Ver2 本地训练前置脚本（Windows PowerShell）
# 模型正式训练前的部分（数据处理 + 折专属预训练）在本地完成：
#   1. 环境与 GPU 检查
#   2. 原始数据前置检查
#   3. 数据重建（strict readiness -> True）
#   4. 训练前验收
#   5. 折专属图预训练（多 GPU 并行，检查点幂等）
#   6. 单折单种子筛选（多 GPU 并行）
#   7. 统计汇总（可选，bootstrap）
# 之后将 data/ 打包传输到远程 GPU 主机，用 run_remote_unattended.sh
# 只跑剩余阶段（确认实验/消融/可解释性）。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/run_local_data_and_pretrain.ps1 [参数]
#
# 参数（均为开关，-Foo 启用）:
#   -SkipData          跳过数据重建（仅校验 data-card strict readiness）
#   -SkipAcceptance    跳过训练前验收（pytest/compileall/smoke）
#   -SkipPretrain      跳过折专属预训练
#   -SkipScreening     跳过单折单种子筛选
#   -SkipAggregate     跳过统计汇总
#   -Folds <n>         本地预训练折数（默认 1，即 fold-00）
#   -Seeds <s1,s2,...> 本地预训练种子（默认 42，即 seed-0042）
#   -Protocols <列表>  本地预训练协议（默认全部 6 个）
#   -Parallel <n>      并行进程数（默认 0 = 自动 = GPU 数）
#   -Epochs <n>         预训练最大 epoch 数（默认 0 = 使用 pretrain_config.json 的 50）
#   -Force             忽略已有检查点强制重跑预训练
#   -VerifyFold        预训练后对每个检查点跑 test-edge 验证（较慢）
#   -BootstrapRuns <n> 层级 bootstrap 重采样次数（默认 2000，-SkipAggregate 时忽略）
#
# 示例:
#   # 数据 + 6 协议 fold-00 seed-42 预训练 + 筛选（推荐，本地约 2-4 小时）
#   powershell -ExecutionPolicy Bypass -File scripts/run_local_data_and_pretrain.ps1
#   # 只重建数据
#   powershell -ExecutionPolicy Bypass -File scripts/run_local_data_and_pretrain.ps1 -SkipPretrain -SkipScreening -SkipAggregate
#   # 全部 5 折 x 3 种子本地预训练（耗时长，谨慎）
#   powershell -ExecutionPolicy Bypass -File scripts/run_local_data_and_pretrain.ps1 -Folds 5 -Seeds 42,3407,8128
# =============================================================================
param(
    [switch]$SkipData,
    [switch]$SkipAcceptance,
    [switch]$SkipPretrain,
    [switch]$SkipScreening,
    [switch]$SkipAggregate,
    [int]$Folds = 1,
    [string]$Seeds = "42",
    [string]$Protocols = "eval-lpo,eval-lco,eval-ldo-kt,eval-ldo-so,eval-lto,eval-db",
    [int]$Parallel = 0,
    [int]$Epochs = 0,
    [switch]$Force,
    [switch]$VerifyFold,
    [int]$BootstrapRuns = 2000
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$LOG_DIR = "data/model/local-logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG_FILE = Join-Path $LOG_DIR ("local-$(Get-Date -Format 'yyyyMMdd-HHmmss').log")
Start-Transcript -Path $LOG_FILE -Append | Out-Null

function Log($msg) { Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" }
function Die($msg) { Write-Error "[FATAL] $msg"; exit 1 }

function Get-GpuCount {
    $n = (nvidia-smi -L 2>$null | Measure-Object -Line).Lines
    if (-not $n) { $n = 0 }
    return [int]$n
}

function Get-AvailDiskGB {
    $p = Get-PSDrive -Name (Split-Path (Get-Location).Path -Qualifier).TrimEnd(':')
    return [int][math]::Floor(($p.Free / 1GB))
}

function Invoke-Uv {
    # 统一入口：全部使用 uv，不直接调 python/pip
    & uv run @args
    if ($LASTEXITCODE -ne 0) { throw "uv run $($args -join ' ') 失败 (exit=$LASTEXITCODE)" }
}

function Invoke-BashScript {
    # Windows 下 bash 可能是 WSL（PATH 里找不到 uv），优先用 Git Bash
    param($script)
    $gitBash = "C:\Program Files\Git\bin\bash.exe"
    if (Test-Path $gitBash) {
        & $gitBash $script
    } else {
        & bash $script
    }
    if ($LASTEXITCODE -ne 0) { throw "$script 失败 (exit=$LASTEXITCODE)" }
}

# ---------------- 阶段 0: 环境检查 ----------------
Log "GDGN Ver2 本地训练前置开始（日志: $LOG_FILE）"
Log "分支: $(git branch --show-current)"
$core_commit = "b735155"
$split_commit = "e205200"
git merge-base --is-ancestor $core_commit HEAD
if ($LASTEXITCODE -ne 0) { Die "缺少核心实现提交 $core_commit" }
git merge-base --is-ancestor $split_commit HEAD
if ($LASTEXITCODE -ne 0) { Die "缺少 split 提交 $split_commit" }

$gpus = Get-GpuCount
if ($gpus -lt 1) { Die "未检测到可用 GPU（nvidia-smi 报告 0 块）" }
if ($Parallel -le 0) { $Parallel = [math]::Min($gpus, 4) }
Log "GPU 数: $gpus，预训练并行度: $Parallel"

$avail = Get-AvailDiskGB
Log "可用磁盘: $avail GB"
if ($avail -lt 30) { Die "可用磁盘不足 30 GB（当前 $avail GB），预训练检查点与 parquet 需要空间" }

# ---------------- 阶段 1: 原始数据前置检查 ----------------
Log "==================== 阶段 1: 原始数据前置检查 ===================="
$raw_files = @(
    "data/raw/cell_line_omics/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv",
    "data/raw/cell_line_omics/DepMap_SomaticMutations.csv",
    "data/raw/cell_line_omics/DepMap_CNGeneWGS.csv",
    "data/raw/cell_line_omics/CCLE_DNA_methylation_TSS1kb.txt",
    "data/raw/cell_line_omics/Cell_lines_annotations_20181226.txt",
    "data/raw/drug_structures/compound_cid_smiles.csv",
    "data/processed/protein_protein_interaction/ppi_dg_filtered.csv",
    "data/processed/drug_gene_interaction/interactions_filtered.csv",
    "data/processed/drug_sensitivity/ic50_matrix.csv"
)
$missing = @()
foreach ($f in $raw_files) { if (-not (Test-Path $f)) { $missing += $f } }
if ($missing.Count -gt 0) {
    Die "缺少原始数据: $($missing -join ', ')，请先按文档第 4 节放置文件"
}
Log "原始数据前置检查通过"

# ---------------- 阶段 2: 数据重建 ----------------
function Check-DataCard {
    $card = "data/model/data-card.md"
    if (-not (Test-Path $card)) { Die "data-card.md 不存在" }
    $cell = (Select-String -Path $card -Pattern "Strict unscaled cell artifact ready:\s*(\w+)" | Select-Object -First 1).Matches[0].Groups[1].Value
    $drug = (Select-String -Path $card -Pattern "Strict unscaled drug artifact ready:\s*(\w+)" | Select-Object -First 1).Matches[0].Groups[1].Value
    if ($cell -ne "True" -or $drug -ne "True") {
        Die "strict readiness 未就绪（cell=$cell drug=$drug），正式 runner 会拒绝启动"
    }
    Log "data-card strict readiness: cell=$cell drug=$drug"
}

Log "==================== 阶段 2: 数据重建 ===================="
if ($SkipData) {
    Log "SkipData: 跳过数据重建，仅校验 data-card"
    Check-DataCard
} else {
    Log "重建 Ver2 严格数据（strict unscaled artifacts）..."
    # 数据脚本为 bash 编写，通过 Git Bash 执行；其余全部走 uv（Windows 原生）
    Invoke-BashScript "scripts/prepare_ver2_data.sh"
    Check-DataCard
}

# ---------------- 阶段 3: 训练前验收 ----------------
Log "==================== 阶段 3: 训练前验收 ===================="
if ($SkipAcceptance) {
    Log "SkipAcceptance: 跳过训练前验收"
} else {
    Invoke-Uv pytest tests/strict_eval -q
    Invoke-Uv python -m compileall -q program tests
    Invoke-Uv python program/smoke_strict_pipeline.py
    Invoke-Uv python program/model/edge_mask.py
    foreach ($p in $Protocols.Split(",")) {
        foreach ($f in 0..($Folds - 1)) {
            $j = "data/model/splits/$p/fold-$( '{0:d2}' -f $f ).json"
            if (-not (Test-Path $j)) { Die "缺少 split 文件: $j" }
            if (-not (Select-String -Path $j -Pattern '"passed": true' -Quiet)) { Die "split audit 未通过: $j" }
        }
    }
    if (-not (Test-Path "data/model/pretrain/edge_split_strict.pt")) { Die "缺少 data/model/pretrain/edge_split_strict.pt" }
    Log "训练前验收通过"
}

# ---------------- 阶段 4: 折专属图预训练（多 GPU 并行） ----------------
function Invoke-FoldPretrainJob {
    param($protocol, $fold, $seed, $gpu)
    $target = "data/model/pretrain/strict/$protocol/fold-$( '{0:d2}' -f $fold )/seed-$( '{0:d4}' -f $seed )"
    $ckpt = "$target/best_encoder.pt"
    if ((Test-Path $ckpt) -and -not $Force) {
        Log "SKIP 已存在: $ckpt"
        return
    }
    $env:CUDA_VISIBLE_DEVICES = "$gpu"
    $args = @("run", "python", "program/model/pretrain.py",
        "--config", "data/model/pretrain/pretrain_config.json",
        "--split-id", $protocol, "--fold", $fold, "--seed", $seed,
        "--output-dir", $target)
    if ($Force) { $args += "--force" }
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "预训练失败: $protocol fold-$fold seed-$seed" }
    if ($VerifyFold) {
        & uv run python program/verify_pretrain.py --ckpt $ckpt --split-id $protocol --fold $fold
        if ($LASTEXITCODE -ne 0) { throw "verify_pretrain 失败: $protocol fold-$fold" }
    }
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}

Log "==================== 阶段 4: 折专属图预训练 ===================="
if ($SkipPretrain) {
    Log "SkipPretrain: 跳过折专属预训练"
} else {
    $protocolList = $Protocols.Split(",")
    $seedList = $Seeds.Split(",") | ForEach-Object { [int]$_ }
    $foldsList = 0..($Folds - 1)
    $procs = @()
    $index = 0

    function Get-PretrainArgs($pro, $fd, $sd, $gp) {
        $target = "data/model/pretrain/strict/$pro/fold-$( '{0:d2}' -f $fd )/seed-$( '{0:d4}' -f $sd )"
        $extra = @()
        if ($Force) { $extra += "--force" }
        if ($Epochs -gt 0) { $extra += "--epochs"; $extra += $Epochs }
        return @("run", "python", "program/model/pretrain.py",
            "--config", "data/model/pretrain/pretrain_config.json",
            "--split-id", $pro, "--fold", $fd, "--seed", $sd,
            "--output-dir", $target) + $extra
    }

    foreach ($p in $protocolList) {
        foreach ($f in $foldsList) {
            foreach ($s in $seedList) {
                $gpu = $index % $Parallel
                $target = "data/model/pretrain/strict/$p/fold-$( '{0:d2}' -f $f )/seed-$( '{0:d4}' -f $s )"
                $ckpt = "$target/best_encoder.pt"
                if ((Test-Path $ckpt) -and -not $Force) {
                    Log "SKIP 已存在: $ckpt"
                    $index++
                    continue
                }
                $index++
                $uvArgs = Get-PretrainArgs $p $f $s $gpu
                if ($Parallel -eq 1) {
                    # 单卡：前台直跑，tqdm 实时显示
                    Log ("启动预训练 #" + $index + ": $p fold-$f seed-$s -> GPU $gpu（前台）")
                    $env:CUDA_VISIBLE_DEVICES = "$gpu"
                    & uv @uvArgs
                    if ($LASTEXITCODE -ne 0) { Die "预训练失败: $p fold-$f seed-$s" }
                    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
                    if ($VerifyFold) {
                        & uv run python program/verify_pretrain.py --ckpt $ckpt --split-id $p --fold $f
                        if ($LASTEXITCODE -ne 0) { Die "verify_pretrain 失败: $p fold-$f" }
                    }
                } else {
                    # 多卡：Start-Process 继承控制台输出（Start-Job 会缓冲输出导致进度条不可见）
                    Log ("启动预训练 #" + $index + ": $p fold-$f seed-$s -> GPU $gpu（后台，输出实时显示）")
                    $psi = New-Object System.Diagnostics.ProcessStartInfo
                    $psi.FileName = (Get-Command uv).Source
                    $psi.ArgumentList.AddRange([string[]]$uvArgs)
                    $psi.UseShellExecute = $false
                    $psi.RedirectStandardOutput = $false
                    $psi.RedirectStandardError = $false
                    $psi.CreateNoWindow = $true
                    $psi.Environment["CUDA_VISIBLE_DEVICES"] = "$gpu"
                    $procs += [System.Diagnostics.Process]::Start($psi)
                    if ($procs.Count -ge $Parallel) {
                        $procs | Wait-Process
                        $failed = $procs | Where-Object { $_.ExitCode -ne 0 }
                        if ($failed) { Die "预训练任务失败: $($failed | ForEach-Object { $_.StartInfo.ArgumentList -join ' ' })" }
                        $procs = @()
                    }
                }
            }
        }
    }
    $procs | Wait-Process
    $failed = $procs | Where-Object { $_.ExitCode -ne 0 }
    if ($failed) { Die "预训练任务失败: $($failed | ForEach-Object { $_.StartInfo.ArgumentList -join ' ' })" }

    $missing = @()
    foreach ($p in $protocolList) {
        foreach ($f in $foldsList) {
            foreach ($s in $seedList) {
                $ckpt = "data/model/pretrain/strict/$p/fold-$( '{0:d2}' -f $f )/seed-$( '{0:d4}' -f $s )/best_encoder.pt"
                if (-not (Test-Path $ckpt)) { $missing += $ckpt }
            }
        }
    }
    if ($missing.Count -gt 0) {
        Die "折专属预训练检查点不完整，缺少: $($missing -join ', ')"
    }
    $total = $protocolList.Count * $foldsList.Count * $seedList.Count
    Log "折专属预训练检查点完整（$total 个）"
}

# ---------------- 阶段 5: 单折单种子筛选（多 GPU 并行） ----------------
Log "==================== 阶段 5: 单折单种子筛选 ===================="
if ($SkipScreening) {
    Log "SkipScreening: 跳过筛选"
} else {
    foreach ($p in $Protocols.Split(",")) {
        $ckpt = "data/model/pretrain/strict/$p/fold-00/seed-0042/best_encoder.pt"
        if (-not (Test-Path $ckpt)) { Die "缺少筛选所需预训练检查点: $ckpt" }
    }
    $manifest = "data/model/experiments/screening-manifest.jsonl"
    Invoke-Uv python program/generate_run_manifest.py --sweep configs/sweeps/screening.yaml --output $manifest
    $nRuns = (Get-Content $manifest | Measure-Object -Line).Lines
    Log "筛选: $nRuns 个运行，并行 $Parallel 个进程"
    $procs = @()
    for ($i = 0; $i -lt $nRuns; $i++) {
        $gpu = $i % $Parallel
        if ($Parallel -eq 1) {
            Log ("筛选运行 #" + ($i + 1) + "/$nRuns -> GPU $gpu（前台）")
            $env:CUDA_VISIBLE_DEVICES = "$gpu"
            & uv run python program/run_manifest_entry.py --manifest $manifest --index $i
            if ($LASTEXITCODE -ne 0) { Die "筛选任务失败: index=$i" }
            Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        } else {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = (Get-Command uv).Source
            $psi.ArgumentList.AddRange([string[]]@("run", "python", "program/run_manifest_entry.py", "--manifest", $manifest, "--index", $i))
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $false
            $psi.RedirectStandardError = $false
            $psi.CreateNoWindow = $true
            $psi.Environment["CUDA_VISIBLE_DEVICES"] = "$gpu"
            $procs += [System.Diagnostics.Process]::Start($psi)
            if ($procs.Count -ge $Parallel) {
                $procs | Wait-Process
                $failed = $procs | Where-Object { $_.ExitCode -ne 0 }
                if ($failed) { Die "筛选任务失败: index 范围 $($i - $Parallel + 1)..$i" }
                $procs = @()
            }
        }
    }
    $procs | Wait-Process
    $failed = $procs | Where-Object { $_.ExitCode -ne 0 }
    if ($failed) { Die "筛选任务失败" }
    Invoke-Uv python program/aggregate_strict_results.py
    Log "筛选完成，结果在 data/model/experiments/"
}

# ---------------- 阶段 6: 统计汇总 ----------------
Log "==================== 阶段 6: 统计汇总 ===================="
if ($SkipAggregate) {
    Log "SkipAggregate: 跳过统计汇总"
} else {
    Invoke-Uv python program/aggregate_strict_results.py `
        --root data/model/experiments `
        --output-dir data/model/experiments `
        --hierarchical-bootstrap-runs $BootstrapRuns
    Log "统计汇总完成"
}

Log "本地训练前置完成。"
Log "后续步骤: 将 data/ 打包传输到远程 GPU 主机，运行 run_remote_unattended.sh（SKIP_DATA=1 SKIP_PRETRAIN=1）只跑确认实验/消融/可解释性。"
Stop-Transcript | Out-Null
