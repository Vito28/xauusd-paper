$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
$credentialInput = "protocol=https`nhost=github.com`n`n"
$credentialLines = @($credentialInput | git -c credential.interactive=never credential fill 2>$null)
if ($LASTEXITCODE -ne 0) { Write-Output '{"status":"no_stored_github_credential"}'; exit 0 }
$credentialData = @{}
foreach ($line in $credentialLines) { if ($line -match '^([^=]+)=(.*)$') { $credentialData[$matches[1]] = $matches[2] } }
$credentialLines = $null
if (-not $credentialData.ContainsKey('password')) { Write-Output '{"status":"no_stored_github_credential"}'; exit 0 }
$apiHeaders = @{ Authorization = "Bearer $($credentialData['password'])"; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28'; 'User-Agent' = 'Codex-Repository-Task' }
$credentialData.Clear()
try {
    $repoApi = 'https://api.github.com/repos/Vito28/xauusd-paper'
    $repoResult = Invoke-RestMethod -Uri $repoApi -Headers $apiHeaders -Method Get
    $defaultStatus = $repoResult.default_branch
    if ($repoResult.default_branch -ne 'main' -and $repoResult.permissions.admin) {
        $repoUpdate = Invoke-RestMethod -Uri $repoApi -Headers $apiHeaders -Method Patch -ContentType 'application/json' -Body '{"default_branch":"main"}'
        $defaultStatus = $repoUpdate.default_branch
    }
    $headFilter = [uri]::EscapeDataString('Vito28:codex/market-data-dashboard')
    $existingPrResponse = Invoke-RestMethod -Uri ($repoApi + '/pulls?state=open&base=main&head=' + $headFilter) -Headers $apiHeaders -Method Get
    $existingPrs = @($existingPrResponse | Where-Object { $_.number -and $_.base.ref -eq 'main' -and $_.head.ref -eq 'codex/market-data-dashboard' })
    if ($existingPrs.Count -gt 0) {
        $prResult = $existingPrs[0]
    } else {
        $prBody = Get-Content -LiteralPath 'reports/pr-preparation/PR.md' -Raw
        $prPayload = @{ title = 'feat: tambah collector MT5, PostgreSQL, analytics, dan dashboard XAUUSD'; body = $prBody; base = 'main'; head = 'codex/market-data-dashboard'; draft = $true } | ConvertTo-Json -Depth 5
        $prResult = Invoke-RestMethod -Uri ($repoApi + '/pulls') -Headers $apiHeaders -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($prPayload))
    }
    if (-not $prResult.html_url) { throw 'Pull request result did not include a URL.' }
    @{ status = 'success'; default_branch = $defaultStatus; pull_request_url = $prResult.html_url; number = $prResult.number; base = $prResult.base.ref; head = $prResult.head.ref; draft = $prResult.draft } | ConvertTo-Json -Compress
} catch {
    $safeStatus = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
    @{ status = 'github_api_failed'; http_status = $safeStatus } | ConvertTo-Json -Compress
} finally {
    $apiHeaders.Clear()
    $credentialInput = $null
}
