# =============================================================================
# install-prerequisites.ps1
# Автоматическая установка всех инструментов из Шага 1 README.md
# Запуск: правой кнопкой -> "Запустить с PowerShell" (от имени администратора)
# Или из PowerShell:  powershell -ExecutionPolicy Bypass -File install-prerequisites.ps1
# =============================================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = 'Stop'

# ---- Проверка наличия winget ------------------------------------------------
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget не найден. Установите 'App Installer' из Microsoft Store:" -ForegroundColor Red
    Write-Host "https://apps.microsoft.com/detail/9NBLGGH4NNS1" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== Установка инструментов для проекта ===" -ForegroundColor Cyan
Write-Host ""

# ---- Список пакетов: id для winget + версия из таблицы README ---------------
# Если Version пуст — ставим последнюю стабильную (для пакетов с >= в таблице).
$packages = @(
    @{ Name = 'Docker Desktop'; Id = 'Docker.DockerDesktop'; Version = '4.25.0'; Check = 'docker --version' },         # >= 4.25 (фиксируем минимум из таблицы)
    @{ Name = 'kind 0.20.0';    Id = 'Kubernetes.kind';      Version = '0.20.0'; Check = 'kind --version' },           # точная версия
    @{ Name = 'kubectl 1.28.0'; Id = 'Kubernetes.kubectl';   Version = '1.28.0'; Check = 'kubectl version --client' }, # >= 1.28
    @{ Name = 'Python 3.10';    Id = 'Python.Python.3.10';   Version = '';       Check = 'python --version' },         # ветка 3.10
    @{ Name = 'Node.js LTS';    Id = 'OpenJS.NodeJS.LTS';    Version = '';       Check = 'node --version' }            # >= 18 (LTS)
)

foreach ($pkg in $packages) {
    Write-Host "--- $($pkg.Name) ---" -ForegroundColor Green
    $args = @('install', '--id', $pkg.Id, '--exact',
              '--accept-source-agreements', '--accept-package-agreements', '--silent')
    if ($pkg.Version) { $args += @('--version', $pkg.Version) }
    try {
        winget @args
    } catch {
        Write-Host "Ошибка установки $($pkg.Name): $_" -ForegroundColor Red
    }
    Write-Host ""
}

# ---- Обновление PATH в текущей сессии ---------------------------------------
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User')

# ---- Проверка установленных версий ------------------------------------------
Write-Host "=== Проверка установленных версий ===" -ForegroundColor Cyan
foreach ($pkg in $packages) {
    Write-Host "$($pkg.Name): " -NoNewline -ForegroundColor Yellow
    try {
        Invoke-Expression $pkg.Check
    } catch {
        Write-Host "не найдено в PATH (перезапустите терминал)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Готово ===" -ForegroundColor Cyan
Write-Host "ВАЖНО:" -ForegroundColor Yellow
Write-Host " 1. Перезапустите PowerShell, чтобы PATH обновился."
Write-Host " 2. Запустите Docker Desktop вручную (первый раз) и дождитесь иконки в трее."
Write-Host " 3. В Docker Desktop -> Settings -> Resources поставьте >= 6 ГБ RAM."
Write-Host " 4. Убедитесь, что включена WSL2 (Docker Desktop предложит сам)."
