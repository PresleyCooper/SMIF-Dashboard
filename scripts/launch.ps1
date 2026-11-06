<#
    launch.ps1 — start the dashboard if it is not already running, then open it.

    Streamlit is a web server, not a file, so a bookmark to localhost only
    works while something is serving it. This starts that something, waits for
    the port to actually accept a connection, and only then opens the browser —
    opening immediately gives "can't reach this site", because Streamlit takes a
    few seconds to boot and the browser does not retry.

    Running it twice is harmless: if the port is already answering it just opens
    a tab against the server that is already up.
#>

param(
    [int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Port([int]$p) {
    # A TCP connect, not Test-NetConnection: that pings first and takes about a
    # second per call, which turns the wait loop below into a minute of nothing.
    $client = New-Object Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $p); $true }
    catch { $false }
    finally { $client.Dispose() }
}

function Get-Python {
    foreach ($candidate in @("python", "py")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            # The Microsoft Store ships a stub "python.exe" that does nothing but
            # open the Store page. It is on PATH by default and would hang here.
            if ($cmd.Source -notlike "*WindowsApps\python*.exe" -or
                (& $candidate -c "print(1)" 2>$null) -eq "1") {
                return $candidate
            }
        }
    }
    throw "Python was not found on PATH. Install it, or edit this script."
}

if (Test-Port $Port) {
    Write-Host "Already running on port $Port."
} else {
    $python = Get-Python
    Write-Host "Starting the dashboard ($python, port $Port)..."
    $args = @("-m", "streamlit", "run", "app.py",
              "--server.port", $Port,
              "--server.headless", "true")
    Start-Process -FilePath $python -ArgumentList $args `
                  -WorkingDirectory $root -WindowStyle Hidden

    $ready = $false
    foreach ($i in 1..90) {                 # up to 45 seconds
        Start-Sleep -Milliseconds 500
        if (Test-Port $Port) { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Warning "The server did not come up within 45 seconds."
        Write-Warning "Run it in the foreground to see why:"
        Write-Warning "  python -m streamlit run app.py"
        exit 1
    }
    Write-Host "Ready."
}

if (-not $NoBrowser) { Start-Process "http://localhost:$Port" }
