<#
    Puts a "SMIF Dashboard" shortcut on the Desktop (and optionally the Start
    menu) pointing at the silent launcher. Run once:

        powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1

    Remove it by deleting the shortcut; nothing else is installed.
#>
param([switch]$StartMenu)

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "SMIF Dashboard.vbs"
if (-not (Test-Path $target)) { throw "Launcher not found at $target" }

$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath("Desktop"))
if ($StartMenu) {
    $places += Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
}

foreach ($dir in $places) {
    $link = $shell.CreateShortcut((Join-Path $dir "SMIF Dashboard.lnk"))
    $link.TargetPath = "wscript.exe"
    $link.Arguments = '"' + $target + '"'
    $link.WorkingDirectory = $root
    $link.Description = "KSU SMIF portfolio analytics"
    # A Windows built-in chart icon, so the shortcut is not a blank page.
    $link.IconLocation = "$env:SystemRoot\System32\imageres.dll,174"
    $link.Save()
    Write-Host "Shortcut created in $dir"
}
