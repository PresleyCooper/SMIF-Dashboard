' Same as the .bat, but with no console window at all.
' Use this one for a Desktop or Start-menu shortcut.
Set shell = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & here & "\scripts\launch.ps1""", 0, False
