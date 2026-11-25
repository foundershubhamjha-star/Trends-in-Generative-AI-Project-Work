# Start the Dubbing Studio web UI. Right-click > Run with PowerShell, or: .\run_ui.ps1
Set-Location $PSScriptRoot
& .\venv\Scripts\Activate.ps1
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not running; the agent's LLM rewrite step will be skipped. Start Ollama from the Start menu for the full agent."
}
python app.py
