param(
    [string]$Server = $env:IDS_SERVER_HOST,
    [string]$RemoteRoot = "/opt/ids_revision/rm_dmvt"
)

$ErrorActionPreference = "Stop"
$Plink = "C:\Program Files\PuTTY\plink.exe"
$Pscp = "C:\Program Files\PuTTY\pscp.exe"
$Project = Split-Path -Parent $PSScriptRoot

if (-not $Server) {
    throw "Set IDS_SERVER_HOST before deploying: `$env:IDS_SERVER_HOST = '<host>'"
}

$Password = $env:IDS_SERVER_PW
if (-not $Password) {
    throw "Set IDS_SERVER_PW before deploying: `$env:IDS_SERVER_PW = '<password>'"
}

$HostKey = $env:IDS_SERVER_HOSTKEY
if (-not $HostKey) {
    throw "Set IDS_SERVER_HOSTKEY to the server's SSH host-key fingerprint"
}


& $Plink -batch -ssh -P 22 -l root -pw $Password -hostkey $HostKey $Server `
    "mkdir -p $RemoteRoot/configs $RemoteRoot/docs $RemoteRoot/scripts $RemoteRoot/tests"
& $Pscp -batch -r -P 22 -l root -pw $Password -hostkey $HostKey `
    "$Project\src" "root@${Server}:$RemoteRoot/"
& $Pscp -batch -r -P 22 -l root -pw $Password -hostkey $HostKey `
    "$Project\configs" "$Project\docs" "root@${Server}:$RemoteRoot/"
& $Pscp -batch -r -P 22 -l root -pw $Password -hostkey $HostKey `
    "$Project\tests" "$Project\scripts" "root@${Server}:$RemoteRoot/"
& $Pscp -batch -P 22 -l root -pw $Password -hostkey $HostKey `
    "$Project\pyproject.toml" "$Project\README.md" "$Project\.python-version" `
    "$Project\scripts\bootstrap_server.sh" "root@${Server}:$RemoteRoot/"

Write-Output "Deployed source to $RemoteRoot. Dependencies were not installed."
