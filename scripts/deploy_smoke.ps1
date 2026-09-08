param(
    [string]$Server = $env:IDS_SERVER_HOST,
    [string]$RemoteRoot = "/opt/ids_revision/rm_dmvt_smoke"
)

$ErrorActionPreference = "Stop"

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

$Plink = "C:\Program Files\PuTTY\plink.exe"
$Pscp = "C:\Program Files\PuTTY\pscp.exe"
$Project = Split-Path -Parent $PSScriptRoot

& $Plink -batch -ssh -P 22 -l root -pw $Password -hostkey $HostKey $Server "mkdir -p $RemoteRoot"
& $Pscp -batch -r -P 22 -l root -pw $Password -hostkey $HostKey `
    "$Project\src" "$Project\scripts\server_smoke.py" "root@${Server}:$RemoteRoot/"
& $Plink -batch -ssh -P 22 -l root -pw $Password -hostkey $HostKey $Server `
    "cd $RemoteRoot && timeout 180 /usr/bin/python3 server_smoke.py"
