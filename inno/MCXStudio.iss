;;======================================================================
;; Inno Setup Script for MCX Studio and MCX Suite Windows Installer
;;
;; Author: Qianqian Fang <q.fang at neu.edu>
;; Initially added on Jun. 28, 2020
;; URL: https://mcx.space
;; Github: https://github.com/fangq/mcx
;;
;;======================================================================

#define MyAppName "MCX Studio"
#define MyAppDir "MCXStudio"
; the version can be set on the command line: iscc /DMyAppVersion=nightly MCXStudio.iss
#ifndef MyAppVersion
  #define MyAppVersion "v2026"
#endif
#define MyAppPublisher "COTILab"
#define MyAppURL "https://mcx.space"
#define MyAppExeName "mcxstudio.exe"

;;----------------------------------------------------------------------

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{0A4EBE79-4F72-4379-8C40-2A395E069EC8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={code:GetDefaultDir}\{#MyAppDir}
DisableProgramGroupPage=no
; written for the installer: what is being installed and its license, and
; a getting-started page; the full citation list stays in mcx's NOTICE.txt
LicenseFile=installer_license.txt
InfoAfterFile=installer_readme.txt
; setup asks whether to install for all users (PATH and MATLABPATH/OCTAVE_PATH
; in the machine environment) or for the current user only (in the user's)
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename="{#MyAppDir}-{#MyAppVersion}"
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=true
; a MATLAB or Octave that has loaded a toolbox's mex file holds it open; the
; default filter (*.exe,*.dll,*.chm) misses those, and a silent install then
; aborts on the locked file instead of offering to close the program
CloseApplicationsFilter=*.exe,*.dll,*.chm,*.mex,*.mexw64
ChangesAssociations=yes
SetupIconFile=mcxstudio_2023.ico
ShowLanguageDialog=yes
ArchitecturesInstallIn64BitMode=x64compatible
DefaultGroupName="{#MyAppName} {#MyAppVersion}"
VersionInfoCompany={#MyAppPublisher}
VersionInfoCopyRight="(C) {#MyAppPublisher}"
UninstallDisplayIcon={app}\{#MyAppExeName}

;;----------------------------------------------------------------------

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "portuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
;Name: "hindi"; MessagesFile: "compiler:Languages\Hindi.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "chinesetraditional"; MessagesFile: "compiler:Languages\ChineseTraditional.isl"

;;----------------------------------------------------------------------

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "envPath"; Description: "Add to PATH variable" 
Name: "addMATLABPath"; Description: "Add the toolboxes to the MATLAB and Octave paths (MATLABPATH, OCTAVE_PATH)"

;;----------------------------------------------------------------------

[Dirs]
Name: "{app}\MCXSuite"

;;----------------------------------------------------------------------

[Files]
; MCX Studio 2 (mcxstudio2/, built as mcxstudio.exe): one executable with the
; 3-D viewer built in, so mcxshow/mcxviewer, the locale folder and the
; plink/pscp remote-run helpers of the first Studio are no longer shipped
Source: "..\mcxstudio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer_readme.txt"; DestDir: "{app}"; DestName: "GETTING_STARTED.txt"; Flags: ignoreversion
; the photon launcher and redbirdpy of the portable packages made by mcx2026.sh
Source: "..\photon"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\Python\*"; DestDir: "{app}\Python"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "..\MCXSuite\*"; DestDir: "{app}\MCXSuite"; Excludes: "AUTO_BUILD_*.log,mcxshow.*,mcxviewer.*,mcx\utils\*,mmc\matlab\*"; Flags: ignoreversion recursesubdirs createallsubdirs
; the MATLAB/Octave toolboxes stay under {app}; MATLAB and Octave find them
; through MATLABPATH and OCTAVE_PATH, so no MATLAB install is written to
; (redbird's rbfemmatrix.mex, a Linux Octave build redbird used to track, is
; left out: Octave on Windows takes it over rbfemmatrix.m and fails to load it)
Source: "..\MATLAB\*"; DestDir: "{app}\MATLAB"; Excludes: "AUTO_BUILD_*.log,redbird\matlab\rbfemmatrix.mex"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\MCXSuite\mcx\utils\*"; DestDir: "{app}\MATLAB\mcxtools"; Excludes: "AUTO_BUILD_*.log"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\MCXSuite\mmc\matlab\*"; DestDir: "{app}\MATLAB\mmctools"; Excludes: "AUTO_BUILD_*.log"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

;;----------------------------------------------------------------------

[Icons]
; --user: sessions without a writable folder of their own -- new ones, and
; the examples installed under {app} -- go to %USERPROFILE%\MCXOutput
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--user"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--user"; Tasks: desktopicon
Name: "{group}\MCX Website"; Filename: "https://mcx.space/"
Name: "{group}\MCX Wiki"; Filename: "https://mcx.space/wiki/"
Name: "{group}\MCX Forum"; Filename: "https://groups.google.com/g/mcx-users"

;;----------------------------------------------------------------------

[Registry]
; Create "Software\My Company\My Program" keys under CURRENT_USER or
; LOCAL_MACHINE depending on administrative or non administrative install
; mode. The flags tell it to always delete the "My Program" key upon
; uninstall, and delete the "My Company" key if there is nothing left in it.
Root: HKA; Subkey: "Software\COTILab"; Flags: uninsdeletekeyifempty
Root: HKA; Subkey: "Software\COTILab\MCXStudio"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\COTILab\MCXStudio\Settings"; ValueType: string; ValueName: "Language"; ValueData: "{language}"
Root: HKA; Subkey: "Software\COTILab\MCXStudio\Settings"; ValueType: string; ValueName: "LanguageCode"; ValueData: "{code:GetLanguageCode}"
; First-Studio projects: MCX Studio 2 opens a session of a .mcxp as the mcx
; input it stood for (mcxmcxp.pas), so the association stays
Root: HKA; Subkey: "Software\Classes\.mcxp"; ValueType: string; ValueName: ""; ValueData: "MCXStudio.mcxp"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.mcxp\OpenWithProgids"; ValueType: string; ValueName: "MCXStudio.mcxp"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\MCXStudio.mcxp"; ValueType: string; ValueName: ""; ValueData: "MCXStudio"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\MCXStudio.mcxp\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\mcxstudio.exe,0"
Root: HKA; Subkey: "Software\Classes\MCXStudio.mcxp\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\mcxstudio.exe"" ""--user"" ""%1"""
; Offered under "Open with" for .json, not made the default: .json is not ours
Root: HKA; Subkey: "Software\Classes\.json\OpenWithProgids"; ValueType: string; ValueName: "MCXStudio.json"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\MCXStudio.json"; ValueType: string; ValueName: ""; ValueData: "MCX Studio input"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\MCXStudio.json\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\mcxstudio.exe,0"
Root: HKA; Subkey: "Software\Classes\MCXStudio.json\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\mcxstudio.exe"" ""--user"" ""%1"""
; HKA (and HKCU) should only be used for settings which are compatible with
; roaming profiles so settings like paths should be written to HKLM, which
; is only possible in administrative install mode.
Root: HKLM; Subkey: "Software\COTILab"; Flags: uninsdeletekeyifempty; Check: IsAdminInstallMode
Root: HKLM; Subkey: "Software\COTILab\MCXStudio"; Flags: uninsdeletekey; Check: IsAdminInstallMode
Root: HKLM; Subkey: "Software\COTILab\MCXStudio\Settings"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Check: IsAdminInstallMode
; User specific settings should always be written to HKCU, which should only
; be done in non administrative install mode.
Root: HKCU; Subkey: "Software\COTILab\MCXStudio\Settings"; ValueType: string; ValueName: "UserName"; ValueData: "{userinfoname}"; Check: not IsAdminInstallMode
Root: HKCU; Subkey: "Software\COTILab\MCXStudio\Settings"; ValueType: string; ValueName: "UserOrganization"; ValueData: "{userinfoorg}"; Check: not IsAdminInstallMode

; Fix windows graphics driver watchdog time limit
Root: HKLM; Subkey: "System\CurrentControlSet\Control\GraphicsDrivers"; ValueType: dword; ValueName: "TdrDelay"; ValueData: 10000000; Check: IsAdminInstallMode

;;----------------------------------------------------------------------

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--user"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

;;----------------------------------------------------------------------

[Code]

const
  SystemEnvKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  UserEnvKey = 'Environment';
  // the toolbox folders under {app}\MATLAB put on MATLABPATH and OCTAVE_PATH;
  // each one's PKG_ADD finds its Octave mex files under octave\<arch>
  ToolboxDirs = 'mcxlab;mcxlabcl;mmclab;umcxlab;iso2mesh;zmat;redbird\matlab;mcxtools;mmctools';

// codes to finding MATLAB's installation path is derived from
// https://stackoverflow.com/a/46283761/4271392; only used to take out the
// pathdef.m line of the installers before v2026, which copied the toolboxes
// into <matlabroot>\toolbox\mcx
var
  MatlabToolboxLocalPath: string;

function GetDefaultDir(Param: string): string;
begin
  if IsAdminInstallMode then
    Result := ExpandConstant('{commonappdata}')
  else
    Result := ExpandConstant('{localappdata}') + '\Programs';
end;

// Function to convert installer language to application language code
function GetLanguageCode(Param: string): string;
begin
  case ActiveLanguage of
    'english': Result := 'en-us';
    'french': Result := 'fr';
    'german': Result := 'de';
    'spanish': Result := 'es';
    'italian': Result := 'it';
    'japanese': Result := 'ja';
    'korean': Result := 'ko';
    'portuguese': Result := 'pt';
    'hindi': Result := 'hi';
    'chinesesimplified': Result := 'zh-cn';
    'chinesetraditional': Result := 'zh-tw';
  else
    Result := 'en-us'; // Default fallback
  end;
  Log(Format('Language code selected: %s', [Result]));
end;

function InitializeSetup(): Boolean;
var
  MatlabExePath: string;
begin
  MatlabExePath := FileSearch('matlab.exe', GetEnv('PATH'));
  Log(Format('MATLAB Path found at %s', [MatlabExePath]));
  MatlabToolboxLocalPath := '';
  if MatlabExePath <> '' then
    MatlabToolboxLocalPath := ExtractFilePath(MatlabExePath) + '..\toolbox';
  Log(Format('MATLAB Toolbox Path found at %s', [MatlabToolboxLocalPath]));

  Result := True;
end;

// the environment of an all-users install is the machine's, otherwise the
// current user's
function EnvRoot: Integer;
begin
  if IsAdminInstallMode then
    Result := HKEY_LOCAL_MACHINE
  else
    Result := HKEY_CURRENT_USER;
end;

function EnvKey: string;
begin
  if IsAdminInstallMode then
    Result := SystemEnvKey
  else
    Result := UserEnvKey;
end;

// a ';' list without its empty entries, without the entries in Drop (also a
// ';' list, case-insensitive), and without MCXSuite folders that no longer
// exist -- what an install in another folder, or an older installer that
// left stray ';' behind, leaves in PATH
function CleanList(List, Drop: string): string;
var
  Item: string;
  P: Integer;
begin
  Result := '';
  List := List + ';';
  Drop := ';' + Uppercase(Drop) + ';';
  while List <> '' do begin
    P := Pos(';', List);
    Item := Trim(Copy(List, 1, P - 1));
    Delete(List, 1, P);
    if Item = '' then continue;
    if Pos(';' + Uppercase(Item) + ';', Drop) > 0 then continue;
    if (Pos('\MCXSUITE\', Uppercase(Item)) > 0) and (Pos('%', Item) = 0)
       and not DirExists(Item) then continue;
    if Result <> '' then Result := Result + ';';
    Result := Result + Item;
  end;
end;

// add the ';' list Dirs to the end of the environment variable Name, or with
// Add=False take them out; a variable left empty is deleted, except PATH.
// RegWriteStringValue keeps an existing REG_EXPAND_SZ value expandable.
procedure EnvUpdate(Name, Dirs: string; Add: Boolean);
var
  Value, NewValue: string;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, Name, Value) then
    Value := '';
  NewValue := CleanList(Value, Dirs);
  if Add then begin
    if NewValue <> '' then NewValue := NewValue + ';';
    NewValue := NewValue + Dirs;
  end;
  if NewValue = Value then exit;
  if (NewValue = '') and (CompareText(Name, 'Path') <> 0) then begin
    if RegDeleteValue(EnvRoot, EnvKey, Name) then
      Log(Format('Removed %s', [Name]));
  end else if RegWriteStringValue(EnvRoot, EnvKey, Name, NewValue) then
    Log(Format('%s set to [%s]', [Name, NewValue]))
  else
    Log(Format('Error while writing %s: [%s]', [Name, NewValue]));
end;

function SuiteBinDirs: string;
var
  App: string;
begin
  App := ExpandConstant('{app}');
  Result := App + ';' + App + '\MCXSuite\mcx\bin;' + App + '\MCXSuite\mcxcl\bin;' +
            App + '\MCXSuite\mmc\bin;' + App + '\MCXSuite\vkmmc\bin;' +
            App + '\MCXSuite\umcx\bin';
end;

function ToolboxPaths: string;
var
  List, Item: string;
  P: Integer;
begin
  Result := '';
  List := ToolboxDirs + ';';
  while List <> '' do begin
    P := Pos(';', List);
    Item := Copy(List, 1, P - 1);
    Delete(List, 1, P);
    if Result <> '' then Result := Result + ';';
    Result := Result + ExpandConstant('{app}') + '\MATLAB\' + Item;
  end;
end;

function DeleteLineInTemplate(
  FileName: string; StartLine, EndLine, LinePattern: string): Boolean;
var
  Lines: TArrayOfString;
  Count, I, I2: Integer;
  Line: string;
  State: Integer;
begin
  Result := True;

  if not LoadStringsFromFile(FileName, Lines) then begin
    Log(Format('Error reading %s', [FileName]));
    Result := False;
  end else begin
    State := 0;

    Count := GetArrayLength(Lines);
    for I := 0 to Count - 1 do begin
      Line := Trim(Lines[I]);
      if (CompareText(Line, StartLine) = 0) then begin
        State := 1;
        Log(Format('Start line found at %d', [I]));
      end else if (State = 1) and (CompareText(Line, EndLine) = 0) then begin
        Log(Format('Line is not found in the file %d', [I]));
        State := 2;
        break;
      end else if (State = 1) and (Pos(LinePattern,Line) > 0) then   begin
        Log(Format('End line found at %d, deleting', [I]));
        for I2 := I+1 to Count - 1 do
          Lines[I2 - 1] := Lines[I2];
        SetArrayLength(Lines, Count - 1);
        State := 2;

        if not SaveStringsToFile(FileName, Lines, False) then begin
          Log(Format('Error writting %s', [FileName]));
          Result := False;
        end else begin
          Log(Format('Modifications saved to %s', [FileName]));
        end;

        break;
      end;
    end;

    if Result and (State <> 2) then begin
      Log(Format('Spot to insert line was not found in %s', [FileName]));
      Result := False;
    end;
  end;
end;

// the pathdef.m line of the installers before v2026; the toolbox copies
// they left in <matlabroot>\toolbox\mcx are then no longer on the path
procedure RemoveLegacyPathdef;
var
  PathDef: string;
begin
  if (MatlabToolboxLocalPath = '') or not IsAdminInstallMode then exit;
  PathDef := MatlabToolboxLocalPath + '\local\pathdef.m';
  if FileExists(PathDef) then
    DeleteLineInTemplate(PathDef, '%%% BEGIN ENTRIES %%%', '%%% END ENTRIES %%%',
                         '%%% MCX PATHS %%%');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
    if CurStep <> ssPostInstall then exit;
    // the entries go at the end, so nothing of the system's is shadowed --
    // mmc included, as Windows has its own mmc.exe (Management Console)
    if IsTaskSelected('envPath') then
      EnvUpdate('Path', SuiteBinDirs, True);
    RemoveLegacyPathdef;
    if IsTaskSelected('addMATLABPath') then begin
      EnvUpdate('MATLABPATH', ToolboxPaths, True);
      EnvUpdate('OCTAVE_PATH', ToolboxPaths, True);
    end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
    if CurUninstallStep = usPostUninstall then begin
      EnvUpdate('Path', SuiteBinDirs, False);
      EnvUpdate('MATLABPATH', ToolboxPaths, False);
      EnvUpdate('OCTAVE_PATH', ToolboxPaths, False);
      InitializeSetup();
      RemoveLegacyPathdef;
    end;
end;
