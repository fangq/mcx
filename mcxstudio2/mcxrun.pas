{ mcxstudio2 - finding the simulators, asking them what they can do, and
  running one.

  No LCL in here.  Everything is either a pure function over text -- parse a
  device listing, find a percentage in a progress bar -- or a thread around a
  TProcess, so all of it can be tested without a display.

  There is one asynchronous path, and it is a plain TProcess read from a
  TThread.  The obvious alternative, TAsyncProcess.OnReadData, hooks the
  widget set through AddPipeEventHandler, which is implemented for gtk, qt and
  win32 but not for cocoa -- the base returns nil (intfbaselcl.inc:35).  That
  is the whole reason the old GUI carries a second, DARWIN-only busy-poll
  loop beside its event-driven one.  Output.Read blocks, so a thread
  needs neither: no polling, no Application.ProcessMessages, no Sleep. }
unit mcxrun;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Process, mcxdoc;

type
  { What a backend reported about one device.  The parser keeps the whole
    block as well, because the interesting fields differ between CUDA, OpenCL
    and the ROCm port, and a field none of them has today is not worth a
    record member. }
  TMcxDevice = record
    Id: Integer;
    Name: string;
    AutoThread: Integer;
    AutoBlock: Integer;
    Detail: string;
  end;
  TMcxDevices = array of TMcxDevice;

  TMcxLineEvent = procedure(Sender: TObject; const ALine: string) of object;
  TMcxProgressEvent = procedure(Sender: TObject; APercent: Integer) of object;
  TMcxDoneEvent = procedure(Sender: TObject; AExitCode: Integer) of object;

  { Runs one simulation and reports what it says.

    The events arrive on the main thread -- they are delivered through
    TThread.Queue -- so a handler may touch the interface directly. }
  TMcxRunner = class(TThread)
  private
    FExe: string;
    FArgs: TStringList;
    FDir: string;
    FProc: TProcess;
    FBuf: string;
    FEmit: string;
    FPercent: Integer;
    FLastPercent: Integer;
    FExitCode: Integer;
    FOnLine: TMcxLineEvent;
    FOnProgress: TMcxProgressEvent;
    FOnDone: TMcxDoneEvent;
    procedure Feed(const AChunk: string);
    procedure Flush;
    procedure SayLine;
    procedure SayProgress;
    procedure SayDone;
  protected
    procedure Execute; override;
  public
    constructor Create(const AExe: string; AArgs: TStrings; const ADir: string);
    destructor Destroy; override;
    { Asks the simulation to stop.  Terminate on its own only ends the read
      loop; the process has to be told separately or it runs to completion
      with nobody listening. }
    procedure Stop;
    property OnLine: TMcxLineEvent read FOnLine write FOnLine;
    property OnProgress: TMcxProgressEvent read FOnProgress write FOnProgress;
    property OnDone: TMcxDoneEvent read FOnDone write FOnDone;
  end;

{ The executable for a backend, or '' if it is not installed.  Looks beside
  mcxstudio2 first, then in the layout an MCXStudio release unpacks to, then
  on PATH. }
function McxFindExe(ABackend: TMcxBackend): string;

{ The name a backend's binary goes by.  The ROCm port renamed nothing, so HIP
  and CUDA are both "mcx" (src/Makefile:25) -- which is why a build cannot be
  told apart by its file name and has to be asked with -L. }
function McxExeName(ABackend: TMcxBackend): string;

{ Runs a backend with the given arguments and returns everything it printed.
  Blocking, for the short questions -- what devices are there, what
  benchmarks are there, what is this benchmark -- that happen on demand and
  answer at once.  A simulation goes through TMcxRunner instead. }
function McxAsk(const AExe: string; const AArgs: array of string;
  out AText: string): Boolean;

{ Runs <exe> -L and returns what it printed. }
function McxQueryDevices(const AExe: string; out AText: string): Boolean;

{ The names of mcx's built-in benchmarks, from -Q with no name. }
function McxBenchmarks(const AExe: string; AList: TStrings): Boolean;

{ The input JSON of one benchmark, from -Q <name> --dumpjson 2, which prints
  the simulation and exits before any GPU work. }
function McxBenchmarkJSON(const AExe, AName: string; out AJSON: string): Boolean;

{ The devices in a -L listing.

  Generic on purpose: every line is Key<tab>Value, a "Device n of m" line
  starts a new block, and anything else is kept as detail.  The old GUI
  sscanf'd 'Device %d of %d:%s' and read nothing else, so a backend that grew
  a field grew it invisibly. }
function McxParseDevices(const AText: string): TMcxDevices;

{ Text as it would look on a terminal: ANSI colour sequences removed and
  backspaces applied.  mcx draws its progress bar by writing a line, backing
  over it with as many \b as the terminal is wide, and writing it again
  (mcx_utils.c:5031), so without this the log grows by a screenful a tick. }
function McxPlain(const AText: string): string;

{ Takes whatever has arrived so far: moves the complete lines into ALines and
  leaves the unfinished tail in ABuf, collapsed as a terminal would show it.

  The tail is the point.  A progress bar never ends in a newline -- it is
  overwritten in place -- so a reader that only looked at finished lines would
  see no progress at all until the run was over. }
procedure McxSplitOutput(var ABuf: string; ALines: TStrings);

{ The percentage in a progress bar, if ATail ends in one.

  Scans back from the last '%' over digits and spaces and insists on a ']'.
  The ']' is the whole test: mcx prints "] %3d%%" (mcx_utils.c:5043) and also
  prints an absorption fraction as "...: %5.5f%%" (mcx_core.cu:4584), and
  nothing else distinguishes them. }
function McxProgressOf(const ATail: string; out APercent: Integer): Boolean;

{ The command line for a simulation.  Short, because the JSON carries the
  simulation -- which is why the old GUI's 230 lines of flag assembly are not
  here.

  A real file rather than -f -: mcx's stdin detection is positional and
  triggers on the next argument starting with '-' (mcx_utils.c:5202), a real
  path makes the logged command copy-pasteable, and relative paths inside the
  JSON need a working directory to resolve against. }
function McxBuildArgs(const AInputFile: string; ADoc, ARun: TMcxDoc): TStringList;

{ The same thing as one string, for the Command pane and for a bug report. }
function McxCommandLine(const AExe, AInputFile: string;
  ADoc, ARun: TMcxDoc): string;

implementation

const
  ExeNames: array[TMcxBackend] of string = ('mcx', 'mcxcl', 'mmc', 'mcx');

function McxExeName(ABackend: TMcxBackend): string;
begin
  Result := ExeNames[ABackend];
  {$IFDEF WINDOWS}
  Result := Result + '.exe';
  {$ENDIF}
end;

{ The first directory in APaths that holds AName, or ''.

  Written here rather than taken from LazUtils' SearchFileInPath, so that this
  unit needs nothing outside the FPC runtime -- which is what lets its tests
  run with no widget set and no display. }
function FindInPath(const AName, APaths: string): string;
var
  L: TStringList;
  i: Integer;
  Cand: string;
begin
  Result := '';
  L := TStringList.Create;
  try
    L.Delimiter := PathSeparator;
    L.StrictDelimiter := True;
    L.DelimitedText := APaths;
    for i := 0 to L.Count - 1 do
    begin
      if Trim(L[i]) = '' then Continue;
      Cand := IncludeTrailingPathDelimiter(Trim(L[i])) + AName;
      if FileExists(Cand) then Exit(ExpandFileName(Cand));
    end;
  finally
    L.Free;
  end;
end;

function McxFindExe(ABackend: TMcxBackend): string;
var
  Base, Name, Paths: string;
begin
  Name := McxExeName(ABackend);
  Base := ExtractFilePath(ParamStr(0));

  { Beside the binary, in the bin/ of a release tree, and beside the source
    tree a developer built in -- then whatever PATH says.

    Two levels of "beside the source tree", because mcxstudio2 lives inside
    the mcx checkout: from its bin/, mcxcl's own checkout is three levels up,
    not two.  A developer with mcx, mcxcl and mmc cloned next to each other is
    the normal case and was not being found. }
  Paths :=
    Base + PathSeparator +
    Base + ExeNames[ABackend] + PathDelim + 'bin' + PathSeparator +
    Base + '..' + PathDelim + 'bin' + PathSeparator +
    Base + '..' + PathDelim + '..' + PathDelim + 'bin' + PathSeparator +
    Base + '..' + PathDelim + '..' + PathDelim + ExeNames[ABackend] +
      PathDelim + 'bin' + PathSeparator +
    Base + '..' + PathDelim + '..' + PathDelim + '..' + PathDelim +
      ExeNames[ABackend] + PathDelim + 'bin' + PathSeparator +
    IncludeTrailingPathDelimiter(GetUserDir) + 'MCXStudio' + PathSeparator +
    GetEnvironmentVariable('PATH');

  Result := FindInPath(Name, Paths);
end;

function McxPlain(const AText: string): string;
var
  i, n: Integer;
begin
  SetLength(Result, Length(AText));
  n := 0;
  i := 1;
  while i <= Length(AText) do
  begin
    if (AText[i] = #27) and (i < Length(AText)) and (AText[i + 1] = '[') then
    begin
      { ESC [ parameters final-byte.  The final byte is the first one in
        @-~; everything before it is digits and semicolons. }
      Inc(i, 2);
      while (i <= Length(AText)) and not (AText[i] in ['@'..'~']) do Inc(i);
      Inc(i);
    end
    else if AText[i] = #8 then
    begin
      if n > 0 then Dec(n);
      Inc(i);
    end
    else if AText[i] = #13 then
      Inc(i)
    else
    begin
      Inc(n);
      Result[n] := AText[i];
      Inc(i);
    end;
  end;
  SetLength(Result, n);
end;

function McxProgressOf(const ATail: string; out APercent: Integer): Boolean;
var
  i, j: Integer;
begin
  Result := False;
  APercent := -1;

  i := Length(ATail);
  while (i > 0) and (ATail[i] <> '%') do Dec(i);
  if i = 0 then Exit;

  j := i - 1;
  while (j > 0) and (ATail[j] in ['0'..'9']) do Dec(j);
  if j = i - 1 then Exit;                      { a % with no number before it }
  APercent := StrToIntDef(Copy(ATail, j + 1, i - j - 1), -1);

  while (j > 0) and (ATail[j] = ' ') do Dec(j);
  if (j = 0) or (ATail[j] <> ']') then
  begin
    APercent := -1;
    Exit;
  end;

  Result := (APercent >= 0) and (APercent <= 100);
end;

{ 'Device 1 of 2' -> 1, or -1. }
function DeviceIndexOf(const AKey: string): Integer;
var
  Parts: TStringList;
begin
  Result := -1;
  if Pos('Device ', AKey) <> 1 then Exit;
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ' ';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := AKey;
    if (Parts.Count >= 2) then Result := StrToIntDef(Parts[1], -1);
  finally
    Parts.Free;
  end;
end;

function McxParseDevices(const AText: string): TMcxDevices;
var
  Lines: TStringList;
  i, Colon, Idx: Integer;
  Key, Value: string;
begin
  Result := nil;
  Lines := TStringList.Create;
  try
    Lines.Text := AText;
    for i := 0 to Lines.Count - 1 do
    begin
      Colon := Pos(':', Lines[i]);
      if Colon < 2 then Continue;
      Key := Trim(Copy(Lines[i], 1, Colon - 1));
      Value := Trim(Copy(Lines[i], Colon + 1, MaxInt));

      Idx := DeviceIndexOf(Key);
      if Idx >= 0 then
      begin
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)].Id := Idx;
        Result[High(Result)].Name := Value;
        Result[High(Result)].AutoThread := 0;
        Result[High(Result)].AutoBlock := 0;
        Result[High(Result)].Detail := '';
        Continue;
      end;

      if Length(Result) = 0 then Continue;     { a preamble line }
      if SameText(Key, 'Auto-thread') then
        Result[High(Result)].AutoThread := StrToIntDef(Value, 0)
      else if SameText(Key, 'Auto-block') then
        Result[High(Result)].AutoBlock := StrToIntDef(Value, 0);

      { Kept whole as well, so a field no version of this program knows about
        still reaches the person reading the list. }
      if Result[High(Result)].Detail <> '' then
        Result[High(Result)].Detail := Result[High(Result)].Detail + #10;
      Result[High(Result)].Detail := Result[High(Result)].Detail +
        Key + ': ' + Value;
    end;
  finally
    Lines.Free;
  end;
end;

function McxAsk(const AExe: string; const AArgs: array of string;
  out AText: string): Boolean;
var
  P: TProcess;
  Buf: array[0..4095] of Char;
  n, i: Integer;
begin
  Result := False;
  AText := '';
  if (AExe = '') or (not FileExists(AExe)) then Exit;
  P := TProcess.Create(nil);
  try
    P.Executable := AExe;
    for i := 0 to High(AArgs) do P.Parameters.Add(AArgs[i]);
    P.Options := [poUsePipes, poStderrToOutPut, poNoConsole];
    try
      P.Execute;
      repeat
        n := P.Output.Read(Buf, SizeOf(Buf));
        if n > 0 then AText := AText + Copy(Buf, 1, n);
      until n <= 0;
      P.WaitOnExit;
      Result := True;
    except
      { A backend that will not start is not an error worth raising: the
        caller wants an empty answer, not an exception. }
      Result := False;
    end;
  finally
    P.Free;
  end;
  AText := McxPlain(AText);
end;

function McxQueryDevices(const AExe: string; out AText: string): Boolean;
begin
  Result := McxAsk(AExe, ['-L'], AText);
end;

function McxBenchmarks(const AExe: string; AList: TStrings): Boolean;
var
  Text: string;
  Lines: TStringList;
  i: Integer;
  S: string;
begin
  AList.Clear;
  Result := McxAsk(AExe, ['-Q'], Text);
  if not Result then Exit;
  Lines := TStringList.Create;
  try
    Lines.Text := Text;
    { The listing is a heading and then one tab-indented name a line
      (mcx_utils.c:5528).  Taking only the indented lines skips the heading
      without matching its wording, which is translated. }
    for i := 0 to Lines.Count - 1 do
    begin
      S := Lines[i];
      if (S = '') or (S[1] <> #9) then Continue;
      S := Trim(S);
      if S <> '' then AList.Add(S);
    end;
  finally
    Lines.Free;
  end;
  Result := AList.Count > 0;
end;

function McxBenchmarkJSON(const AExe, AName: string; out AJSON: string): Boolean;
var
  Text: string;
  p: Integer;
begin
  AJSON := '';
  Result := McxAsk(AExe, ['-Q', AName, '--dumpjson', '2', '-n', '0'], Text);
  if not Result then Exit;
  { mcx prints its banner first, so the document starts at the first brace. }
  p := Pos('{', Text);
  Result := p > 0;
  if Result then AJSON := Copy(Text, p, MaxInt);
end;

function McxBuildArgs(const AInputFile: string; ADoc, ARun: TMcxDoc): TStringList;
var
  S: string;
begin
  Result := TStringList.Create;
  Result.Add('-f');
  Result.Add(ExtractFileName(AInputFile));

  S := ADoc.AsStr('Session.ID', '');
  if S <> '' then
  begin
    Result.Add('-s');
    Result.Add(S);
  end;

  S := ARun.AsStr('@run.device', '');
  if S <> '' then
  begin
    Result.Add('-G');
    Result.Add(S);
  end;

  S := ARun.AsStr('@run.workload', '');
  if S <> '' then
  begin
    Result.Add('-W');
    Result.Add(S);
  end;

  { Thread and block size are only meaningful with autopilot off, and mcx
    wants to be told so explicitly. }
  if not ADoc.AsBool('Session.DoAutoThread') then
  begin
    Result.Add('-A');
    Result.Add('0');
    if ARun.AsInt('@run.nthread') > 0 then
    begin
      Result.Add('-t');
      Result.Add(IntToStr(ARun.AsInt('@run.nthread')));
    end;
    if ARun.AsInt('@run.nblock') > 0 then
    begin
      Result.Add('-T');
      Result.Add(IntToStr(ARun.AsInt('@run.nblock')));
    end;
  end;

  { The progress bar is how the window knows how far along a run is, so P is
    always asked for -- but added to whatever the file asked for rather than
    instead of it.  mcx's -D replaces the JSON's DebugFlag outright, so
    passing a bare P silently turned off the trajectory recording the person
    had just ticked, and no file appeared. }
  S := ADoc.AsStr('Session.DebugFlag', '');
  if Pos('P', UpperCase(S)) = 0 then S := S + 'P';
  Result.Add('-D');
  Result.Add(S);

  { How many trajectory positions to keep.  Command line only -- mcx's parser
    has no JSON key for it -- and it matters: the default is ten million, and
    ten million positions is a 200 MB text file that takes longer to write
    than the simulation took to run and longer again to read back. }
  if (Pos('M', UpperCase(S)) > 0) and (ARun.AsInt('@run.maxjumpdebug') > 0) then
  begin
    Result.Add('--maxjumpdebug');
    Result.Add(IntToStr(ARun.AsInt('@run.maxjumpdebug')));
  end;
end;

function McxCommandLine(const AExe, AInputFile: string;
  ADoc, ARun: TMcxDoc): string;
var
  Args: TStringList;
  i: Integer;
begin
  if AExe = '' then Result := 'mcx' else Result := ExtractFileName(AExe);
  Args := McxBuildArgs(AInputFile, ADoc, ARun);
  try
    for i := 0 to Args.Count - 1 do
      if Pos(' ', Args[i]) > 0 then
        Result := Result + ' "' + Args[i] + '"'
      else
        Result := Result + ' ' + Args[i];
  finally
    Args.Free;
  end;
end;

procedure McxSplitOutput(var ABuf: string; ALines: TStrings);
var
  p: Integer;
  Line: string;
begin
  p := Pos(#10, ABuf);
  while p > 0 do
  begin
    Line := McxPlain(Copy(ABuf, 1, p - 1));
    Delete(ABuf, 1, p);
    if Line <> '' then ALines.Add(Line);
    p := Pos(#10, ABuf);
  end;
  { Collapsed every time, or the tail grows by a screenful of backspaces on
    every redraw of the bar. }
  ABuf := McxPlain(ABuf);
end;

{ TMcxRunner }

constructor TMcxRunner.Create(const AExe: string; AArgs: TStrings;
  const ADir: string);
begin
  inherited Create(True);
  FreeOnTerminate := False;
  FExe := AExe;
  FDir := ADir;
  FArgs := TStringList.Create;
  if AArgs <> nil then FArgs.Assign(AArgs);
  FLastPercent := -1;
end;

destructor TMcxRunner.Destroy;
begin
  FArgs.Free;
  inherited Destroy;
end;

procedure TMcxRunner.Stop;
begin
  Terminate;
  if (FProc <> nil) and FProc.Running then
    FProc.Terminate(1);
end;

procedure TMcxRunner.SayLine;
begin
  if Assigned(FOnLine) then FOnLine(Self, FEmit);
end;

procedure TMcxRunner.SayProgress;
begin
  if Assigned(FOnProgress) then FOnProgress(Self, FPercent);
end;

procedure TMcxRunner.SayDone;
begin
  if Assigned(FOnDone) then FOnDone(Self, FExitCode);
end;

procedure TMcxRunner.Feed(const AChunk: string);
var
  Lines: TStringList;
  i: Integer;
begin
  FBuf := FBuf + AChunk;
  Lines := TStringList.Create;
  try
    McxSplitOutput(FBuf, Lines);
    for i := 0 to Lines.Count - 1 do
    begin
      { A finished line can be a progress bar too.  The bar is only ended by
        a newline when the run is over, so when a whole run arrives in one
        read -- which is what happens to a fast simulation, or to anything
        read from a file -- the final 100% is a line rather than a tail.
        Reading it only from the tail loses the last reading entirely. }
      if McxProgressOf(Lines[i], FPercent) then
      begin
        if FPercent <> FLastPercent then
        begin
          FLastPercent := FPercent;
          Synchronize(@SayProgress);
        end;
        Continue;
      end;

      FEmit := Lines[i];
      { Synchronize rather than Queue: Queue returns at once, so the next
        line through this loop would overwrite FEmit before the first had
        been read, and a chunk holding five lines would report the fifth
        five times. }
      Synchronize(@SayLine);
    end;
  finally
    Lines.Free;
  end;

  if McxProgressOf(FBuf, FPercent) and (FPercent <> FLastPercent) then
  begin
    FLastPercent := FPercent;
    Synchronize(@SayProgress);
  end;
end;

procedure TMcxRunner.Flush;
begin
  FBuf := McxPlain(FBuf);
  if Trim(FBuf) <> '' then
  begin
    FEmit := FBuf;
    FBuf := '';
    Synchronize(@SayLine);
  end;
end;

procedure TMcxRunner.Execute;
var
  Buf: array[0..4095] of Char;
  n: Integer;
begin
  FExitCode := -1;
  FProc := TProcess.Create(nil);
  try
    FProc.Executable := FExe;
    FProc.Parameters.Assign(FArgs);
    if FDir <> '' then FProc.CurrentDirectory := FDir;
    FProc.Options := [poUsePipes, poStderrToOutPut, poNoConsole];
    try
      FProc.Execute;
      { Read blocks until there is something or the pipe closes, which is what
        makes this a thread rather than a timer. }
      repeat
        n := FProc.Output.Read(Buf, SizeOf(Buf));
        if n > 0 then Feed(Copy(Buf, 1, n));
      until (n <= 0) or Terminated;
      if not Terminated then FProc.WaitOnExit;
      FExitCode := FProc.ExitStatus;
    except
      on E: Exception do
      begin
        FEmit := 'could not start ' + FExe + ': ' + E.Message;
        Synchronize(@SayLine);
      end;
    end;
  finally
    Flush;
    FreeAndNil(FProc);
  end;
  Queue(@SayDone);
end;

end.
