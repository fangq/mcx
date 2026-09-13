{ mcxstudio2 - round-trip tests for the document model.

  Links mcxdoc alone: no LCL, no widgetset, no display.  That is the reason
  mcxdoc has no LCL dependency, and it means this runs in CI unchanged.

  The claim under test is the one the whole design rests on: loading a file and
  saving it again changes nothing mcx would notice, including the Help blocks
  and the compressed JData volumes that the GUI never looks at.  The old GUI
  could not pass this -- it rebuilt a fresh tree from 46 stringified columns
  and dropped everything it did not have a column for.

  Usage: mcxdoctest [path-to-example-dir] }
program mcxdoctest;

{$mode objfpc}{$H+}

uses
  { First, and before anything that starts a thread: without it the runner
    test dies with "This binary has no thread support compiled in".  The GUI
    gets this from the LCL; a console program has to ask. }
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Classes, zstream, fpjson, mcxdoc, mcxrun, mcxjd, mcxmesh;

var
  Checks, Failures: Integer;

procedure Ok(ACond: Boolean; const AWhat: string);
begin
  Inc(Checks);
  if ACond then Exit;
  Inc(Failures);
  WriteLn('  FAIL  ', AWhat);
end;

{ ---------------------------------------------------------------- paths --- }

procedure TestPaths;
var
  D: TMcxDoc;
  A: TJSONData;
begin
  WriteLn('paths');
  D := TMcxDoc.Create;
  try
    { Ensure builds every missing container on the way in, picking an object or
      an array from whether the next step is a name or an index. }
    D.SetNum('Session.Photons', 1e7);
    Ok(D.AsNum('Session.Photons') = 1e7, 'a nested number round-trips');
    Ok(D.Root.Find('Session') <> nil, 'the intermediate object was created');

    D.SetNum('Optode.Source.Pos[2]', 5);
    A := D.Find('Optode.Source.Pos');
    Ok((A <> nil) and (A.JSONType = jtArray), 'an index step creates an array');
    Ok((A <> nil) and (TJSONArray(A).Count = 3), 'the array was padded to the index');
    Ok(D.AsNum('Optode.Source.Pos[2]') = 5, 'the indexed element reads back');

    Ok(not D.Exists('Nothing.Here'), 'a missing path does not report as present');
    Ok(D.Find('Nothing.Here') = nil, 'and Find leaves it missing');
    Ok(not D.Exists('Session.Photons[0]'), 'indexing a scalar finds nothing');

    D.Delete('Session.Photons');
    Ok(not D.Exists('Session.Photons'), 'Delete removes the key');
  finally
    D.Free;
  end;
end;

{ ------------------------------------------------------ type preservation -- }

procedure TestTypePreservation;
var
  D: TMcxDoc;
begin
  WriteLn('type preservation');
  D := TMcxDoc.Create;
  try
    { mcx reads 1 and true alike, so whichever the file used has to survive
      being edited, or the first save would rewrite every example. }
    D.LoadFromString('{"Session":{"DoMismatch":1,"DoSaveRef":true}}');
    D.SetBool('Session.DoMismatch', False);
    Ok(D.Find('Session.DoMismatch').JSONType = jtNumber,
       'a flag stored as a number stays a number');
    Ok(D.AsInt('Session.DoMismatch') = 0, 'and takes the new value');

    D.SetBool('Session.DoSaveRef', False);
    Ok(D.Find('Session.DoSaveRef').JSONType = jtBoolean,
       'a flag stored as a boolean stays a boolean');

    D.LoadFromString('{"Domain":{"Dim":[60,60,60],"LengthUnit":0.5}}');
    D.SetNum('Domain.Dim[0]', 100);
    Ok(D.Find('Domain.Dim[0]').JSONType = jtNumber, 'a dimension stays a number');
    Ok(Pos('.', D.Find('Domain.Dim[0]').AsString) = 0,
       'and an integer dimension does not sprout a fraction');

    D.SetNum('Domain.LengthUnit', 0.25);
    Ok(D.AsNum('Domain.LengthUnit') = 0.25, 'a fractional value keeps its fraction');

    { A value that needs a fraction must be allowed to widen an integer key. }
    D.SetNum('Domain.Dim[1]', 1.5);
    Ok(D.AsNum('Domain.Dim[1]') = 1.5, 'an integer key widens when it has to');

    { Writing the value that is already there must not mark the document
      dirty, or every load would offer to save. }
    D.LoadFromString('{"Session":{"ID":"cube60"}}');
    D.Modified := False;
    D.SetStr('Session.ID', 'cube60');
    Ok(not D.Modified, 'writing an unchanged value leaves the document clean');
    D.SetStr('Session.ID', 'other');
    Ok(D.Modified, 'writing a changed value marks it modified');
  finally
    D.Free;
  end;
end;

{ ------------------------------------------------------------- compare ---- }

procedure TestCompare;
var
  A, B: TMcxDoc;
  Diff: string;
begin
  WriteLn('semantic compare');
  A := TMcxDoc.Create;
  B := TMcxDoc.Create;
  try
    A.LoadFromString('{"a":1,"b":2}');
    B.LoadFromString('{"b":2,"a":1}');
    Ok(McxJSONSame(A.Root, B.Root, Diff), 'key order does not matter');

    A.LoadFromString('{"f":true}');
    B.LoadFromString('{"f":1}');
    Ok(McxJSONSame(A.Root, B.Root, Diff), 'true and 1 are the same flag');

    A.LoadFromString('{"t":5e-09}');
    B.LoadFromString('{"t":0.000000005}');
    Ok(McxJSONSame(A.Root, B.Root, Diff), 'float spelling does not matter');

    A.LoadFromString('{"f":true}');
    B.LoadFromString('{"f":0}');
    Ok(not McxJSONSame(A.Root, B.Root, Diff), 'a genuinely different flag is caught');

    A.LoadFromString('{"a":1}');
    B.LoadFromString('{"a":1,"b":2}');
    Ok(not McxJSONSame(A.Root, B.Root, Diff), 'an extra key is caught');
  finally
    A.Free;
    B.Free;
  end;
end;

{ --------------------------------------------------------- the corpus ----- }

procedure CollectJSON(const ADir: string; AList: TStrings);
var
  R: TSearchRec;
begin
  if FindFirst(IncludeTrailingPathDelimiter(ADir) + '*', faAnyFile, R) = 0 then
  try
    repeat
      if (R.Name = '.') or (R.Name = '..') then Continue;
      if (R.Attr and faDirectory) <> 0 then
        CollectJSON(IncludeTrailingPathDelimiter(ADir) + R.Name, AList)
      else if LowerCase(ExtractFileExt(R.Name)) = '.json' then
        AList.Add(IncludeTrailingPathDelimiter(ADir) + R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
end;

procedure TestCorpus(const ADir: string);
var
  Files: TStringList;
  i, Loaded, Skipped: Integer;
  A, B: TMcxDoc;
  Tmp, Diff: string;
begin
  WriteLn('round-trip over ', ADir);
  Files := TStringList.Create;
  try
    CollectJSON(ADir, Files);
    Files.Sort;
    if Files.Count = 0 then
    begin
      WriteLn('  no .json files found -- is the example path right?');
      Inc(Failures);
      Exit;
    end;

    Loaded := 0;
    Skipped := 0;
    Tmp := GetTempFileName('', 'mcxrt');
    for i := 0 to Files.Count - 1 do
    begin
      A := TMcxDoc.Create;
      B := TMcxDoc.Create;
      try
        if not A.LoadFromFile(Files[i]) then
        begin
          { Not every .json under example/ is an mcx input; some are notes or
            fragments.  Report them but do not fail on them. }
          Inc(Skipped);
          WriteLn('  skip  ', ExtractFileName(Files[i]), ' -- ', A.LastError);
          Continue;
        end;
        Inc(Loaded);
        A.SaveToFile(Tmp);
        Ok(B.LoadFromFile(Tmp), 'reloads: ' + ExtractFileName(Files[i]));
        if McxJSONSame(A.Root, B.Root, Diff) then
          Ok(True, 'round-trips: ' + ExtractFileName(Files[i]))
        else
          Ok(False, 'round-trips: ' + ExtractFileName(Files[i]) + ' -- ' + Diff);
      finally
        A.Free;
        B.Free;
      end;
    end;
    if FileExists(Tmp) then DeleteFile(Tmp);
    WriteLn(Format('  %d files loaded, %d skipped', [Loaded, Skipped]));
  finally
    Files.Free;
  end;
end;


{ ---------------------------------------------------------------- mcxrun -- }

procedure TestPlainText;
begin
  WriteLn('terminal text');
  Ok(McxPlain('plain') = 'plain', 'text with nothing in it is unchanged');
  Ok(McxPlain(#27'[33mProgress'#27'[0m') = 'Progress', 'colour codes removed');
  Ok(McxPlain('abc'#8#8'X') = 'aX', 'backspaces rub out what came before');
  Ok(McxPlain('ab'#13'cd') = 'abcd', 'carriage returns dropped');
  Ok(McxPlain(#8#8'ab') = 'ab', 'a backspace at the start has nothing to eat');
  Ok(McxPlain(#27'[') = '', 'a truncated escape does not run off the end');
end;

procedure TestProgress;
var
  P: Integer;
begin
  WriteLn('progress bar');
  Ok(McxProgressOf('Progress: [======   ]  62%', P) and (P = 62),
     'a percentage after a bracket is progress');
  Ok(McxProgressOf('Progress: [=========] 100%', P) and (P = 100),
     'the last tick reads 100');
  Ok(McxProgressOf('Progress: [] 0%', P) and (P = 0), 'the first tick reads 0');

  { The line mcx prints when it is done: a percentage with no bracket before
    it.  Reading this as progress is exactly the mistake the bracket rule is
    there to prevent. }
  Ok(not McxProgressOf('absorbed: 52.61382%', P),
     'an absorption fraction is not progress');
  Ok(not McxProgressOf('nothing here', P), 'text with no percentage is not progress');
  Ok(not McxProgressOf('] %', P), 'a bracket and a bare percent is not progress');
  Ok(not McxProgressOf('62%', P), 'a percentage with no bracket is not progress');
end;

{ Replays a captured run at several read sizes and checks the percentages
  that come out.

  A chunk boundary can land mid-escape, mid-number or mid-backspace, and each
  of those broke a different draft of this.

  The invariant is not "every step is seen": at a large read size a whole run
  can arrive in one go, and the only percentage still on screen is the last
  one, which is what a terminal would show too.  What has to hold is that the
  readings never go backwards, never leave 0..100, and end at 100.  Only the
  byte-at-a-time replay sees every step, because there every redraw is its own
  read.

  The fixture writes the backspaces before each bar rather than after it,
  which is the order mcx_utils.c:5031 uses -- it backs over the previous bar
  and then draws the new one.  Written the other way round the run ends with
  the bar rubbed out and no percentage anywhere, which is how this test first
  failed. }
procedure TestProgressStream;
const
  Widths: array[0..5] of Integer = (1, 2, 3, 7, 64, 4096);
var
  Raw: string;
  w, i, j, n, Last, Steps: Integer;
  Buf, Seen: string;
  P: Integer;
  Good: Boolean;
  Lines: TStringList;
begin
  WriteLn('progress stream, chopped');
  Lines := TStringList.Create;
  try
  Raw := '';
  for i := 0 to 10 do
    Raw := Raw + StringOfChar(#8, 40) + #27'[33mProgress: [' +
      StringOfChar('=', i) + StringOfChar(' ', 10 - i) + ']' +
      Format('%3d', [i * 10]) + '%' + #27'[0m';
  Raw := Raw + #10'absorbed: 52.61382%'#10;

  for w := 0 to High(Widths) do
  begin
    Buf := '';
    Seen := '';
    Last := -1;
    Steps := 0;
    Good := True;
    i := 1;
    while i <= Length(Raw) do
    begin
      n := Widths[w];
      if i + n - 1 > Length(Raw) then n := Length(Raw) - i + 1;
      Buf := Buf + Copy(Raw, i, n);
      Inc(i, n);
      { Exactly what the runner does with a chunk: take the finished lines
        out of the way, then read the tail. }
      Lines.Clear;
      McxSplitOutput(Buf, Lines);
      { A finished line can be a bar too -- see Feed. }
      for j := 0 to Lines.Count - 1 do
        if McxProgressOf(Lines[j], P) and (P <> Last) then
        begin
          if (P < Last) or (P < 0) or (P > 100) then Good := False;
          Last := P;
          Inc(Steps);
          Seen := Seen + IntToStr(P) + ' ';
        end;
      if McxProgressOf(Buf, P) and (P <> Last) then
      begin
        if (P < Last) or (P < 0) or (P > 100) then Good := False;
        Last := P;
        Inc(Steps);
        Seen := Seen + IntToStr(P) + ' ';
      end;
    end;
    Ok(Good, Format('reads of %d bytes: never backwards, never out of range',
      [Widths[w]]));
    Ok(Last = 100, Format('reads of %d bytes: ends at 100 (%s)',
      [Widths[w], Seen]));
    if Widths[w] = 1 then
      Ok(Steps = 11, 'a byte at a time sees all eleven steps');
  end;
  finally
    Lines.Free;
  end;
end;

procedure TestDevices;
const
  Listing =
    '=============   GPU Infomation  ================'#10 +
    'Device 1 of 2:'#9#9'NVIDIA GeForce RTX 4090'#10 +
    'Compute Capability:'#9'8.9'#10 +
    'Global Memory:'#9#9'25393692672 B'#10 +
    'Number of Cores:'#9'16384'#10 +
    'Auto-thread:'#9#9'1048576'#10 +
    'Auto-block:'#9#9'64'#10 +
    'Device 2 of 2:'#9#9'NVIDIA GeForce RTX 3090'#10 +
    'Auto-thread:'#9#9'524288'#10 +
    'Auto-block:'#9#9'64'#10;
var
  D: TMcxDevices;
begin
  WriteLn('device listing');
  D := McxParseDevices(Listing);
  Ok(Length(D) = 2, 'both devices found');
  if Length(D) < 2 then Exit;
  Ok(D[0].Id = 1, 'the first is device 1');
  Ok(D[0].Name = 'NVIDIA GeForce RTX 4090', 'the name is what followed the tab');
  Ok(D[0].AutoThread = 1048576, 'auto-thread read');
  Ok(D[0].AutoBlock = 64, 'auto-block read');
  Ok(D[1].Id = 2, 'the second is device 2');
  Ok(D[1].Name = 'NVIDIA GeForce RTX 3090', 'and keeps its own name');

  { The point of a key/value parser rather than one sscanf: a field this
    program has never heard of still reaches the person reading the list. }
  Ok(Pos('Compute Capability: 8.9', D[0].Detail) > 0,
     'fields with no record member of their own are kept as detail');
  Ok(Pos('Number of Cores: 16384', D[0].Detail) > 0, 'and so are the rest');

  Ok(Length(McxParseDevices('')) = 0, 'nothing in, nothing out');
  Ok(Length(McxParseDevices('no devices here'#10)) = 0,
     'text that is not a listing yields no devices');
end;

procedure TestArgs;
var
  D, R: TMcxDoc;
  Cmd: string;
begin
  WriteLn('command line');
  D := TMcxDoc.Create;
  R := TMcxDoc.Create;
  try
    D.LoadFromString('{"Session":{"ID":"box","DoAutoThread":1}}');
    R.SetStr('@run.backend', 'mcx');
    Cmd := McxCommandLine('/opt/mcx/bin/mcx', '/tmp/box.json', D, R);
    Ok(Pos('-f box.json', Cmd) > 0, 'the input is passed by name');
    Ok(Pos('-s box', Cmd) > 0, 'the session id is passed');
    Ok(Pos('-D P', Cmd) > 0, 'the progress bar is always asked for');
    Ok(Pos('-A 0', Cmd) = 0, 'autopilot on means no thread flags');

    D.SetBool('Session.DoAutoThread', False);
    R.SetInt('@run.nthread', 65536);
    R.SetInt('@run.nblock', 64);
    R.SetStr('@run.device', '11');
    Cmd := McxCommandLine('/opt/mcx/bin/mcx', '/tmp/box.json', D, R);
    Ok(Pos('-A 0', Cmd) > 0, 'autopilot off is stated explicitly');
    Ok(Pos('-t 65536', Cmd) > 0, 'the thread count is passed');
    Ok(Pos('-T 64', Cmd) > 0, 'the block size is passed');
    Ok(Pos('-G 11', Cmd) > 0, 'the device mask is passed');
  finally
    R.Free;
    D.Free;
  end;
end;


{ Runs a real process through TMcxRunner and checks that the whole chain
  works: a thread, a pipe read that blocks, Synchronize back to the main
  thread, and a process that ends.

  Unix only, because it needs a shell to play the part of mcx.  Everything
  else in this file is portable; this is the one test that has to start
  something.

  Synchronize parks the worker until the main thread calls CheckSynchronize,
  which a console program has to do for itself -- there is no message loop
  here to do it. }
{$IFDEF UNIX}
type
  TRunSpy = class
  public
    Lines: TStringList;
    Percents: TStringList;
    Code: Integer;
    Finished: Boolean;
    constructor Create;
    destructor Destroy; override;
    procedure Line(Sender: TObject; const AText: string);
    procedure Progress(Sender: TObject; APercent: Integer);
    procedure Done(Sender: TObject; AExitCode: Integer);
  end;

constructor TRunSpy.Create;
begin
  Lines := TStringList.Create;
  Percents := TStringList.Create;
  Code := -999;
end;

destructor TRunSpy.Destroy;
begin
  Percents.Free;
  Lines.Free;
  inherited Destroy;
end;

procedure TRunSpy.Line(Sender: TObject; const AText: string);
begin
  Lines.Add(AText);
end;

procedure TRunSpy.Progress(Sender: TObject; APercent: Integer);
begin
  Percents.Add(IntToStr(APercent));
end;

procedure TRunSpy.Done(Sender: TObject; AExitCode: Integer);
begin
  Code := AExitCode;
  Finished := True;
end;

procedure TestRunner;
var
  Script: string;
  F: TStringList;
  Spy: TRunSpy;
  R: TMcxRunner;
  Args: TStringList;
  Waited: Integer;
begin
  WriteLn('running a process');
  Script := GetTempDir(False) + 'mcxstudio2-runner-test.sh';
  F := TStringList.Create;
  try
    F.Add('#!/bin/sh');
    F.Add('echo "MCX Revision stand-in"');
    F.Add('i=0');
    F.Add('while [ $i -le 10 ]; do');
    { \b eight times, then the bar: the order mcx uses. }
    F.Add('  printf "\b\b\b\b\b\b\b\bProgress: [] %3d%%" $((i * 10))');
    F.Add('  i=$((i + 1))');
    F.Add('done');
    F.Add('echo ""');
    F.Add('echo "absorbed: 52.61382%"');
    F.Add('exit 0');
    F.SaveToFile(Script);
  finally
    F.Free;
  end;

  Spy := TRunSpy.Create;
  Args := TStringList.Create;
  try
    Args.Add(Script);
    R := TMcxRunner.Create('/bin/sh', Args, '');
    R.OnLine := @Spy.Line;
    R.OnProgress := @Spy.Progress;
    R.OnDone := @Spy.Done;
    R.Start;

    { A bounded wait: a hang here would otherwise stop the whole suite with
      no output at all. }
    Waited := 0;
    while (not Spy.Finished) and (Waited < 1000) do
    begin
      CheckSynchronize(10);
      Inc(Waited);
    end;
    R.WaitFor;
    R.Free;

    Ok(Spy.Finished, 'the run reported that it finished');
    Ok(Spy.Code = 0, 'and passed on the exit code');
    Ok(Spy.Lines.IndexOf('MCX Revision stand-in') >= 0,
       'a line of output arrived intact');
    Ok(Spy.Lines.IndexOf('absorbed: 52.61382%') >= 0,
       'a percentage that is not progress stayed in the log');
    Ok(Spy.Percents.Count > 0, 'progress was reported at least once');
    if Spy.Percents.Count > 0 then
      Ok(Spy.Percents[Spy.Percents.Count - 1] = '100',
         'and the last reading is 100');
    Ok(Spy.Lines.IndexOf('Progress: [] 100%') < 0,
       'the bar itself did not end up in the log');
  finally
    Args.Free;
    Spy.Free;
    DeleteFile(Script);
  end;
end;
{$ENDIF}


{ ----------------------------------------------------------------- mcxjd -- }

{ Every file with this extension in one directory, no recursion. }
procedure CollectExt(const ADir, AExt: string; AList: TStrings);
var
  R: TSearchRec;
begin
  if FindFirst(IncludeTrailingPathDelimiter(ADir) + '*' + AExt, faAnyFile, R) <> 0 then
    Exit;
  try
    repeat
      if (R.Attr and faDirectory) = 0 then
        AList.Add(IncludeTrailingPathDelimiter(ADir) + R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
end;

procedure TestJData;
var
  A: TMcxArray;
  Files: TStringList;
  i, j, Found: Integer;
  Total: Int64;
  Lo, Hi: Double;
  Dir: string;
begin
  WriteLn('JNIfTI arrays');
  { mcx's own output, if any is lying about -- these are results, not inputs,
    so they are not in the repository and the test says so rather than
    failing when they are absent. }
  Dir := '../bin/';
  Files := TStringList.Create;
  try
    CollectExt(Dir, '.jnii', Files);
    if Files.Count = 0 then
    begin
      WriteLn('  no .jnii files in ', Dir, ' -- run a simulation to make some');
      Exit;
    end;
    Found := 0;
    for i := 0 to Files.Count - 1 do
    begin
      if not McxLoadArray(Files[i], '', A) then
      begin
        WriteLn('  FAIL  could not read ', ExtractFileName(Files[i]));
        Inc(Failures);
        Continue;
      end;
      Inc(Found);
      Ok(A.Kind <> akNone, ExtractFileName(Files[i]) + ': knows its type');
      Ok(McxArrayCount(A) > 0, ExtractFileName(Files[i]) + ': has elements');
      { The shape and the payload have to agree, which is the one thing a
        wrong element size or a short inflate would break. }
      if Length(A.Dims) > 0 then
      begin
        Total := 1;
        for j := 0 to Length(A.Dims) - 1 do Total := Total * A.Dims[j];
        Ok(Total = McxArrayCount(A),
           ExtractFileName(Files[i]) + ': shape matches the data length');
      end;
      { A file of NaN throughout is a run that diverged rather than a decode
        that went wrong, so it is reported and not counted against the reader.
        What is asserted either way is that asking for the range of one does
        not raise -- which, before McxArrayRange stepped over the non-finite,
        is exactly what it did. }
      if McxArrayRange(A, Lo, Hi) then
        Ok(Hi > Lo, ExtractFileName(Files[i]) + ': the values are not all one')
      else
        WriteLn('  note  ', ExtractFileName(Files[i]),
                ': no finite values -- the run that wrote it did not converge');
    end;
    WriteLn(Format('  %d file(s) decoded', [Found]));
  finally
    Files.Free;
  end;
end;

{ Builds a BJData document byte by byte and reads it back.

  A fixture rather than a file, because there is no .bnii in the repository
  and a test that skips when its input is missing is a test that never runs.
  Every byte here is what src/ubj/ubjw.c would emit. }
procedure TestBJData;
var
  M: TMemoryStream;
  BJ: TMcxBJData;
  A: TMcxArray;
  i: Integer;
  f: Single;
  d: TJSONData;
  Parsed: Boolean;

  procedure PutByte(B: Byte);
  begin
    M.Write(B, 1);
  end;

  procedure PutMark(C: Char);
  begin
    PutByte(Ord(C));
  end;

  { A key: an integer marker, a length, the text.  No S marker -- that is
    the one place BJData leaves it out.  An element of a container whose
    type is already S is written the same way, and for the same reason. }
  procedure PutKey(const S: string);
  var
    j: Integer;
  begin
    PutMark('U');
    PutByte(Length(S));
    for j := 1 to Length(S) do PutByte(Ord(S[j]));
  end;

  { A string value: the marker the container did not supply, then the key
    form above. }
  procedure PutText(const S: string);
  begin
    PutMark('S');
    PutKey(S);
  end;

  { Deflates a run of singles, the way mcx compresses a volume before
    writing it as _ArrayZipData_. }
  function Zip(ACount: Integer; AScale: Single): TBytes;
  var
    Src, Dst: TMemoryStream;
    C: TCompressionStream;
    j: Integer;
    v: Single;
  begin
    Src := TMemoryStream.Create;
    Dst := TMemoryStream.Create;
    try
      for j := 0 to ACount - 1 do
      begin
        v := j * AScale;
        Src.Write(v, 4);
      end;
      Src.Position := 0;
      C := TCompressionStream.Create(clDefault, Dst);
      try
        C.CopyFrom(Src, Src.Size);
      finally
        C.Free;
      end;
      SetLength(Result, Dst.Size);
      Move(Dst.Memory^, Result[0], Dst.Size);
    finally
      Dst.Free;
      Src.Free;
    end;
  end;

  { An annotated array whose payload is the deflated bytes, which is the
    shape of every result mcx writes as .bnii or .jdb. }
  procedure PutZipped(const AKey: string; ACount: Integer; AScale: Single);
  var
    Z: TBytes;
    j, k: Integer;
  begin
    Z := Zip(ACount, AScale);
    PutKey(AKey);
    PutMark('{');
    PutKey('_ArrayType_'); PutText('single');
    PutKey('_ArraySize_');
    PutMark('['); PutMark('$'); PutMark('l'); PutMark('#');
    PutMark('U'); PutByte(1);
    k := ACount; M.Write(k, 4);
    PutKey('_ArrayZipType_'); PutText('zlib');
    PutKey('_ArrayZipSize_'); PutMark('l'); k := ACount; M.Write(k, 4);
    PutKey('_ArrayZipData_');
    PutMark('['); PutMark('$'); PutMark('U'); PutMark('#');
    PutMark('l'); k := Length(Z); M.Write(k, 4);
    for j := 0 to High(Z) do PutByte(Z[j]);
    PutMark('}');
  end;

begin
  WriteLn('BJData');
  M := TMemoryStream.Create;
  BJ := TMcxBJData.Create;
  try
    PutMark('{');

    PutKey('name');
    PutMark('S'); PutMark('U'); PutByte(3);
    PutByte(Ord('b')); PutByte(Ord('o')); PutByte(Ord('x'));

    PutKey('count');
    PutMark('l');
    i := 216000;
    M.Write(i, 4);                      { little-endian, as mcx writes it }

    PutKey('flag');
    PutMark('T');

    { Small typed array: expanded, because a shape has to read as a list. }
    PutKey('Dim');
    PutMark('['); PutMark('$'); PutMark('U'); PutMark('#');
    PutMark('U'); PutByte(3);
    PutByte(60); PutByte(60); PutByte(60);

    { Large typed array: kept as bytes. }
    PutKey('Data');
    PutMark('['); PutMark('$'); PutMark('d'); PutMark('#');
    PutMark('u'); i := 100; M.Write(i, 2);
    for i := 0 to 99 do
    begin
      f := i * 0.5;
      M.Write(f, 4);
    end;

    { A container of a marker-only type carries no payload at all. }
    PutKey('Empties');
    PutMark('['); PutMark('$'); PutMark('T'); PutMark('#');
    PutMark('U'); PutByte(4);

    { A container of strings.  The type is on the container, so an element is
      a length and its bytes with no S of its own.  This is not an exotic
      corner: mcx opens every .bnii and .jdb with _DataInfo_, and
      _DataInfo_.Parser holds three of these (mcx_utils.c:673) before the
      file says anything else. }
    PutKey('Parser');
    PutMark('['); PutMark('$'); PutMark('S'); PutMark('#');
    PutMark('U'); PutByte(2);
    PutKey('https://neurojson.org/download/pyjdata');
    PutKey('https://neurojson.org/download/pybjdata');

    { Compressed payloads, either side of the threshold at which the reader
      stops expanding a container into numbers: the long one comes back as a
      blob and the short one as a list, and both have to inflate. }
    PutZipped('NIFTIData', 2000, 0.25);
    PutZipped('Small', 5, 1.5);

    PutMark('}');

    M.Position := 0;
    { Loaded first, and into a variable, because the two arguments of Ok are
      not evaluated in the order they are written -- so reading BJ.Error in
      the second one reported the error from before the parse, which is to
      say nothing at all, exactly when there was something to say. }
    Parsed := BJ.LoadFromStream(M);
    Ok(Parsed, 'the document parses: ' + BJ.Error);
    if BJ.Root = nil then Exit;

    Ok(BJ.Root.FindPath('name').AsString = 'box', 'a string survives');
    Ok(BJ.Root.FindPath('count').AsInt64 = 216000,
       'a little-endian int32 reads back');
    Ok(BJ.Root.FindPath('flag').AsBoolean, 'a marker-only true reads back');

    d := BJ.Root.FindPath('Dim');
    Ok((d <> nil) and (d.JSONType = jtArray) and (d.Count = 3) and
       (d.Items[2].AsInteger = 60), 'a small typed array became numbers');

    { The marker-only container is what would run off the end if its type
      were assumed to have a payload -- so everything after it reading
      correctly is the real assertion here. }
    d := BJ.Root.FindPath('Empties');
    Ok((d <> nil) and (d.JSONType = jtArray) and (d.Count = 4),
       'a container of markers has a count but no bytes');

    Ok(BJ.GetArray('Data', A), 'the big array comes back as a blob');
    Ok(A.Kind = akSingle, 'as the type it was written in');
    Ok(McxArrayCount(A) = 100, 'with every element');
    Ok(Abs(McxArrayValue(A, 0) - 0) < 1e-6, 'first value');
    Ok(Abs(McxArrayValue(A, 99) - 49.5) < 1e-6, 'last value');
    Ok(BJ.Root.FindPath('Data._ArrayType_').AsString = 'single',
       'and it is annotated the way a text file would be');

    d := BJ.Root.FindPath('Parser');
    Ok((d <> nil) and (d.JSONType = jtArray) and (d.Count = 2),
       'a typed container of strings has both of them');
    Ok((d <> nil) and (d.Count = 2) and
       (d.Items[1].AsString = 'https://neurojson.org/download/pybjdata'),
       'and each one reads back whole');

    { The compressed payload: bytes in a binary file rather than base64 in a
      string, which is what reading it as a string used to raise on. }
    Ok(BJ.GetArray('NIFTIData', A), 'a deflated blob inflates');
    Ok(A.Kind = akSingle, 'to the type the annotations gave it');
    Ok(McxArrayCount(A) = 2000, 'with every element');
    Ok(Abs(McxArrayValue(A, 1999) - 499.75) < 1e-4, 'and the values are right');

    Ok(BJ.GetArray('Small', A), 'so does one the reader expanded into numbers');
    Ok(McxArrayCount(A) = 5, 'with every element');
    Ok(Abs(McxArrayValue(A, 4) - 6.0) < 1e-6, 'and the values are right');
  finally
    BJ.Free;
    M.Free;
  end;
end;

procedure TestBJDataRefuses;
var
  M: TMemoryStream;
  BJ: TMcxBJData;
  B: Byte;
begin
  WriteLn('BJData, malformed');
  M := TMemoryStream.Create;
  BJ := TMcxBJData.Create;
  try
    { An unknown marker, and then nothing. }
    B := Ord('Q');
    M.Write(B, 1);
    M.Position := 0;
    Ok(not BJ.LoadFromStream(M), 'an unknown marker is refused');
    Ok(BJ.Error <> '', 'and says why: ' + BJ.Error);

    { A container that claims more elements than it carries. }
    M.Clear;
    B := Ord('['); M.Write(B, 1);
    B := Ord('$'); M.Write(B, 1);
    B := Ord('d'); M.Write(B, 1);
    B := Ord('#'); M.Write(B, 1);
    B := Ord('U'); M.Write(B, 1);
    B := 200;      M.Write(B, 1);
    M.Position := 0;
    Ok(not BJ.LoadFromStream(M), 'a truncated container is refused');
  finally
    BJ.Free;
    M.Free;
  end;
end;


{ ---------------------------------------------------------------- mcxmesh -- }

{ A cube cut into six tetrahedra: the smallest mesh with a real surface, and
  one whose answer is known without computing it.  Eight nodes, six elements,
  and a surface of twelve triangles -- two per face of the cube -- because
  every interior face is shared by two tetrahedra and every boundary face by
  one.  Written here rather than read from mmc's examples so the test travels
  with the repository. }
procedure TestMesh;
var
  Dir: string;
  F: TStringList;
  M: TMcxTetMesh;
  Faces: TMcxFaces;
  Lo, Hi: TMcxNode;
begin
  WriteLn('tetrahedral mesh');
  Dir := GetTempDir(False);
  F := TStringList.Create;
  try
    F.Add('1	8');
    F.Add('1	  0.0	  0.0	  0.0');
    F.Add('2	 10.0	  0.0	  0.0');
    F.Add('3	 10.0	 10.0	  0.0');
    F.Add('4	  0.0	 10.0	  0.0');
    F.Add('5	  0.0	  0.0	 10.0');
    F.Add('6	 10.0	  0.0	 10.0');
    F.Add('7	 10.0	 10.0	 10.0');
    F.Add('8	  0.0	 10.0	 10.0');
    F.SaveToFile(Dir + 'node_mcxtest.dat');

    F.Clear;
    F.Add('1	6');
    F.Add('1	1	2	8	4	1');
    F.Add('2	1	2	6	8	1');
    F.Add('3	2	3	4	8	1');
    F.Add('4	2	3	7	8	1');
    F.Add('5	2	6	7	8	1');
    F.Add('6	1	5	6	8	1');
    F.SaveToFile(Dir + 'elem_mcxtest.dat');
  finally
    F.Free;
  end;

  M := TMcxTetMesh.Create;
  try
    Ok(M.LoadFromDir(Dir, 'mcxtest'), 'the mesh loads: ' + M.Error);
    Ok(M.NodeCount = 8, 'eight nodes');
    Ok(M.ElemCount = 6, 'six elements');
    M.Bounds(Lo, Hi);
    Ok((Lo.x = 0) and (Hi.x = 10) and (Hi.z = 10), 'the bounds are the cube');
    Faces := M.Surface;
    Ok(Length(Faces) = 12,
       Format('twelve surface triangles, two a face (got %d)', [Length(Faces)]));
    Ok(M.LoadFromDir(Dir, 'nosuchmesh') = False, 'a missing mesh is refused');
    Ok(M.Error <> '', 'and says why');
  finally
    M.Free;
    DeleteFile(Dir + 'node_mcxtest.dat');
    DeleteFile(Dir + 'elem_mcxtest.dat');
  end;
end;

var
  Dir: string;
begin
  if ParamCount >= 1 then
    Dir := ParamStr(1)
  else
    Dir := '../example';

  TestPaths;
  TestTypePreservation;
  TestCompare;
  TestPlainText;
  TestProgress;
  TestProgressStream;
  TestDevices;
  TestArgs;
  {$IFDEF UNIX}
  TestRunner;
  {$ENDIF}
  TestJData;
  TestBJData;
  TestBJDataRefuses;
  TestMesh;
  TestCorpus(Dir);

  WriteLn;
  WriteLn(Format('%d checks, %d failures', [Checks, Failures]));
  if Failures > 0 then Halt(1);
end.
