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
  SysUtils, Classes, fpjson, mcxdoc;

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
  TestCorpus(Dir);

  WriteLn;
  WriteLn(Format('%d checks, %d failures', [Checks, Failures]));
  if Failures > 0 then Halt(1);
end.
