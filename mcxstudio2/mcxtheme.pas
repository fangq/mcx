unit mcxtheme;

{ mcxstudio2 - the colour themes.

  The whole window is already mixed from three roles: a surface, the text on
  it, and one accent that is the only thing that ever means "this one".  A
  theme is therefore three colours and nothing else, and everything that was
  written against clBtnFace / clWindowText / clHighlight is written against
  McxBase / McxText / McxAccent instead.

  "System" is the default and is not a fourth palette -- it returns the
  theme colours themselves, so a desktop that changes its own look is
  followed, which is where the first MCX Studio's hard-coded greys failed. }

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Graphics, Forms;

type
  TMcxTheme = (mtSystem, mtLight, mtDark, mtOcean);

const
  McxThemeNames: array[TMcxTheme] of string =
    ('Follow the desktop', 'Light', 'Dark', 'Ocean');

{ The three roles.  Every colour in the window comes from these. }
function McxBase: TColor;
function McxText: TColor;
function McxAccent: TColor;

{ Near-white or near-black, whichever can be read on ABack.  Needed wherever
  something sits on the accent rather than on the surface: the accent is a
  fixed colour and the surface is not, so no single text colour works on
  both. }
function McxReadable(ABack: TColor): TColor;

{ A colour APercent of the way from A to B.  Here rather than in the form
  because the view needs it too. }
function McxBlend(A, B: TColor; APercent: Integer): TColor;

function McxCurrentTheme: TMcxTheme;
procedure McxSetTheme(ATheme: TMcxTheme);

{ The name as it is written in the settings file, and back.  A name rather
  than the ordinal, so inserting a theme does not silently move everyone
  who had picked a later one. }
function McxThemeToStr(ATheme: TMcxTheme): string;
function McxStrToTheme(const AName: string): TMcxTheme;

{ Remembered between sessions, in a one-line file of its own next to the
  saved pane layout.  Its own file because a layout written by an older build
  is discarded wholesale, and a colour choice should not go with it. }
procedure McxLoadTheme;
procedure McxSaveTheme;

implementation

type
  TMcxPalette = record
    Base, Text, Accent: TColor;
  end;

const
  { TColor literals are $00BBGGRR, so these read backwards: $00C86E1E is
    r=1e g=6e b=c8, a mid blue.  Kept as literals because the alternative is
    a constructor call, which a const array cannot have. }
  Palettes: array[TMcxTheme] of TMcxPalette = (
    { System -- never read; see McxBase. }
    (Base: clBtnFace;            Text: clWindowText;          Accent: clHighlight),
    (Base: TColor($00F2F1EF);    Text: TColor($00262220);     Accent: TColor($00C86E1E)),
    (Base: TColor($0034302D);    Text: TColor($00EAE6E4);     Accent: TColor($00E69646)),
    (Base: TColor($00382A1E);    Text: TColor($00EEE2D6);     Accent: TColor($00A0A000))
  );

var
  GTheme: TMcxTheme = mtSystem;

function McxCurrentTheme: TMcxTheme;
begin
  Result := GTheme;
end;

procedure McxSetTheme(ATheme: TMcxTheme);
begin
  GTheme := ATheme;
end;

function McxBase: TColor;
begin
  if GTheme = mtSystem then Result := ColorToRGB(clBtnFace)
  else Result := Palettes[GTheme].Base;
end;

function McxText: TColor;
begin
  if GTheme = mtSystem then Result := ColorToRGB(clWindowText)
  else Result := Palettes[GTheme].Text;
end;

function McxAccent: TColor;
begin
  if GTheme = mtSystem then Result := ColorToRGB(clHighlight)
  else Result := Palettes[GTheme].Accent;
end;

function McxReadable(ABack: TColor): TColor;
var
  r, g, b: Byte;
begin
  RedGreenBlue(ColorToRGB(ABack), r, g, b);
  { Rec. 601 luma, which is close enough for a two-way decision. }
  if (r * 299 + g * 587 + b * 114) div 1000 > 140 then
    Result := TColor($00201C1A)
  else
    Result := TColor($00F4F2F0);
end;

function McxBlend(A, B: TColor; APercent: Integer): TColor;
var
  ra, ga, ba, rb, gb, bb: Byte;
begin
  RedGreenBlue(ColorToRGB(A), ra, ga, ba);
  RedGreenBlue(ColorToRGB(B), rb, gb, bb);
  Result := RGBToColor(
    ra + (Integer(rb) - ra) * APercent div 100,
    ga + (Integer(gb) - ga) * APercent div 100,
    ba + (Integer(bb) - ba) * APercent div 100);
end;

function McxThemeToStr(ATheme: TMcxTheme): string;
const
  Keys: array[TMcxTheme] of string = ('system', 'light', 'dark', 'ocean');
begin
  Result := Keys[ATheme];
end;

function McxStrToTheme(const AName: string): TMcxTheme;
var
  T: TMcxTheme;
begin
  for T := Low(TMcxTheme) to High(TMcxTheme) do
    if SameText(McxThemeToStr(T), AName) then Exit(T);
  Result := mtSystem;
end;

function McxThemeFile: string;
begin
  Result := IncludeTrailingPathDelimiter(GetAppConfigDir(False)) + 'theme';
end;

procedure McxLoadTheme;
var
  F: TStringList;
begin
  if not FileExists(McxThemeFile) then Exit;
  F := TStringList.Create;
  try
    try
      F.LoadFromFile(McxThemeFile);
      if F.Count > 0 then GTheme := McxStrToTheme(Trim(F[0]));
    except
      { An unreadable preference is the default preference. }
    end;
  finally
    F.Free;
  end;
end;

procedure McxSaveTheme;
var
  F: TStringList;
begin
  F := TStringList.Create;
  try
    try
      ForceDirectories(ExtractFilePath(McxThemeFile));
      F.Add(McxThemeToStr(GTheme));
      F.SaveToFile(McxThemeFile);
    except
      { Not worth refusing to close over. }
    end;
  finally
    F.Free;
  end;
end;

end.
