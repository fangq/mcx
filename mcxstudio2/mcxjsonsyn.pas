unit mcxjsonsyn;

{ mcxstudio2 - syntax colouring for the JSON pane.

  SynEdit ships highlighters for a dozen languages and not for JSON.  The
  JavaScript one is the usual substitute and is nearly right -- JSON is a
  subset of JavaScript's literals -- but it cannot tell a key from a string
  value, because in JavaScript there is no difference.  In JSON there is:
  half of reading a simulation file is following the keys down to the one you
  want, and a pane where "Photons" and "benchmark1" are the same colour makes
  you read the punctuation to tell them apart.

  So: a scanner of its own, six token kinds.  It is short because JSON is
  short -- no comments, no operators, no identifiers, and a string cannot
  span a line, which is what lets a key be recognised by looking ahead for
  the colon rather than by carrying state between lines.

  Written over a PChar with a #0 at the end, and signalling end-of-line
  through the token kind rather than through the position, because that is
  the shape SynEdit's own highlighters have and the shape its painter
  expects.  Testing the position instead ends the line while the last token
  is still the current one, so every line loses its final brace or comma --
  and indexing a string one past its end is a range check error waiting for
  whichever build has them on. }

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Graphics, SynEditHighlighter, SynEditTypes;

type
  { Prefixed mj rather than jt: fpjson's TJSONtype already owns jtString and
    jtNumber, and a unit that uses both would get whichever was listed
    second. }
  TMcxJsonToken = (mjNone, mjSpace, mjKey, mjString, mjNumber, mjLiteral,
                   mjSymbol, mjError);

  TMcxJsonSyn = class(TSynCustomHighlighter)
  private
    { The line is kept as well as pointed at: the PChar is into its
      storage. }
    FLineStr: string;
    FLine: PChar;
    FLineNumber: Integer;
    FRun: Integer;          { 0-based, as SynEdit's own scanners are }
    FTokenPos: Integer;
    FToken: TMcxJsonToken;
    FAttri: array[TMcxJsonToken] of TSynHighlighterAttributes;
    function  LooksLikeKey: Boolean;
    procedure ScanString;
    procedure ScanNumber;
    procedure ScanWord;
  public
    constructor Create(AOwner: TComponent); override;
    procedure SetLine(const NewValue: string; LineNumber: Integer); override;
    procedure Next; override;
    function  GetEol: Boolean; override;
    function  GetToken: string; override;
    procedure GetTokenEx(out TokenStart: PChar; out TokenLength: Integer);
      override;
    function  GetTokenAttribute: TSynHighlighterAttributes; override;
    function  GetTokenKind: Integer; override;
    function  GetTokenPos: Integer; override;
    function  GetDefaultAttribute(Index: Integer): TSynHighlighterAttributes;
      override;
    class function GetLanguageName: string; override;
    { Repaints itself in colours mixed from the three the window uses, so the
      pane belongs to whichever theme is on rather than to a palette of its
      own.  ABack is what it will be drawn on. }
    procedure Recolour(ABack, AInk, AAccent: TColor);
    property KeyAttri: TSynHighlighterAttributes read FAttri[mjKey];
    property StringAttri: TSynHighlighterAttributes read FAttri[mjString];
    property NumberAttri: TSynHighlighterAttributes read FAttri[mjNumber];
    property LiteralAttri: TSynHighlighterAttributes read FAttri[mjLiteral];
    property SymbolAttri: TSynHighlighterAttributes read FAttri[mjSymbol];
  end;

implementation

{ A colour APercent of the way from A to B.  A copy rather than a use of
  mcxtheme's, so the highlighter does not depend on the rest of the program
  and can be tested on its own. }
function Mix(A, B: TColor; APercent: Integer): TColor;
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

constructor TMcxJsonSyn.Create(AOwner: TComponent);
const
  Names: array[TMcxJsonToken] of string =
    ('', 'Space', 'Key', 'String', 'Number', 'Literal', 'Symbol', 'Error');
var
  T: TMcxJsonToken;
begin
  inherited Create(AOwner);
  for T := mjSpace to mjError do
  begin
    FAttri[T] := TSynHighlighterAttributes.Create(Names[T], Names[T]);
    AddAttribute(FAttri[T]);
  end;
  FAttri[mjKey].Style := [fsBold];
  SetAttributesOnChange(@DefHighlightChange);
  Recolour(clWindow, clWindowText, clHighlight);
end;

class function TMcxJsonSyn.GetLanguageName: string;
begin
  Result := 'JSON';
end;

procedure TMcxJsonSyn.Recolour(ABack, AInk, AAccent: TColor);
begin
  FAttri[mjSpace].Foreground := AInk;
  FAttri[mjKey].Foreground := AAccent;
  { A value is the ink itself; the punctuation between values is faded most,
    because it is the part you never read on purpose. }
  FAttri[mjString].Foreground := Mix(AInk, AAccent, 55);
  FAttri[mjNumber].Foreground := Mix(AInk, ABack, 15);
  FAttri[mjLiteral].Foreground := Mix(AInk, AAccent, 75);
  FAttri[mjLiteral].Style := [fsItalic];
  FAttri[mjSymbol].Foreground := Mix(AInk, ABack, 45);
  FAttri[mjError].Foreground := clRed;
end;

procedure TMcxJsonSyn.SetLine(const NewValue: string; LineNumber: Integer);
begin
  inherited;
  FLineStr := NewValue;
  FLine := PChar(FLineStr);
  FLineNumber := LineNumber;
  FRun := 0;
  Next;
end;

function TMcxJsonSyn.GetEol: Boolean;
begin
  Result := FToken = mjNone;
end;

{ A string is a key when the next thing that is not a space is a colon.  No
  state has to survive the line for this, because JSON has no string that
  can contain a newline. }
function TMcxJsonSyn.LooksLikeKey: Boolean;
var
  i: Integer;
begin
  i := FRun;
  while FLine[i] in [' ', #9] do Inc(i);
  Result := FLine[i] = ':';
end;

procedure TMcxJsonSyn.ScanString;
begin
  Inc(FRun);
  while FLine[FRun] <> #0 do
  begin
    if FLine[FRun] = '\' then
    begin
      { Whatever follows is escaped -- but not the terminator, or the scan
        would run off the end of the buffer. }
      if FLine[FRun + 1] = #0 then Break;
      Inc(FRun);
    end
    else if FLine[FRun] = '"' then
    begin
      Inc(FRun);
      if LooksLikeKey then FToken := mjKey else FToken := mjString;
      Exit;
    end;
    Inc(FRun);
  end;
  { Ran off the end without closing: JSON has no multi-line string, so this
    is broken rather than continued. }
  FToken := mjError;
end;

procedure TMcxJsonSyn.ScanNumber;
begin
  if FLine[FRun] = '-' then Inc(FRun);
  while FLine[FRun] in ['0'..'9', '.'] do Inc(FRun);
  if FLine[FRun] in ['e', 'E'] then
  begin
    Inc(FRun);
    if FLine[FRun] in ['+', '-'] then Inc(FRun);
    while FLine[FRun] in ['0'..'9'] do Inc(FRun);
  end;
  FToken := mjNumber;
end;

procedure TMcxJsonSyn.ScanWord;
var
  S: string;
begin
  while FLine[FRun] in ['a'..'z', 'A'..'Z'] do Inc(FRun);
  SetString(S, FLine + FTokenPos, FRun - FTokenPos);
  if (S = 'true') or (S = 'false') or (S = 'null') then FToken := mjLiteral
  else FToken := mjError;
end;

procedure TMcxJsonSyn.Next;
begin
  FTokenPos := FRun;
  case FLine[FRun] of
    #0: FToken := mjNone;
    ' ', #9:
      begin
        while FLine[FRun] in [' ', #9] do Inc(FRun);
        FToken := mjSpace;
      end;
    '"': ScanString;
    '-', '0'..'9': ScanNumber;
    'a'..'z', 'A'..'Z': ScanWord;
    '{', '}', '[', ']', ':', ',':
      begin
        Inc(FRun);
        FToken := mjSymbol;
      end;
  else
    Inc(FRun);
    FToken := mjError;
  end;
end;

function TMcxJsonSyn.GetToken: string;
begin
  SetString(Result, FLine + FTokenPos, FRun - FTokenPos);
end;

procedure TMcxJsonSyn.GetTokenEx(out TokenStart: PChar;
  out TokenLength: Integer);
begin
  TokenLength := FRun - FTokenPos;
  TokenStart := FLine + FTokenPos;
end;

function TMcxJsonSyn.GetTokenAttribute: TSynHighlighterAttributes;
begin
  if FToken = mjNone then Result := nil else Result := FAttri[FToken];
end;

function TMcxJsonSyn.GetTokenKind: Integer;
begin
  Result := Ord(FToken);
end;

function TMcxJsonSyn.GetTokenPos: Integer;
begin
  Result := FTokenPos;
end;

function TMcxJsonSyn.GetDefaultAttribute(Index: Integer):
  TSynHighlighterAttributes;
begin
  case Index of
    SYN_ATTR_STRING: Result := FAttri[mjString];
    SYN_ATTR_WHITESPACE: Result := FAttri[mjSpace];
    SYN_ATTR_NUMBER: Result := FAttri[mjNumber];
    SYN_ATTR_KEYWORD: Result := FAttri[mjLiteral];
  else
    Result := nil;
  end;
end;

end.
