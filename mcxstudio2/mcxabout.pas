unit mcxabout;

{ mcxstudio2 - the About box, and the F1 help window.

  Both are built here and both are built in code: they have no settings on
  them, so there is nothing for the designer to hold, and a form file would
  only be one more thing to keep in step with the DPI sweep.

  About is not only a credit.  It is the one place that says which binaries
  this copy actually found, what the GPU driver calls itself and what display
  scale is in force -- the three things a bug report needs and the three
  things nobody can be expected to know off-hand. }

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, StdCtrls, ExtCtrls, Graphics, Buttons,
  Clipbrd, mcxdpi, mcxicons;

const
  { Studio's own version, which is not mcx's: a front end is released on its
    own clock, and a report saying only "mcx 2025" cannot be placed. }
  McxStudioVersion = '2.0-alpha';

{ ADetails is the machine-specific part -- backend paths, the GL renderer --
  which the caller collects because only it knows them. }
procedure McxShowAbout(AOwner: TComponent; const ADetails: string);

{ A help window rather than a message box: help is read, and a message box
  cannot be left open beside the setting it describes. }
procedure McxShowHelp(AOwner: TComponent; const ATitle, AText: string);

implementation

const
  Blurb =
    'A graphical front end for MCX, MCX-CL and MMC -- Monte Carlo photon ' +
    'transport in voxels and in meshes.' + LineEnding + LineEnding +
    'MCX Studio is free software under the GNU General Public License, ' +
    'version 3 or later, and comes with no warranty.' + LineEnding +
    LineEnding +
    'Copyright (c) 2009-2026 Qianqian Fang and the MCX contributors.' +
    LineEnding + 'https://mcx.space';

type
  { Carries the text the copy button copies.  A class only because an event
    handler has to be a method; it owns nothing and dies with the dialog. }
  TMcxCopier = class
    Text: string;
    procedure Click(Sender: TObject);
  end;

procedure TMcxCopier.Click(Sender: TObject);
begin
  Clipboard.AsText := Text;
end;

{ A form that is as tall as its contents and no taller.  Built the same way
  for both windows, because the only difference between them is the picture. }
function MakeWindow(AOwner: TComponent; const ACaption: string;
  AWidth: Integer): TForm;
begin
  Result := TForm.Create(AOwner);
  Result.Caption := ACaption;
  Result.Position := poMainFormCenter;
  Result.BorderStyle := bsDialog;
  { Raw 96-dpi numbers: the form scaler installed at startup is what turns
    these into pixels, and a size already scaled would be scaled twice. }
  Result.ClientWidth := AWidth;
  Result.ClientHeight := 260;
end;

{ The strip of buttons along the bottom.  alBottom before the text, so the
  text gets what is left rather than the other way round. }
function MakeButtons(AForm: TForm): TPanel;
begin
  Result := TPanel.Create(AForm);
  Result.Parent := AForm;
  Result.Align := alBottom;
  Result.Height := 40;
  Result.BevelOuter := bvNone;
  Result.Caption := '';
end;

{ AOrder counts from the right edge.  Aligned rather than positioned: the
  panel has no width yet when the buttons are added, so anything measured
  from its right edge here would be measured from zero. }
function AddButton(APanel: TPanel; const ACaption: string;
  AModal: TModalResult; AOrder: Integer): TBitBtn;
begin
  Result := TBitBtn.Create(APanel);
  Result.Parent := APanel;
  { alRight siblings are ordered by the Left they have when they are aligned,
    rightmost last -- so a larger Left is further right. }
  Result.Left := 1000 - AOrder * 100;
  Result.Align := alRight;
  Result.Width := 100;
  Result.BorderSpacing.Around := 6;
  Result.Caption := ACaption;
  Result.ModalResult := AModal;
  { Escape closes what Escape ought to close. }
  if AModal = mrOK then
  begin
    Result.Cancel := True;
    Result.Default := True;
  end;
end;

procedure McxShowAbout(AOwner: TComponent; const ADetails: string);
var
  F: TForm;
  Logo: TPortableNetworkGraphic;
  Img: TImage;
  Head: TLabel;
  Body: TLabel;
  Info: TMemo;
  Bar: TPanel;
  Copy_: TBitBtn;
  Copier: TMcxCopier;
  Full: string;
begin
  F := MakeWindow(AOwner, 'About MCX Studio', 460);
  F.ClientHeight := 420;
  try
    Bar := MakeButtons(F);
    AddButton(Bar, 'Close', mrOK, 0);
    Copier := TMcxCopier.Create;
    Copy_ := AddButton(Bar, 'Copy details', mrNone, 1);
    Copy_.OnClick := @Copier.Click;
    Copy_.Hint := 'Puts the block below on the clipboard, for a bug report';
    Copy_.ShowHint := True;

    Img := TImage.Create(F);
    Img.Parent := F;
    Img.Align := alTop;
    Img.Height := 96;
    Img.Center := True;
    Img.Proportional := True;
    Img.Stretch := True;
    Logo := McxAppLogo;
    if Logo <> nil then
    try
      Img.Picture.Assign(Logo);
    finally
      Logo.Free;
    end;

    Head := TLabel.Create(F);
    Head.Parent := F;
    Head.Top := 100;
    Head.Align := alTop;
    Head.Alignment := taCenter;
    Head.Font.Height := -McxScale96(15);
    Head.Font.Style := [fsBold];
    Head.Caption := 'MCX Studio ' + McxStudioVersion;
    Head.BorderSpacing.Top := 8;

    Body := TLabel.Create(F);
    Body.Parent := F;
    Body.Top := 200;
    Body.Align := alTop;
    Body.Alignment := taCenter;
    Body.WordWrap := True;
    Body.BorderSpacing.Around := 12;
    Body.Caption := Blurb;

    Info := TMemo.Create(F);
    Info.Parent := F;
    Info.Align := alClient;
    Info.BorderSpacing.Around := 12;
    Info.ReadOnly := True;
    Info.ScrollBars := ssAutoVertical;
    Info.Font.Pitch := fpFixed;
    Info.Font.Name := McxDefaultFontName;
    Full := Format('MCX Studio %s'#10'%s %s / %s'#10'Display scale: %d dpi'#10,
      [McxStudioVersion, {$I %FPCTARGETOS%}, {$I %FPCTARGETCPU%},
       {$I %FPCVERSION%}, McxDesiredPPI]) + ADetails;
    Info.Lines.Text := Full;
    Copier.Text := Full;

    try
      F.ShowModal;
    finally
      Copier.Free;
    end;
  finally
    F.Free;
  end;
end;

procedure McxShowHelp(AOwner: TComponent; const ATitle, AText: string);
var
  F: TForm;
  Body: TMemo;
  Bar: TPanel;
begin
  F := MakeWindow(AOwner, ATitle, 480);
  try
    F.BorderStyle := bsSizeable;
    Bar := MakeButtons(F);
    AddButton(Bar, 'Close', mrOK, 0);

    Body := TMemo.Create(F);
    Body.Parent := F;
    Body.Align := alClient;
    Body.BorderSpacing.Around := 12;
    Body.ReadOnly := True;
    Body.WordWrap := True;
    Body.ScrollBars := ssAutoVertical;
    Body.BorderStyle := bsNone;
    Body.Lines.Text := AText;
    F.ShowModal;
  finally
    F.Free;
  end;
end;

end.
