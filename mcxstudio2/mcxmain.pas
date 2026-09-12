{ mcxstudio2 - the main window.

  The form is a design-time .lfm and is meant to stay that way: controls are
  placed and tuned in the Lazarus designer, and mcxdoc's binding table connects
  each one to a path in the simulation document by component Name.  Only things
  whose count is not known until runtime are built here in code -- the media and
  detector tables, the shape list, the device list, and the source parameter
  slots, whose labels change with the source type.

  Sizes written in FormCreate are plain 96-dpi numbers; the startup sweep in
  mcxdpi scales the finished form.  Anything created after that sweep has to
  scale itself with McxScale96.  See the two-tier rule in mcxdpi. }
unit mcxmain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ComCtrls, ExtCtrls,
  ActnList, Menus, ImgList,
  mcxdpi, mcxicons;

type

  { TfmMain }

  TfmMain = class(TForm)
    ilIcons: TImageList;
    alMain: TActionList;
    tbMain: TToolBar;
    sbMain: TStatusBar;
    procedure FormCreate(Sender: TObject);
  private
    procedure BuildIcons;
  public
  end;

var
  fmMain: TfmMain;

implementation

{$R *.lfm}

{ TfmMain }

procedure TfmMain.FormCreate(Sender: TObject);
begin
  BuildIcons;
end;

{ The image list ships empty and is drawn here, once, at the display's scale
  and in the current theme's button-text colour, so the icons stay legible
  under a dark theme instead of being black on black. }
procedure TfmMain.BuildIcons;
var
  Size: Integer;
begin
  Size := McxScale96(16);
  ilIcons.Width := Size;
  ilIcons.Height := Size;
  McxBuildIconList(ilIcons, McxIconNames, clBtnText);
end;

end.
