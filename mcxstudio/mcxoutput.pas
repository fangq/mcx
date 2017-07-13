unit mcxoutput;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, SynEdit, synhighlighterunixshellscript,
  LResources, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls, Menus;

type

  { TfmOutput }

  TfmOutput = class(TForm)
    Button1: TButton;
    miClearLog: TMenuItem;
    mmOutput: TSynEdit;
    Panel1: TPanel;
    PopupMenu1: TPopupMenu;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    procedure miClearLogClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  fmOutput: TfmOutput;

implementation

{ TfmOutput }

procedure TfmOutput.miClearLogClick(Sender: TObject);
begin
  mmOutput.Lines.Clear;
end;

initialization
  {$I mcxoutput.lrs}

end.

