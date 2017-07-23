unit mcxoutput;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, SynEdit, synhighlighterunixshellscript,
  LResources, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls, Menus,
  AsyncProcess;

type

  { TfmOutput }

  TfmOutput = class(TForm)
    Button1: TButton;
    btSendCmd: TButton;
    edCmdInput: TLabeledEdit;
    miClearLog: TMenuItem;
    mmOutput: TSynEdit;
    Panel1: TPanel;
    Panel2: TPanel;
    PopupMenu1: TPopupMenu;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    procedure btSendCmdClick(Sender: TObject);
    procedure edCmdInputKeyPress(Sender: TObject; var Key: char);
    procedure miClearLogClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    pMCX: TAsyncProcess;
  end;

var
  fmOutput: TfmOutput;

implementation

{ TfmOutput }

procedure TfmOutput.miClearLogClick(Sender: TObject);
begin
  mmOutput.Lines.Clear;
end;

procedure TfmOutput.btSendCmdClick(Sender: TObject);
var
   cmd: string;
begin
  cmd:=edCmdInput.Text;
  if(Length(cmd)=0) or (pMCX=nil) then exit;
  pMCX.Input.Write(cmd[1], Length(cmd));
  mmOutput.Lines.Add('"User input:" '+cmd);
  edCmdInput.Text:='';
end;

procedure TfmOutput.edCmdInputKeyPress(Sender: TObject; var Key: char);
begin
  if Key = #13 then begin
    btSendCmdClick(Sender);
  end;
end;

initialization
  {$I mcxoutput.lrs}

end.

