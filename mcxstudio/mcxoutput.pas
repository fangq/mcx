unit mcxoutput;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, SynEdit, synhighlighterunixshellscript,
  LResources, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls, Menus,
  AsyncProcess, LCLType, ClipBrd;

type

  { TfmOutput }

  TfmOutput = class(TForm)
    Button1: TButton;
    btSendCmd: TButton;
    edCmdInput: TComboBox;
    Label1: TLabel;
    miCopy: TMenuItem;
    miClearLog: TMenuItem;
    mmOutput: TSynEdit;
    Panel1: TPanel;
    Panel2: TPanel;
    PopupMenu1: TPopupMenu;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    procedure btSendCmdClick(Sender: TObject);
    procedure edCmdInputKeyPress(Sender: TObject; var Key: char);
    procedure miClearLogClick(Sender: TObject);
    procedure miCopyClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    pProc: TAsyncProcess;
  end;

var
  fmOutput: TfmOutput;

implementation

{ TfmOutput }

procedure TfmOutput.miClearLogClick(Sender: TObject);
begin
  mmOutput.Lines.Clear;
end;

procedure TfmOutput.miCopyClick(Sender: TObject);
begin
   Clipboard.AsText:=mmOutput.SelText;
end;

procedure TfmOutput.btSendCmdClick(Sender: TObject);
var
   cmd: string;
begin
  cmd:=edCmdInput.Text+#10;
  if(Length(cmd)=0) or (pProc=nil) or (not pProc.Running) then exit;
  pProc.Input.Write(cmd[1], Length(cmd));
  mmOutput.Lines.Add('"User input:" '+cmd);
  if(edCmdInput.Items.IndexOf(cmd)>0) then begin
      edCmdInput.Items.Insert(0,cmd);
  end;
  edCmdInput.Text:='';
end;

procedure TfmOutput.edCmdInputKeyPress(Sender: TObject; var Key: char);
begin
  if (Key = #13) or (Key = #10) then begin
    btSendCmdClick(Sender);
//  end else if (Key = #38 {VK_UP}) or (Key= #40{VK_DOWN}) then begin
//     edCmdInput.DroppedDown:=true;
  end;
end;

initialization
  {$I mcxoutput.lrs}

end.

