unit mcxnewsession;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls;

type

  { TfmNewSession }

  TfmNewSession = class(TForm)
    btCancel: TButton;
    btOK: TButton;
    edSession: TEdit;
    grProgram: TRadioGroup;
    Label4: TLabel;
    Panel1: TPanel;
    procedure btOKClick(Sender: TObject);
    procedure edSessionKeyPress(Sender: TObject; var Key: char);
    procedure FormShow(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  fmNewSession: TfmNewSession;

implementation

uses mcxgui;

{ TfmNewSession }

procedure TfmNewSession.btOKClick(Sender: TObject);
begin
    try
      if(Length(edSession.Text)=0) then
           raise Exception.Create('Session ID can not be empty');
      if (Tag=0) and (fmMCX.lvJobs.FindCaption(0,edSession.Text,true,true,true) <> nil) then
           raise Exception.Create('Session name already has aready existed!');
      ModalResult := mrOK;
    except
      On E : Exception do
        MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
    end;
end;

procedure TfmNewSession.edSessionKeyPress(Sender: TObject; var Key: char);
begin

end;

procedure TfmNewSession.FormShow(Sender: TObject);
begin
     edSession.SetFocus;
end;

initialization
  {$I mcxnewsession.lrs}

end.

