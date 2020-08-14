unit mcxconfig;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, EditBtn, Grids, ExtCtrls, JSONPropStorage;

type

  { TfmConfig }

  TfmConfig = class(TForm)
    btCancel: TButton;
    btOK: TButton;
    cbHost: TComboBox;
    ckUseManualPath: TCheckBox;
    edPort: TEdit;
    edRemoteOutputPath: TEdit;
    edRemotePath: TEdit;
    edSCPPath: TFileNameEdit;
    edSSHPath: TFileNameEdit;
    edUserName: TEdit;
    edWorkPath: TDirectoryEdit;
    edWorkPath2: TFileNameEdit;
    grConfig: TGroupBox;
    GroupBox2: TGroupBox;
    GroupBox3: TGroupBox;
    GroupBox4: TGroupBox;
    GroupBox5: TGroupBox;
    GroupBox6: TGroupBox;
    GroupBox7: TGroupBox;
    GroupBox8: TGroupBox;
    jsonConfig: TJSONPropStorage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Panel1: TPanel;
    dlBrowsePath: TSelectDirectoryDialog;
    edLocalPath: TStringGrid;
    procedure btCancelClick(Sender: TObject);
    procedure btOKClick(Sender: TObject);
    procedure ckUseManualPathChange(Sender: TObject);
    procedure edLocalPathButtonClick(Sender: TObject; aCol, aRow: Integer);
    procedure edWorkPathButtonClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormShow(Sender: TObject);
    procedure jsonConfigRestoreProperties(Sender: TObject);
    procedure jsonConfigSaveProperties(Sender: TObject);
  private

  public

  end;

var
  fmConfig: TfmConfig;

implementation

{ TfmConfig }

procedure TfmConfig.edWorkPathButtonClick(Sender: TObject);
begin
end;

procedure TfmConfig.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
   CloseAction:=caHide;
end;

procedure TfmConfig.FormShow(Sender: TObject);
begin

end;

procedure TfmConfig.jsonConfigRestoreProperties(Sender: TObject);
begin
    edLocalPath.Cols[0].Delimiter:=PathSeparator;
    if Length(edLocalPath.Hint)>0 then begin
       edLocalPath.Cols[0].DelimitedText:=edLocalPath.Hint;
    end else begin
       edLocalPath.Hint:=edLocalPath.Cols[0].DelimitedText;
    end;
end;

procedure TfmConfig.jsonConfigSaveProperties(Sender: TObject);
begin
    edLocalPath.Cols[0].Delimiter:=PathSeparator;
    edLocalPath.Hint:=edLocalPath.Cols[0].DelimitedText;
end;

procedure TfmConfig.edLocalPathButtonClick(Sender: TObject; aCol, aRow: Integer);
var
    path: string;
begin
    path:=edLocalPath.Cells[aCol,aRow];
    if not path.IsEmpty and DirectoryExists(path) then
        dlBrowsePath.InitialDir:=path;
    if(dlBrowsePath.Execute) then begin
        edLocalPath.Cells[aCol,aRow]:=dlBrowsePath.FileName;
    end;
end;

procedure TfmConfig.ckUseManualPathChange(Sender: TObject);
begin
    grConfig.Enabled:=ckUseManualPath.Checked;
end;

procedure TfmConfig.btOKClick(Sender: TObject);
begin
  Close;
end;

procedure TfmConfig.btCancelClick(Sender: TObject);
begin
  Close;
end;

initialization

{$I mcxconfig.lrs}

end.

