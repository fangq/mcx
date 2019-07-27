unit mcxconfig;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, EditBtn, Grids, ExtCtrls;

type

  { TfmConfig }

  TfmConfig = class(TForm)
    btCancel: TButton;
    btOK: TButton;
    edWorkPath: TDirectoryEdit;
    edRemoteOutputPath: TEdit;
    edSCPPath: TFileNameEdit;
    edRemotePath: TEdit;
    edSSHPath: TFileNameEdit;
    edWorkPath2: TFileNameEdit;
    GroupBox1: TGroupBox;
    GroupBox3: TGroupBox;
    GroupBox4: TGroupBox;
    GroupBox5: TGroupBox;
    GroupBox6: TGroupBox;
    GroupBox7: TGroupBox;
    Panel1: TPanel;
    dlBrowsePath: TSelectDirectoryDialog;
    edLocalPath: TStringGrid;
    procedure edLocalPathButtonClick(Sender: TObject; aCol, aRow: Integer);
    procedure edWorkPathButtonClick(Sender: TObject);
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

initialization

{$I mcxconfig.lrs}

end.

