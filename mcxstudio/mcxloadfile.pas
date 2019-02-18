unit mcxloadfile;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, StdCtrls, Spin, EditBtn;

type

  { TfmDataFile }

  TfmDataFile = class(TForm)
    btCancel: TButton;
    btOK: TButton;
    edDataFormat: TComboBox;
    edNx: TSpinEdit;
    edNy: TSpinEdit;
    edNz: TSpinEdit;
    edNt: TSpinEdit;
    edHeaderSize: TSpinEdit;
    edDataFile: TFileNameEdit;
    Label10: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    Panel1: TPanel;
  private

  public

  end;

var
  fmDataFile: TfmDataFile;

implementation

initialization
  {$I mcxloadfile.lrs}

end.

