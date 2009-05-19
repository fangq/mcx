unit mcxgui;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, IniPropStorage, Menus, ComCtrls, ExtCtrls, Spin, EditBtn, Buttons;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    btRun: TBitBtn;
    btQuit: TBitBtn;
    btStop: TBitBtn;
    ckReflect: TCheckBox;
    ckSaveData: TCheckBox;
    ckNormalize: TCheckBox;
    edThread: TComboBox;
    edMove: TEdit;
    edSession: TEdit;
    edT0: TEdit;
    edT1: TEdit;
    edConfigFile: TFileNameEdit;
    ImageList1: TImageList;
    IniPropStorage1: TIniPropStorage;
    Label1: TLabel;
    Label10: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    lvJobs: TListView;
    MainMenu1: TMainMenu;
    mmOutput: TMemo;
    MenuItem1: TMenuItem;
    MenuItem10: TMenuItem;
    MenuItem11: TMenuItem;
    MenuItem12: TMenuItem;
    MenuItem13: TMenuItem;
    MenuItem14: TMenuItem;
    MenuItem15: TMenuItem;
    MenuItem16: TMenuItem;
    MenuItem2: TMenuItem;
    MenuItem3: TMenuItem;
    MenuItem4: TMenuItem;
    MenuItem5: TMenuItem;
    MenuItem6: TMenuItem;
    MenuItem7: TMenuItem;
    MenuItem8: TMenuItem;
    MenuItem9: TMenuItem;
    plSetting: TPanel;
    grArray: TRadioGroup;
    edRespin: TSpinEdit;
    edGate: TSpinEdit;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    sbInfo: TStatusBar;
    ToolBar1: TToolBar;
    ToolButton1: TToolButton;
    ToolButton2: TToolButton;
    ToolButton3: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    procedure MenuItem13Click(Sender: TObject);
    procedure MenuItem8Click(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
  end; 

var
  fmMCX: TfmMCX;

implementation

{ TfmMCX }

procedure TfmMCX.MenuItem8Click(Sender: TObject);
begin
  Close;
end;

procedure TfmMCX.MenuItem13Click(Sender: TObject);
begin

end;

initialization
  {$I mcxgui.lrs}

end.

