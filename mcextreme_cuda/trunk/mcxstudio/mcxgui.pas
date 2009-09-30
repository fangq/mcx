unit mcxgui;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, process, FileUtil, LResources, Forms, Controls, Graphics,
  Dialogs, StdCtrls, IniPropStorage, Menus, ComCtrls, ExtCtrls, Spin, EditBtn,
  Buttons, ActnList, lcltype;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    doClearLog: TAction;
    doListGPU: TAction;
    doWeb: TAction;
    doAbout: TAction;
    doHelp: TAction;
    doRunAll: TAction;
    doStop: TAction;
    doRun: TAction;
    doVerify: TAction;
    doDeleteItem: TAction;
    doAddItem: TAction;
    doOpen: TAction;
    doSave: TAction;
    doInitEnv: TAction;
    ActionList1: TActionList;
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
    MenuItem17: TMenuItem;
    MenuItem18: TMenuItem;
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
    plSettings: TPanel;
    grArray: TRadioGroup;
    edRespin: TSpinEdit;
    edGate: TSpinEdit;
    Process1: TProcess;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    sbInfo: TStatusBar;
    ToolBar1: TToolBar;
    ToolButton1: TToolButton;
    ToolButton10: TToolButton;
    ToolButton11: TToolButton;
    ToolButton12: TToolButton;
    ToolButton13: TToolButton;
    ToolButton14: TToolButton;
    ToolButton15: TToolButton;
    ToolButton16: TToolButton;
    ToolButton17: TToolButton;
    ToolButton18: TToolButton;
    ToolButton19: TToolButton;
    ToolButton2: TToolButton;
    ToolButton3: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    ToolButton7: TToolButton;
    ToolButton8: TToolButton;
    ToolButton9: TToolButton;
    procedure doAddItemExecute(Sender: TObject);
    procedure doDeleteItemExecute(Sender: TObject);
    procedure doExitExecute(Sender: TObject);
    procedure doOpenExecute(Sender: TObject);
    procedure doVerifyExecute(Sender: TObject);
    procedure edConfigFileChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure lvJobsSelectItem(Sender: TObject; Item: TListItem;
      Selected: Boolean);
    procedure ToolButton18Click(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    MapList: TStringList;
    function CreateCmd:string;
    procedure VarifyInput;
    procedure AddLog(str:string);
    procedure ListToPanel2(node:TListItem);
    procedure PanelToList2(node:TListItem);
  end;

var
  fmMCX: TfmMCX;
  ProfileChanged: Boolean;

implementation

{ TfmMCX }
procedure TfmMCX.AddLog(str:string);
begin
    mmOutput.Lines.Add(str);
end;

procedure TfmMCX.edConfigFileChange(Sender: TObject);
begin
  if(Length(edSession.Text)=0) then
       edSession.Text:=ExtractFileName(edConfigFile.FileName);
end;

procedure TfmMCX.doExitExecute(Sender: TObject);
begin
  ShowMessage('here you go');
  Close;
end;

procedure TfmMCX.doAddItemExecute(Sender: TObject);
var
   node: TListItem;
   i:integer;
begin
   if(Trim(edSession.Caption)='') then
   begin
        AddLog('You have to supply a unique session name');
        edSession.SetFocus;
        exit;
   end;
   node:=lvJobs.Items.Add;
   for i:=0 to MapList.Count-1 do node.SubItems.Add('');
   node.Caption:=edSession.Caption;
   lvJobs.Selected:=node;
end;

procedure TfmMCX.doDeleteItemExecute(Sender: TObject);
begin
  if not (lvJobs.Selected = nil) then
  begin
        if(Application.MessageBox('The selected configuration will be deleted, are you sure?',
          'Confirm', MB_YESNOCANCEL)=IDYES) then
            exit;
        lvJobs.Items.Delete(lvJobs.Selected.Index);
  end;
end;

procedure TfmMCX.doOpenExecute(Sender: TObject);
begin

end;

procedure TfmMCX.doVerifyExecute(Sender: TObject);
begin
    VarifyInput;
end;

procedure TfmMCX.FormCreate(Sender: TObject);
begin
    MapList:=TStringList.Create();
    MapList.Add('Session');
    MapList.Add('InputFile');
    MapList.Add('ThreadNum');
    MapList.Add('MoveNum');
    MapList.Add('RespinNum');
    MapList.Add('ArrayOrder');
    MapList.Add('TStart');
    MapList.Add('TEnd');
    MapList.Add('GateNum');
    MapList.Add('DoReflect');
    MapList.Add('DoSave');
    MapList.Add('DoNormalize');
    ProfileChanged:=false;
end;

procedure TfmMCX.FormDestroy(Sender: TObject);
begin
    MapList.Free;
end;

procedure TfmMCX.lvJobsSelectItem(Sender: TObject; Item: TListItem;
  Selected: Boolean);
begin
     if(not Selected) then begin
          PanelToList2(Item);
          ListToPanel2(lvJobs.Selected);
     end
end;

procedure TfmMCX.ToolButton18Click(Sender: TObject);
begin

end;

procedure TfmMCX.VarifyInput;
var
    nthread, nmove: integer;
    t0,t1: extended;
begin

    btRun.Enabled:=false;

    if(Length(edConfigFile.FileName)=0) then
        raise Exception.Create('Config file must be specified');
    if(not FileExists(edConfigFile.FileName)) then
        raise Exception.Create('Config file does not exist, please check the path');
    try
        nthread:=StrToInt(edThread.Text);
        nmove:=StrToInt(edMove.Text);
        t0:=StrToFloat(edT0.Text);
        t1:=StrToFloat(edT1.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, move and time gate values');
    end;
    if(nthread<512) then
       AddLog('Warning: increase thread numbers to 1024 or above may boost the speed significantly');
    if(nthread>2048) then
       AddLog('Warning: you may need a high-end graphics card to use more threads');
    if(nmove>1e7) then
       AddLog('Warning: you can increase respin number to get more photons');
    if(t1<=t0) then
       raise Exception.Create('End time comes before the start time!');

    btRun.Enabled:=true;
end;

function TfmMCX.CreateCmd:string;
var
    nthread, nmove: integer;
    t0,t1: extended;
    cmd: string;
begin
//    cmd:='"'+Config.MCXExe+'" ';
    cmd:='mcextreme';
    if(Length(edSession.Text)>0) then
       cmd:=cmd+' -s "'+Trim(edSession.Text)+'" ';
    if(Length(edConfigFile.FileName)>0) then
       cmd:=cmd+' -f "'+Trim(edConfigFile.FileName)+'" ';
    try
        nthread:=StrToInt(edThread.Text);
        nmove:=StrToInt(edMove.Text);
        t0:=StrToFloat(edT0.Text);
        t1:=StrToFloat(edT1.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, move and time gate values');
    end;

    cmd:=cmd+Format(' -t %d -m %d -r %d -a %d ',[nthread,nmove,edRespin.Value,grArray.ItemIndex]);
    cmd:=cmd+Format(' -U %d -S %d -b %d ',[ckNormalize.Checked,ckSaveData.Checked,ckReflect.Checked]);

    Result:=cmd;
    AddLog('Command:');
    AddLog(cmd);
end;

procedure TfmMCX.PanelToList2(node:TListItem);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    iname: string;
    i,idx: integer;
begin
    if(node=nil) then exit;
    for i:=0 to plSettings.ControlCount-1 do
    begin
        try
        if(plSettings.Controls[i] is TSpinEdit) then begin
           se:=plSettings.Controls[i] as TSpinEdit;
           AddLog(se.Hint);
           idx:=MapList.IndexOf(se.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(se.Value);
           continue;
        end;
        if(plSettings.Controls[i] is TEdit) then begin
           ed:=plSettings.Controls[i] as TEdit;
           AddLog(ed.Hint);
           idx:=MapList.IndexOf(ed.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=ed.Text;
           continue;
        end;
        if(plSettings.Controls[i] is TRadioGroup) then begin
           gr:=plSettings.Controls[i] as TRadioGroup;
           AddLog(gr.Hint);
           idx:=MapList.IndexOf(gr.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
           continue;
        end;
        if(plSettings.Controls[i] is TComboBox) then begin
           cb:=plSettings.Controls[i] as TComboBox;
           AddLog(cb.Hint);
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=cb.Text;
           continue;
        end;
        if(plSettings.Controls[i] is TCheckBox) then begin
           ck:=plSettings.Controls[i] as TCheckBox;
           AddLog(ck.Hint);
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(Integer(ck.Checked));
           continue;
        end;

        finally
        end;
    end;
end;

procedure TfmMCX.ListToPanel2(node:TListItem);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    iname: string;
    i,idx: integer;
begin
    if(node=nil) then exit;
    for i:=0 to plSettings.ControlCount-1 do
    begin
        try

        if(plSettings.Controls[i] is TSpinEdit) then begin
           se:=plSettings.Controls[i] as TSpinEdit;
           idx:=MapList.IndexOf(se.Hint);
           if(idx>=0) then se.Value:=StrToInt(node.SubItems.Strings[idx]);
           continue;
        end;
        if(plSettings.Controls[i] is TEdit) then begin
           ed:=plSettings.Controls[i] as TEdit;
           idx:=MapList.IndexOf(ed.Hint);
           if(idx>=0) then ed.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSettings.Controls[i] is TRadioGroup) then begin
           gr:=plSettings.Controls[i] as TRadioGroup;
           AddLog(gr.Hint);
           idx:=MapList.IndexOf(gr.Hint);
           if(idx>=0) then gr.ItemIndex:=StrToInt(node.SubItems.Strings[idx]);
           continue;
        end;
        if(plSettings.Controls[i] is TComboBox) then begin
           cb:=plSettings.Controls[i] as TComboBox;
           AddLog(cb.Hint);
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then cb.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSettings.Controls[i] is TCheckBox) then begin
           ck:=plSettings.Controls[i] as TCheckBox;
           AddLog(ck.Hint);
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then ck.Checked:=(node.SubItems.Strings[i]='1');
           continue;
        end;
        finally
        end;
    end;
end;
initialization
  {$I mcxgui.lrs}

end.

