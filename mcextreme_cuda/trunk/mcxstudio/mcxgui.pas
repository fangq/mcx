unit mcxgui;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, IniPropStorage, Menus, ComCtrls, ExtCtrls, Spin, EditBtn, Buttons,
  stringhashlist, ActnList, process, mcxabout;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    mcxdoWeb: TAction;
    mcxdoAbout: TAction;
    mcxdoHelp: TAction;
    mcxdoRunAll: TAction;
    mcxdoStop: TAction;
    mcxdoRun: TAction;
    mcxdoVerify: TAction;
    mcxdoDeleteItem: TAction;
    mcxdoAddItem: TAction;
    mcxdoOpen: TAction;
    mcxdoClose: TAction;
    mcxdoExit: TAction;
    mcxdoQuery: TAction;
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
    plSetting: TPanel;
    grArray: TRadioGroup;
    edRespin: TSpinEdit;
    edGate: TSpinEdit;
    pMCX: TProcess;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    sbInfo: TStatusBar;
    ToolBar1: TToolBar;
    ToolButton1: TToolButton;
    tbtRunAll: TToolButton;
    ToolButton11: TToolButton;
    ToolButton12: TToolButton;
    ToolButton13: TToolButton;
    ToolButton14: TToolButton;
    ToolButton15: TToolButton;
    ToolButton16: TToolButton;
    ToolButton17: TToolButton;
    ToolButton2: TToolButton;
    ToolButton3: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    tbtVerify: TToolButton;
    tbtRun: TToolButton;
    tbtStop: TToolButton;
    procedure lvJobsChange(Sender: TObject; Item: TListItem; Change: TItemChange
      );
    procedure lvJobsSelectItem(Sender: TObject; Item: TListItem;
      Selected: Boolean);
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure edConfigFileChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure mcxdoRunExecute(Sender: TObject);
    procedure mcxdoStopExecute(Sender: TObject);
    procedure mcxdoVerifyExecute(Sender: TObject);
    procedure MenuItem8Click(Sender: TObject);
    procedure ToolButton4Click(Sender: TObject);
    procedure tbtRunClick(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    MapList: TStringHashList;
    function CreateCmd:string;
    procedure VarifyInput;
    procedure AddLog(str:string);
    procedure ListToPanel;
    procedure PanelToList;

  end;

var
  fmMCX: TfmMCX;

implementation

{ TfmMCX }

procedure TfmMCX.MenuItem8Click(Sender: TObject);
begin
end;

procedure TfmMCX.ToolButton4Click(Sender: TObject);
begin

end;

procedure TfmMCX.tbtRunClick(Sender: TObject);
begin

end;

procedure TfmMCX.AddLog(str:string);
begin
    mmOutput.Lines.Add(str);
end;

procedure TfmMCX.edConfigFileChange(Sender: TObject);
begin
  if(Length(edSession.Text)=0) then
       edSession.Text:=ExtractFileName(edConfigFile.FileName);
end;

procedure TfmMCX.mcxdoExitExecute(Sender: TObject);
begin
  Close;
end;

procedure TfmMCX.mcxdoAboutExecute(Sender: TObject);
begin
    fmAbout:=TfmAbout.Create(Application);
    fmAbout.ShowModal;
    fmAbout.Free;
end;

procedure TfmMCX.mcxdoAddItemExecute(Sender: TObject);
var
   newnode: TListItem;
   sid: string;
begin
   sid:=InputBox('Session Name','Please use letters and numbers only','');
   if(Length(Trim(sid))>0) then
   begin
      newnode:=lvJobs.Items.Add;
      newnode.Caption:=sid;
      lvJobs.Selected:= newnode;
   end;
end;

procedure TfmMCX.mcxdoDeleteItemExecute(Sender: TObject);
var
   id: integer;
begin
   if(lvJobs.Selected <> nil) then
   begin
      id:=lvJobs.Selected.Index;
      if(not id=0) then
      begin
         lvJobs.Items.Delete(id);
         lvJobs.Selected:=lvJobs.Items.Item[id-1];
      end;
   end;
end;

procedure TfmMCX.lvJobsChange(Sender: TObject; Item: TListItem;
  Change: TItemChange);
begin

end;

procedure TfmMCX.lvJobsSelectItem(Sender: TObject; Item: TListItem;
  Selected: Boolean);
begin
    if(lvJobs.Selected <> nil) and (lvJobs.Selected.Index>0) then
        ListToPanel;
end;

procedure TfmMCX.FormCreate(Sender: TObject);
begin
    MapList:=TStringHashList.Create(true);
    MapList.Add('Session',edSession);
    MapList.Add('InputFile',edConfigFile);
    MapList.Add('ThreadNum',edThread);
    MapList.Add('MoveNum',edMove);
    MapList.Add('RespinNum',edRespin);
    MapList.Add('ArrayOrder',grArray);
    MapList.Add('TStart',edT0);
    MapList.Add('TEnd',edT1);
    MapList.Add('GateNum',edGate);
    MapList.Add('DoReflect',ckReflect);
    MapList.Add('DoSave',ckSaveData);
    MapList.Add('DoNormalize',ckNormalize);
end;

procedure TfmMCX.FormDestroy(Sender: TObject);
begin
    MapList.Free;
end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
         pMCX.CommandLine:=CreateCmd;
         pMCX.Options := [poUsePipes];
         AddLog('-- Executing MCX --');
         pMCX.Execute;

         tbtStop.Enabled:=true;
         btStop.Enabled:=true;
         tbtRun.Enabled:=false;
    end
end;

procedure TfmMCX.mcxdoStopExecute(Sender: TObject);
begin
    if(pMCX.Running) then pMCX.Terminate(0);
    Sleep(1000);
    if(not pMCX.Running) then begin
         tbtStop.Enabled:=false;
         btStop.Enabled:=false;
         tbtRun.Enabled:=true;
         AddLog('-- Stopped MCX --');
    end
end;

procedure TfmMCX.mcxdoVerifyExecute(Sender: TObject);
begin
    VarifyInput;
end;

procedure TfmMCX.VarifyInput;
var
    nthread, nmove: integer;
    t0,t1: extended;
begin

    btRun.Enabled:=false;
    tbtRun.Enabled:=false;

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
    tbtRun.Enabled:=true;
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

procedure TfmMCX.ListToPanel;
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    iname: string;
    i,cc: integer;
begin
    if(lvJobs.Selected=nil) then
        exit;

    cc:=lvJobs.Selected.SubItems.Count;
    for i:=0 to cc-1 do
    begin

        iname:=lvJobs.Column[i].Caption;
        if(TObject(MapList.Data[iname]) is TEdit) then
        begin
           ed:=TObject(MapList.Data[iname]) as TEdit;
           ed.Text:=lvJobs.Selected.SubItems.Strings[i];
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TCheckBox) then
        begin
           ck:=TObject(MapList.Data[iname]) as TCheckBox;
           ck.Checked:=(lvJobs.Selected.SubItems.Strings[i]='1');
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TComboBox) then
        begin
           cb:=TObject(MapList.Data[iname]) as TComboBox;
           cb.Text:=lvJobs.Selected.SubItems.Strings[i];
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TSpinEdit) then
        begin
           se:=TObject(MapList.Data[iname]) as TSpinEdit;
           se.Value:=StrToInt(lvJobs.Selected.SubItems.Strings[i]);
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TRadioGroup) then
        begin
           gr:=TObject(MapList.Data[iname]) as TRadioGroup;
           gr.ItemIndex:=StrToInt(lvJobs.Selected.SubItems.Strings[i]);
           continue;
        end;
    end;
end;

procedure TfmMCX.PanelToList;
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    iname: string;
    i: integer;
begin
    if(lvJobs.Selected=nil) then
        exit;

    for i:=0 to lvJobs.Columns.Count-1 do
    begin
        if(lvJobs.Selected.SubItems.Count<=i) then
             lvJobs.Selected.SubItems.Add('');

        iname:=lvJobs.Column[i].Caption;
        if(TObject(MapList.Data[iname]) is TEdit) then
        begin
           ed:=TObject(MapList.Data[iname]) as TEdit;
           lvJobs.Selected.SubItems.Strings[i]:=ed.Text;
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TCheckBox) then
        begin
           ck:=TObject(MapList.Data[iname]) as TCheckBox;
           lvJobs.Selected.SubItems.Strings[i]:=Format('%d',[ck.Checked]);
           continue;
        end;

        if(TObject(MapList.Data[name]) is TComboBox) then
        begin
           cb:=TObject(MapList.Data[name]) as TComboBox;
           cb.Text:=lvJobs.Selected.SubItems.Strings[i];
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TSpinEdit) then
        begin
           se:=TObject(MapList.Data[iname]) as TSpinEdit;
           se.Value:=StrToInt(lvJobs.Selected.SubItems.Strings[i]);
           continue;
        end;

        if(TObject(MapList.Data[iname]) is TRadioGroup) then
        begin
           gr:=TObject(MapList.Data[iname]) as TRadioGroup;
           lvJobs.Selected.SubItems.Strings[i]:=AnsiString(gr.ItemIndex);
           continue;
        end;
    end;
end;

initialization
  {$I mcxgui.lrs}

end.

