unit mcxgui;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, process, FileUtil, LResources, Forms, Controls, Graphics,
  Dialogs, StdCtrls, IniPropStorage, Menus, ComCtrls, ExtCtrls, Spin, EditBtn,
  Buttons, ActnList, lcltype, mcxabout;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    mcxdoDefault: TAction;
    mcxdoClearLog: TAction;
    mcxdoSave: TAction;
    mcxdoExit: TAction;
    mcxdoListGPU: TAction;
    mcxdoQuery: TAction;
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
    mcxdoInitEnv: TAction;
    acMCX: TActionList;
    ckReflect: TCheckBox;
    ckSaveData: TCheckBox;
    ckNormalize: TCheckBox;
    edThread: TComboBox;
    edMove: TEdit;
    edSession: TEdit;
    edT0: TEdit;
    edT1: TEdit;
    pMCX: TProcess;
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
    Process1: TProcess;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    sbInfo: TStatusBar;
    tbtRun: TToolButton;
    tbtRunAll: TToolButton;
    tbtStop: TToolButton;
    tbtVerify: TToolButton;
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
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoDefaultExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure mcxdoOpenExecute(Sender: TObject);
    procedure mcxdoRunExecute(Sender: TObject);
    procedure mcxdoStopExecute(Sender: TObject);
    procedure mcxdoVerifyExecute(Sender: TObject);
    procedure edConfigFileChange(Sender: TObject);
    procedure edRespinChange(Sender: TObject);
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
    procedure UpdateActions(actlst: TActionList; ontag,offtag: string);
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

procedure TfmMCX.edRespinChange(Sender: TObject);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    fed:TFileNameEdit;
    idx: integer;
    node: TListItem;
begin
    if(Sender=nil) or (lvJobs.Selected=nil) then exit;
    try
    node:=lvJobs.Selected;
    if(Sender is TSpinEdit) then begin
       se:=Sender as TSpinEdit;
       AddLog(se.Hint);
       idx:=MapList.IndexOf(se.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(se.Value);
    end else if(Sender is TEdit) then begin
       ed:=Sender as TEdit;
       idx:=MapList.IndexOf(ed.Hint);
       if(idx=0) then  node.Caption:=ed.Text;
       if(idx>0) then
                  node.SubItems.Strings[idx-1]:=ed.Text;
    end else if(Sender is TRadioGroup) then begin
       gr:=Sender as TRadioGroup;
       idx:=MapList.IndexOf(gr.Hint);
       if(idx>0) then
                  node.SubItems.Strings[idx-1]:=IntToStr(gr.ItemIndex);
    end else if(Sender is TComboBox) then begin
       cb:=Sender as TComboBox;
       idx:=MapList.IndexOf(cb.Hint);
       if(idx>0) then
                  node.SubItems.Strings[idx-1]:=cb.Text;
    end else if(Sender is TCheckBox) then begin
       ck:=Sender as TCheckBox;
       idx:=MapList.IndexOf(ck.Hint);
       if(idx>0) then
                  node.SubItems.Strings[idx-1]:=IntToStr(Integer(ck.Checked));
    end else if(Sender is TFileNameEdit) then begin
       fed:=Sender as TFileNameEdit;
       idx:=MapList.IndexOf(fed.Hint);
       if(idx>0) then
                  node.SubItems.Strings[idx-1]:=fed.FileName;
    end;
    UpdateActions(acMCX,'','Work');
    except
    end;
end;

procedure TfmMCX.mcxdoExitExecute(Sender: TObject);
begin
  Close;
end;

procedure TfmMCX.mcxdoAddItemExecute(Sender: TObject);
var
   node: TListItem;
   i:integer;
   sessionid: string;
begin
   sessionid:=InputBox('Set Session Name','Please supply a string that uniquely identifies this session','');
   if(length(sessionid)=0) then exit;
   node:=lvJobs.Items.Add;
   for i:=0 to lvJobs.Columns.Count-1 do node.SubItems.Add('');
   node.Caption:=sessionid;
   plSetting.Enabled:=true;
   lvJobs.Selected:=node;
   edSession.Text:=sessionid;
   mcxdoDefaultExecute(nil);
end;

procedure TfmMCX.mcxdoDefaultExecute(Sender: TObject);
begin
      edSession.Text:='';
      edConfigFile.FileName:='';
      edThread.Text:='1796';
      edMove.Text:='1000000';
      edT0.Text:='1e-9';
      edT1.Text:='1e-10';
      edGate.Value:=1;
      edRespin.Value:=1;
      grArray.ItemIndex:=0;
end;

procedure TfmMCX.UpdateActions(actlst: TActionList; ontag,offtag: string);
var
   i: integer;
begin
   for i:=0 to actlst.ActionCount-1 do begin
       if (Length(ontag)>0) and (actlst.Actions[i].Category=ontag) then begin
             (actlst.Actions[i] as TAction).Enabled:=true;
       end else if (Length(offtag)>0) and (actlst.Actions[i].Category=offtag) then
             (actlst.Actions[i] as TAction).Enabled:=false;
   end;
end;


procedure TfmMCX.mcxdoAboutExecute(Sender: TObject);
var
    fmAbout:TfmAbout;
begin
     fmAbout:=TfmAbout.Create(Application);
     fmAbout.ShowModal;
     fmAbout.Free;
end;

procedure TfmMCX.mcxdoDeleteItemExecute(Sender: TObject);
begin
  if not (lvJobs.Selected = nil) then
  begin
        if(Application.MessageBox('The selected configuration will be deleted, are you sure?',
          'Confirm', MB_YESNOCANCEL)=IDYES) then
            exit;
        lvJobs.Items.Delete(lvJobs.Selected.Index);
  end;
end;

procedure TfmMCX.mcxdoOpenExecute(Sender: TObject);
begin

end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          pMCX.CommandLine:=CreateCmd;
          pMCX.Options := [poUsePipes];
          AddLog('-- Executing MCX --');
          pMCX.Execute;

          mcxdoStop.Enabled:=true;
          mcxdoRun.Enabled:=false;
    end;
end;

procedure TfmMCX.mcxdoStopExecute(Sender: TObject);
begin
     if(pMCX.Running) then pMCX.Terminate(0);
     Sleep(1000);
     if(not pMCX.Running) then begin
          mcxdoStop.Enabled:=false;
          mcxdoRun.Enabled:=true;
          AddLog('-- Stopped MCX --');
     end
end;

procedure TfmMCX.mcxdoVerifyExecute(Sender: TObject);
begin
    VarifyInput;
end;

procedure TfmMCX.FormCreate(Sender: TObject);
var
   i: integer;
begin
    MapList:=TStringList.Create();
    MapList.Clear;
    for i:=0 to lvJobs.Columns.Count-1 do begin
        MapList.Add(lvJobs.Columns.Items[i].Caption);
    end;
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
          if not (lvJobs.Selected=nil) then
              plSetting.Enabled:=true;
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

    UpdateActions(acMCX,'Work','');
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
    i,idx: integer;
begin
    if(node=nil) then exit;
    for i:=0 to plSetting.ControlCount-1 do
    begin
        try
        if(plSetting.Controls[i] is TSpinEdit) then begin
           se:=plSetting.Controls[i] as TSpinEdit;
           AddLog(se.Hint);
           idx:=MapList.IndexOf(se.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(se.Value);
           continue;
        end;
        if(plSetting.Controls[i] is TEdit) then begin
           ed:=plSetting.Controls[i] as TEdit;
           AddLog(ed.Hint);
           idx:=MapList.IndexOf(ed.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=ed.Text;
           continue;
        end;
        if(plSetting.Controls[i] is TRadioGroup) then begin
           gr:=plSetting.Controls[i] as TRadioGroup;
           AddLog(gr.Hint);
           idx:=MapList.IndexOf(gr.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
           continue;
        end;
        if(plSetting.Controls[i] is TComboBox) then begin
           cb:=plSetting.Controls[i] as TComboBox;
           AddLog(cb.Hint);
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=cb.Text;
           continue;
        end;
        if(plSetting.Controls[i] is TCheckBox) then begin
           ck:=plSetting.Controls[i] as TCheckBox;
           AddLog(ck.Hint);
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(Integer(ck.Checked));
           continue;
        end;

        except
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
    i,idx: integer;
begin
    if(node=nil) then exit;
    for i:=0 to plSetting.ControlCount-1 do
    begin
        if(plSetting.Controls[i] is TSpinEdit) then begin
           se:=plSetting.Controls[i] as TSpinEdit;
           idx:=MapList.IndexOf(se.Hint);
           if(idx>=0) then begin
                try
                      se.Value:=StrToInt(node.SubItems.Strings[idx]);
                except
                end;
           end;
           continue;
        end;
        if(plSetting.Controls[i] is TEdit) then begin
           ed:=plSetting.Controls[i] as TEdit;
           idx:=MapList.IndexOf(ed.Hint);
           if(idx>=0) then ed.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSetting.Controls[i] is TRadioGroup) then begin
           gr:=plSetting.Controls[i] as TRadioGroup;
           AddLog(gr.Hint);
           idx:=MapList.IndexOf(gr.Hint);
           if(idx>=0) then begin
                try
                      gr.ItemIndex:=StrToInt(node.SubItems.Strings[idx]);
                except
                end;
           end;
           continue;
        end;
        if(plSetting.Controls[i] is TComboBox) then begin
           cb:=plSetting.Controls[i] as TComboBox;
           AddLog(cb.Hint);
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then cb.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSetting.Controls[i] is TCheckBox) then begin
           ck:=plSetting.Controls[i] as TCheckBox;
           AddLog(ck.Hint);
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then ck.Checked:=(node.SubItems.Strings[idx]='1');
           continue;
        end;
    end;
end;
initialization
  {$I mcxgui.lrs}

end.

