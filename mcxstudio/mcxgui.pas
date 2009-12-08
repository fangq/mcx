unit mcxgui;
{==============================================================================
    Monte Carlo eXtreme (MCX) Studio
-------------------------------------------------------------------------------
    Author: Qianqian Fang
    Email : fangq at nmr.mgh.harvard.edu
    Web   : http://mcx.sourceforge.net
    License: GNU General Public License version 3 (GPLv3)
===============================================================================}
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, process, FileUtil, LResources, Forms, Controls,
  Graphics, Dialogs, StdCtrls, Menus, ComCtrls, ExtCtrls, Spin,
  EditBtn, Buttons, ActnList, lcltype, AsyncProcess,
  inifiles, mcxabout, unix;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    OpenProject: TOpenDialog;
    pMCX: TAsyncProcess;
    ckAtomic: TCheckBox;
    edBlockSize: TComboBox;
    Label11: TLabel;
    mcxSetCurrent: TAction;
    acInteract: TActionList;
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
    edBubble: TEdit;
    edConfigFile: TFileNameEdit;
    ImageList1: TImageList;
    Label1: TLabel;
    Label10: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
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
    SaveProject: TSaveDialog;
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
    procedure ckAtomicClick(Sender: TObject);
    procedure edConfigFileEnter(Sender: TObject);
    procedure edConfigFileExit(Sender: TObject);
    procedure lvJobsChange(Sender: TObject; Item: TListItem; Change: TItemChange
      );
    procedure lvJobsDeletion(Sender: TObject; Item: TListItem);
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoDefaultExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure mcxdoHelpExecute(Sender: TObject);
    procedure mcxdoOpenExecute(Sender: TObject);
    procedure mcxdoQueryExecute(Sender: TObject);
    procedure mcxdoRunExecute(Sender: TObject);
    procedure mcxdoSaveExecute(Sender: TObject);
    procedure mcxdoStopExecute(Sender: TObject);
    procedure mcxdoVerifyExecute(Sender: TObject);
    procedure edConfigFileChange(Sender: TObject);
    procedure edRespinChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure lvJobsSelectItem(Sender: TObject; Item: TListItem;
      Selected: Boolean);
    procedure mcxdoWebExecute(Sender: TObject);
    procedure mcxSetCurrentExecute(Sender: TObject);
    procedure pMCXReadData(Sender: TObject);
    procedure pMCXTerminate(Sender: TObject);
    procedure ToolButton14Click(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    MapList: TStringList;
    function CreateCmd:string;
    function CreateCmdOnly:string;
    procedure VarifyInput;
    procedure AddLog(str:string);
    procedure ListToPanel2(node:TListItem);
    procedure PanelToList2(node:TListItem);
    procedure UpdateMCXActions(actlst: TActionList; ontag,offtag: string);
    function  GetMCXOutput (outputstr: string) : string;
    procedure SaveTasksToIni(fname: string);
    procedure LoadTasksFromIni(fname: string);
    function  GetBrowserPath : string;
    function  SearchForExe(const fname : string) : string;
  end;

var
  fmMCX: TfmMCX;
  ProfileChanged: Boolean;
  MaxWait: integer;
  TaskFile: string;

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
       idx:=MapList.IndexOf(se.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(se.Value);
    end else if(Sender is TEdit) then begin
       ed:=Sender as TEdit;
       idx:=MapList.IndexOf(ed.Hint);
       if(ed.Hint = 'Session') then  node.Caption:=ed.Text;
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=ed.Text;
    end else if(Sender is TRadioGroup) then begin
       gr:=Sender as TRadioGroup;
       idx:=MapList.IndexOf(gr.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
    end else if(Sender is TComboBox) then begin
       cb:=Sender as TComboBox;
       idx:=MapList.IndexOf(cb.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=cb.Text;
    end else if(Sender is TCheckBox) then begin
       ck:=Sender as TCheckBox;
       idx:=MapList.IndexOf(ck.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(Integer(ck.Checked));
    end else if(Sender is TFileNameEdit) then begin
       fed:=Sender as TFileNameEdit;
       idx:=MapList.IndexOf(fed.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=fed.FileName;
    end;
    UpdateMCXActions(acMCX,'','Work');
    except
    end;
end;

procedure TfmMCX.mcxdoExitExecute(Sender: TObject);
begin
    if(mcxdoSave.Enabled) then begin
       if (Application.MessageBox('The current session has not been saved, do you want to save before exit?',
         'Confirm', MB_YESNOCANCEL)=IDYES) then
            mcxdoSaveExecute(Sender);
    end;
    Close;
end;

procedure TfmMCX.mcxdoHelpExecute(Sender: TObject);
begin
   Shell(GetBrowserPath + ' http://mcx.sourceforge.net/cgi-bin/index.cgi?Doc');
end;

procedure TfmMCX.mcxdoAddItemExecute(Sender: TObject);
var
   node: TListItem;
   i:integer;
   sessionid: string;
begin
   sessionid:=InputBox('Set Session Name','Please type in a unique session name','');
   if(length(sessionid)=0) then exit;
   for i:=0 to lvJobs.Items.Count-1 do begin
        if(lvJobs.Items.Item[i].Caption = sessionid) then
           raise Exception.Create('Session name already used!');
   end;
   node:=lvJobs.Items.Add;
   for i:=0 to lvJobs.Columns.Count-1 do node.SubItems.Add('');
   node.Caption:=sessionid;
   plSetting.Enabled:=true;
   lvJobs.Selected:=node;
   mcxdoDefaultExecute(nil);
   edSession.Text:=sessionid;
   UpdateMCXActions(acMCX,'','Work');
   UpdateMCXActions(acMCX,'Preproc','');
   UpdateMCXActions(acMCX,'SelectedJob','');
end;

procedure TfmMCX.mcxdoDefaultExecute(Sender: TObject);
begin
      //edSession.Text:='';
      edConfigFile.FileName:='';
      edThread.Text:='1796';
      edMove.Text:='1000000';
      edBlockSize.Text:='128';
      edBubble.Text:='0';
      edGate.Value:=1;
      edRespin.Value:=1;
      grArray.ItemIndex:=0;
      ckReflect.Checked:=true;
      ckSaveData.Checked:=true;
      ckNormalize.Checked:=true;
      ckAtomic.Checked:=false;
      if not (lvJobs.Selected = nil) then
         PanelToList2(lvJobs.Selected);
end;

procedure TfmMCX.UpdateMCXActions(actlst: TActionList; ontag,offtag: string);
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

procedure TfmMCX.lvJobsChange(Sender: TObject; Item: TListItem;
  Change: TItemChange);
begin
  mcxdoSave.Enabled:=true;
end;

procedure TfmMCX.lvJobsDeletion(Sender: TObject; Item: TListItem);
begin

end;

procedure TfmMCX.ckAtomicClick(Sender: TObject);
begin
  if(ckAtomic.Checked) then begin
        ShowMessage('You selected to use atomic operations. We suggest you only using this mode when the accuracy near the source is critically important. Atomic mode is about 5 times slower than non-atomic one, and you should use a thread number between 500~1000.');
  end;
end;

procedure TfmMCX.edConfigFileEnter(Sender: TObject);
begin
  lvJobs.Enabled:=false;
end;

procedure TfmMCX.edConfigFileExit(Sender: TObject);
begin
  lvJobs.Enabled:=true;
end;

procedure TfmMCX.mcxdoDeleteItemExecute(Sender: TObject);
begin
  if not (lvJobs.Selected = nil) then
  begin
        if not (Application.MessageBox('The selected configuration will be deleted, are you sure?',
          'Confirm', MB_YESNOCANCEL)=IDYES) then
            exit;
        lvJobs.Items.Delete(lvJobs.Selected.Index);
        if not (lvJobs.Selected = nil) then
            ListToPanel2(lvJobs.Selected);
  end;
end;

procedure TfmMCX.mcxdoOpenExecute(Sender: TObject);
begin
  if(OpenProject.Execute) then begin
    TaskFile:=OpenProject.FileName;
    if(mcxdoSave.Enabled) then begin
       if(Application.MessageBox('The current session has not been saved, do you want to discard?',
         'Confirm', MB_YESNOCANCEL)=IDYES) then
            LoadTasksFromIni(TaskFile);
    end else begin
            LoadTasksFromIni(TaskFile);
    end;
  end
end;

procedure TfmMCX.mcxdoQueryExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          pMCX.CommandLine:=CreateCmdOnly+' -L';
          pMCX.Options := [poUsePipes];
          AddLog('-- Executing MCX --');
          pMCX.Execute;

          mcxdoStop.Enabled:=true;
          mcxdoRun.Enabled:=false;
    end;
end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          pMCX.CommandLine:=CreateCmd;
          //pMCX.CommandLine:='du /usr/ --max-depth=1';
          pMCX.Options := [poUsePipes];
          AddLog('-- Executing MCX --');
          pMCX.Execute;
          mcxdoStop.Enabled:=true;
          mcxdoRun.Enabled:=false;
          sbInfo.Panels[0].Text := 'Status: busy';
    end;
end;

procedure TfmMCX.mcxdoSaveExecute(Sender: TObject);
begin
  if(SaveProject.Execute) then begin
    TaskFile:=SaveProject.FileName;
    if(length(TaskFile) >0) then begin
        SaveTasksToIni(TaskFile);
        mcxdoSave.Enabled:=false;
    end;
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
    for i:=1 to lvJobs.Columns.Count-1 do begin
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
          if (lvJobs.Selected=nil) then begin
          end
     end
end;


function TfmMCX.SearchForExe(const fname : string) : string;
begin
   Result :=
    SearchFileInPath(fname, '', GetEnvironmentVariable('PATH'),
                     PathSeparator, [sffDontSearchInBasePath]);
end;

function TfmMCX.GetBrowserPath : string;
  {Return path to first browser found.}
begin
   Result := SearchForExe('firefox');
   if Result = '' then
     Result := SearchForExe('konqueror');  {KDE browser}
   if Result = '' then
     Result := SearchForExe('epiphany');  {GNOME browser}
   if Result = '' then
     Result := SearchForExe('mozilla');
   if Result = '' then
     Result := SearchForExe('opera');
end;

procedure TfmMCX.mcxdoWebExecute(Sender: TObject);
begin
  Shell(GetBrowserPath + ' http://mcx.sourceforge.net');
end;

procedure TfmMCX.mcxSetCurrentExecute(Sender: TObject);
begin
     if not (lvJobs.Selected = nil) then begin
         ListToPanel2(lvJobs.Selected);
         plSetting.Enabled:=true;
         mcxdoVerify.Enabled:=true;
     end;
end;

procedure TfmMCX.pMCXReadData(Sender: TObject);
begin
     mmOutput.Lines.Text:=GetMCXOutput(mmOutput.Lines.Text);
     if not (pMCX.Running) then
         pMCXTerminate(Sender);
end;

procedure TfmMCX.pMCXTerminate(Sender: TObject);
begin
     mcxdoStop.Enabled:=false;
     mcxdoRun.Enabled:=true;
     sbInfo.Panels[0].Text := 'Status: idle';
     AddLog('Task complete');
end;

procedure TfmMCX.ToolButton14Click(Sender: TObject);
begin

end;

function TfmMCX.GetMCXOutput (outputstr: string) : string;
var
    Buffer: string;
    BytesAvailable: DWord;
    BytesRead:LongInt;
begin
   Result:= outputstr;

   //if(pMCX.Running) then
   if true then
    begin
      BytesAvailable := pMCX.Output.NumBytesAvailable;
      BytesRead := 0;
      while BytesAvailable>0 do
      begin
        SetLength(Buffer, BytesAvailable);
        BytesRead := pMCX.OutPut.Read(Buffer[1], BytesAvailable);
        Result := Result + copy(Buffer,1, BytesRead);
        BytesAvailable := pMCX.Output.NumBytesAvailable;
      end;
    end;
end;

procedure TfmMCX.SaveTasksToIni(fname: string);
var
   inifile: TIniFile;
   i,j: integer;
begin
     inifile:=TIniFile.Create(fname);
     for i:=0 to lvJobs.Items.Count-1 do begin
          for j:=0 to lvJobs.Columns.Count-1 do begin
              inifile.WriteString(lvJobs.Items.Item[i].Caption,
                                  lvJobs.Columns.Items[j].Caption,
                                  lvJobs.Items.Item[i].SubItems.Strings[j]);
          end;
     end;
     inifile.UpdateFile;
     inifile.Free;
end;

procedure TfmMCX.LoadTasksFromIni(fname: string);
var
   sessions,vals:TStringList;
   inifile: TIniFile;
   i,j: integer;
   node: TListItem;
begin
     sessions:=TStringList.Create;
     vals:=TStringList.Create;
     inifile:=TIniFile.Create(fname);
     inifile.ReadSections(sessions);
     for i:=0 to sessions.Count-1 do begin
         node:=lvJobs.Items.Add;
         node.Caption:=sessions.Strings[i];
         inifile.ReadSectionValues(node.Caption,vals);
         for j:=0 to lvJobs.Columns.Count-1 do
             node.SubItems.Add('');
         for j:=0 to lvJobs.Columns.Count-1 do begin
             node.SubItems.Strings[j]:=vals.Values[lvJobs.Columns.Items[j].Caption];
         end
     end;
     inifile.Free;
     vals.Free;
     sessions.Free;
end;


procedure TfmMCX.VarifyInput;
var
    nthread, nmove, nblock: integer;
    radius,t1: extended;
    exepath: string;
begin
  try
    if(Length(edConfigFile.FileName)=0) then
        raise Exception.Create('Config file must be specified');
    if(not FileExists(edConfigFile.FileName)) then
        raise Exception.Create('Config file does not exist, please check the path');
    try
        nthread:=StrToInt(edThread.Text);
        nmove:=StrToInt(edMove.Text);
        radius:=StrToFloat(edBubble.Text);
        nblock:=StrToInt(edBlockSize.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, move and time gate values');
    end;
    if(nthread<512) then
       AddLog('Warning: increase thread numbers to 1024 or above may boost the speed significantly');
    if(nthread>2048) then
       AddLog('Warning: you may need a high-end graphics card to use more threads');
    if(nmove>1e7) then
       AddLog('Warning: you can increase respin number to get more photons');
    if(nblock<0) then
       raise Exception.Create('Thread block number can not be negative');
    if(radius<0) then
       raise Exception.Create('Bubble radius can not be negative');

    exepath:=SearchForExe(CreateCmdOnly);
    if(exepath='') then
       raise Exception.Create('Can not find mcx in the search path');


    UpdateMCXActions(acMCX,'Work','');
  except
    On E : Exception do
      ShowMessage(E.Message);
  end;
end;

function TfmMCX.CreateCmdOnly:string;
var
    cmd: string;
begin
    cmd:='mcx';
    if(ckAtomic.Checked) then cmd:='mcx_atomic';
    Result:=cmd;
end;

function TfmMCX.CreateCmd:string;
var
    nthread, nmove, nblock: integer;
    bubbleradius: extended;
    cmd: string;
begin
//    cmd:='"'+Config.MCXExe+'" ';
    cmd:=CreateCmdOnly;
    if(Length(edSession.Text)>0) then
       cmd:=cmd+' --session "'+Trim(edSession.Text)+'" ';
    if(Length(edConfigFile.FileName)>0) then
       cmd:=cmd+' --input "'+Trim(edConfigFile.FileName)+'" ';
    try
        nthread:=StrToInt(edThread.Text);
        nmove:=StrToInt(edMove.Text);
        nblock:=StrToInt(edBlockSize.Text);
        bubbleradius:=StrToFloat(edBubble.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, move and time gate values');
    end;

    cmd:=cmd+Format(' --thread %d --move %d --repeat %d --array %d --blocksize %d --skipradius %f ',
      [nthread,nmove,edRespin.Value,grArray.ItemIndex,nblock,bubbleradius]);
    cmd:=cmd+Format(' --normalize %d --save2pt %d --reflect %d ',
      [Integer(ckNormalize.Checked),Integer(ckSaveData.Checked),Integer(ckReflect.Checked)]);

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
           idx:=MapList.IndexOf(se.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(se.Value);
           continue;
        end;
        if(plSetting.Controls[i] is TEdit) then begin
           ed:=plSetting.Controls[i] as TEdit;
           idx:=MapList.IndexOf(ed.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=ed.Text;
           continue;
        end;
        if(plSetting.Controls[i] is TRadioGroup) then begin
           gr:=plSetting.Controls[i] as TRadioGroup;
           idx:=MapList.IndexOf(gr.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
           continue;
        end;
        if(plSetting.Controls[i] is TComboBox) then begin
           cb:=plSetting.Controls[i] as TComboBox;
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=cb.Text;
           continue;
        end;
        if(plSetting.Controls[i] is TCheckBox) then begin
           ck:=plSetting.Controls[i] as TCheckBox;
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
    fed:TFileNameEdit;
    i,idx: integer;
begin
    if(node=nil) then exit;
    edSession.Text:=node.Caption;

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
        if(plSetting.Controls[i] is TFileNameEdit) then begin
           fed:=plSetting.Controls[i] as TFileNameEdit;
           idx:=MapList.IndexOf(fed.Hint);
           if(idx>=0) then fed.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSetting.Controls[i] is TRadioGroup) then begin
           gr:=plSetting.Controls[i] as TRadioGroup;
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
           idx:=MapList.IndexOf(cb.Hint);
           if(idx>=0) then cb.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(plSetting.Controls[i] is TCheckBox) then begin
           ck:=plSetting.Controls[i] as TCheckBox;
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then ck.Checked:=(node.SubItems.Strings[idx]='1');
           continue;
        end;
    end;
end;
initialization
  {$I mcxgui.lrs}

end.

