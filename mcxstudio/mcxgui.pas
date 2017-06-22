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
  Classes, SysUtils, process, FileUtil, LvlGraphCtrl, TAGraph, SynEdit,
  SynHighlighterAny, SynHighlighterPerl, synhighlighterunixshellscript,
  LResources, Forms, Controls, Graphics, Dialogs, StdCtrls, Menus, ComCtrls,
  ExtCtrls, Spin, EditBtn, Buttons, ActnList, lcltype, AsyncProcess, ValEdit,
  Grids, inifiles, fpjson, jsonparser, mcxabout;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    Button1: TButton;
    ckAutopilot: TCheckBox;
    ckSaveExit: TCheckBox;
    ckSrcFrom0: TCheckBox;
    ckSaveRef: TCheckBox;
    ckNormalize: TCheckBox;
    ckReflect: TCheckBox;
    ckSaveData: TCheckBox;
    ckSaveDetector: TCheckBox;
    ckSkipVoid: TCheckBox;
    edBlockSize: TComboBox;
    edBubble: TEdit;
    edConfigFile: TFileNameEdit;
    edDetectedNum: TEdit;
    edGate: TSpinEdit;
    edGPUID: TComboBox;
    edPhoton: TEdit;
    edReseed: TEdit;
    edRespin: TSpinEdit;
    edSeed: TEdit;
    edSession: TEdit;
    edThread: TComboBox;
    edUnitInMM: TEdit;
    grAdditional: TGroupBox;
    grArray: TRadioGroup;
    grBasic: TGroupBox;
    grGPU: TGroupBox;
    grSwitches: TGroupBox;
    HeaderControl1: THeaderControl;
    Label1: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    Label14: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    lvJobs: TListView;
    mcxdoHelpOptions: TAction;
    miClearLog: TMenuItem;
    OpenProject: TOpenDialog;
    pcSimuEditor: TPageControl;
    Panel1: TPanel;
    Panel2: TPanel;
    plOutput: TPanel;
    pMCX: TAsyncProcess;
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
    ImageList1: TImageList;
    MainMenu1: TMainMenu;
    MenuItem17: TMenuItem;
    MenuItem18: TMenuItem;
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
    pExternal: TProcess;
    PopupMenu1: TPopupMenu;
    RadioButton1: TRadioButton;
    RadioButton2: TRadioButton;
    SaveProject: TSaveDialog;
    sbInfo: TStatusBar;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    Splitter3: TSplitter;
    Splitter4: TSplitter;
    Splitter5: TSplitter;
    StaticText1: TStaticText;
    StaticText2: TStaticText;
    StaticText3: TStaticText;
    sgMedia: TStringGrid;
    sgDet: TStringGrid;
    sgConfig: TStringGrid;
    mmOutput: TSynEdit;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    tabInputData: TTabSheet;
    TabSheet2: TTabSheet;
    tbtRun: TToolButton;
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
    TreeView1: TTreeView;
    procedure ckAtomicClick(Sender: TObject);
    procedure ckUseInputFileChange(Sender: TObject);
    procedure FormDockOver(Sender: TObject; Source: TDragDockObject; X,
      Y: Integer; State: TDragState; var Accept: Boolean);
    procedure lvJobsChange(Sender: TObject; Item: TListItem; Change: TItemChange
      );
    procedure lvJobsDeletion(Sender: TObject; Item: TListItem);
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoDefaultExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure mcxdoHelpExecute(Sender: TObject);
    procedure mcxdoHelpOptionsExecute(Sender: TObject);
    procedure mcxdoOpenExecute(Sender: TObject);
    procedure mcxdoQueryExecute(Sender: TObject);
    procedure mcxdoRunAllExecute(Sender: TObject);
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
    procedure miClearLogClick(Sender: TObject);
    procedure mmOutput1DragDrop(Sender, Source: TObject; X, Y: Integer);
    procedure mmOutput1DragOver(Sender, Source: TObject; X, Y: Integer;
      State: TDragState; var Accept: Boolean);
    procedure mmOutput1MouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure plOutputDockOver(Sender: TObject; Source: TDragDockObject; X,
      Y: Integer; State: TDragState; var Accept: Boolean);
    procedure plSettingDockOver(Sender: TObject; Source: TDragDockObject; X,
      Y: Integer; State: TDragState; var Accept: Boolean);
    procedure pMCXReadData(Sender: TObject);
    procedure pMCXTerminate(Sender: TObject);
    procedure RadioButton1Change(Sender: TObject);
    procedure sgMediaEditingDone(Sender: TObject);
    procedure sgMediaGetEditMask(Sender: TObject; ACol, ARow: Integer;
      var Value: string);
    procedure StaticText1DblClick(Sender: TObject);
    procedure StaticText2Click(Sender: TObject);
    procedure StaticText2DblClick(Sender: TObject);
    procedure StaticText3DblClick(Sender: TObject);
    procedure ToolButton14Click(Sender: TObject);
    procedure ValueListEditor1Click(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    MapList: TStringList;
    function CreateCmd:string;
    function CreateCmdOnly:string;
    function GridToJSONArray(grid:TStringGrid):string;
    function GridToJSONStruct(grid:TStringGrid):string;
    procedure VerifyInput;
    procedure AddLog(str:string);
    procedure ListToPanel2(node:TListItem);
    procedure PanelToList2(node:TListItem);
    procedure UpdateMCXActions(actlst: TActionList; ontag,offtag: string);
    function  GetMCXOutput (outputstr: string) : string;
    procedure SaveTasksToIni(fname: string);
    procedure LoadTasksFromIni(fname: string);
    procedure RunExternalCmd(cmd: string);
    function  GetBrowserPath : string;
    function  SearchForExe(fname : string) : string;
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
       if(ck.Hint='Autopilot') then begin
           edThread.Enabled:=not ck.Checked;
           edBlockSize.Enabled:=not ck.Checked;
       end;
       if(ck.Hint='SaveDetector') then begin
           edDetectedNum.Enabled:=ck.Checked;
       end;
    end else if(Sender is TFileNameEdit) then begin
       fed:=Sender as TFileNameEdit;
       idx:=MapList.IndexOf(fed.Hint);
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=fed.Text;
    end;
    UpdateMCXActions(acMCX,'','Work');
    UpdateMCXActions(acMCX,'','Run');
    mcxdoSave.Enabled:=true;
    except
    end;
end;

procedure TfmMCX.mcxdoExitExecute(Sender: TObject);
var
   ret:integer;
begin
    if(mcxdoSave.Enabled) then begin
       ret:=Application.MessageBox('The current session has not been saved, do you want to save before exit?',
         'Confirm', MB_YESNOCANCEL);
       if (ret=IDYES) then
            mcxdoSaveExecute(Sender);
       if (ret=IDCANCEL) then
            exit;
    end;
    Close;
end;

procedure TfmMCX.mcxdoHelpExecute(Sender: TObject);
begin
   RunExternalCmd(GetBrowserPath + ' http://mcx.sourceforge.net/cgi-bin/index.cgi?Doc');
end;

procedure TfmMCX.mcxdoHelpOptionsExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          pMCX.CommandLine:=CreateCmdOnly;
          //pMCX.Options := [poUsePipes, poStderrToOutput];
          AddLog('"-- Print MCX Command Line Options --"');
          mmOutput.Tag:=mmOutput.Lines.Count;
          pMCX.Execute;
    end;
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
   pcSimuEditor.Enabled:=true;
   lvJobs.Selected:=node;
   mcxdoDefaultExecute(nil);
   edSession.Text:=sessionid;
   UpdateMCXActions(acMCX,'','Work');
   UpdateMCXActions(acMCX,'','Run');
   UpdateMCXActions(acMCX,'Preproc','');
   UpdateMCXActions(acMCX,'SelectedJob','');
end;

procedure TfmMCX.mcxdoDefaultExecute(Sender: TObject);
begin
      //edSession.Text:='';
      edConfigFile.FileName:='';
      edThread.Text:='4096';
      edPhoton.Text:='1e7';
      edBlockSize.Text:='64';
      edBubble.Text:='-2';
      edGate.Value:=1;
      edRespin.Value:=1;
      grArray.ItemIndex:=0;
      ckReflect.Checked:=true;   //-b
      ckSaveData.Checked:=true;   //-S
      ckNormalize.Checked:=true;   //-U
      ckSaveDetector.Checked:=true;   //-d
      ckSaveExit.Checked:=false;  //-x
      ckSaveRef.Checked:=false;  //-X
      ckSrcFrom0.Checked:=false;  //-z
      ckSkipVoid.Checked:=false;  //-k
      ckAutopilot.Checked:=true;
      edThread.Enabled:=false;
      edBlockSize.Enabled:=false;
      edUnitInMM.Text:='1';
      edGPUID.ItemIndex:=0;
      edDetectedNum.Text:='10000000';
      edSeed.Text:='0';
      edReseed.Text:='10000000';
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
end;

procedure TfmMCX.ckUseInputFileChange(Sender: TObject);
begin

end;

procedure TfmMCX.FormDockOver(Sender: TObject; Source: TDragDockObject; X,
  Y: Integer; State: TDragState; var Accept: Boolean);
begin
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
            ListToPanel2(lvJobs.Selected)
        else
            mcxdoDeleteItem.Enabled:=false;
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
          //pMCX.Options := [poUsePipes, poStderrToOutput];
          AddLog('"-- Printing GPU Information --"');
          mmOutput.Tag:=mmOutput.Lines.Count;
          pMCX.Execute;

          UpdateMCXActions(acMCX,'Run','');
    end;
end;

procedure TfmMCX.mcxdoRunAllExecute(Sender: TObject);
begin

end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          pMCX.CommandLine:=CreateCmd;
          //pMCX.CommandLine:='du /usr/ --max-depth=1';
          //pMCX.Options := pMCX.Options+[poUsePipes];
          AddLog('"-- Executing MCX --"');
          mmOutput.Tag:=mmOutput.Lines.Count;
          pMCX.Execute;
          mcxdoStop.Enabled:=true;
          mcxdoRun.Enabled:=false;
          sbInfo.Panels[0].Text := 'Status: busy';
          sbInfo.Color := clRed;
          UpdateMCXActions(acMCX,'Run','');
          Application.ProcessMessages;
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
          if(mcxdoVerify.Enabled) then
             mcxdoRun.Enabled:=true;
          AddLog('-- Stopped MCX --');
     end
end;

procedure TfmMCX.mcxdoVerifyExecute(Sender: TObject);
begin
    VerifyInput;
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
    if not (SearchForExe(CreateCmdOnly) = '') then begin
        mcxdoQuery.Enabled:=true;
        mcxdoHelpOptions.Enabled:=true;
    end;
end;

procedure TfmMCX.FormDestroy(Sender: TObject);
begin
    MapList.Free;
end;

procedure TfmMCX.lvJobsSelectItem(Sender: TObject; Item: TListItem;
  Selected: Boolean);
begin
     if(Selected) then begin
          if (lvJobs.Selected=nil) then begin
          end else begin
              mcxdoDeleteItem.Enabled:=true;
          end;
     end
end;

procedure TfmMCX.RunExternalCmd(cmd: string);
var
  Proc : TProcess;
begin
  Proc := TProcess.Create(nil);
  try
    Proc.CommandLine := cmd;
    //PRoc.Options := Proc.Options + [poWaitOnExit];
    PRoc.Execute;
  finally
    Proc.free;
  end;
end;

function TfmMCX.SearchForExe(fname : string) : string;
begin
   {$IFDEF WINDOWS}
   if (Pos('.exe',Trim(LowerCase(fname)))<=0) or (Pos('.exe',Trim(LowerCase(fname))) <> Length(Trim(fname))-3) then
           fname:=fname+'.exe';
   {$ENDIF}
   Result :=
    SearchFileInPath(fname, '', ExtractFilePath(Application.ExeName)+PathSeparator+GetEnvironmentVariable('PATH'),
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
   if Result = '' then
     Result := SearchForExe('open'); // mac os
   if Result = '' then
     Result :='cmd /c start'; // windows
end;

procedure TfmMCX.mcxdoWebExecute(Sender: TObject);
begin
  RunExternalCmd(GetBrowserPath + ' http://mcx.space');
end;

procedure TfmMCX.mcxSetCurrentExecute(Sender: TObject);
begin
     if not (lvJobs.Selected = nil) then begin
         ListToPanel2(lvJobs.Selected);
         plSetting.Enabled:=true;
         pcSimuEditor.Enabled:=true;
         mcxdoVerify.Enabled:=true;
         mcxdoDefault.Enabled:=true;
     end;
end;

procedure TfmMCX.miClearLogClick(Sender: TObject);
begin
    mmOutput.Lines.Clear;
end;

procedure TfmMCX.mmOutput1DragDrop(Sender, Source: TObject; X, Y: Integer);
begin

end;

procedure TfmMCX.mmOutput1DragOver(Sender, Source: TObject; X, Y: Integer;
  State: TDragState; var Accept: Boolean);
begin
end;

procedure TfmMCX.mmOutput1MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
end;

procedure TfmMCX.plOutputDockOver(Sender: TObject; Source: TDragDockObject; X,
  Y: Integer; State: TDragState; var Accept: Boolean);
var
   pos:TRect;
   mm: TMemo;
begin
     Accept:=false;
     if (Sender is TMemo) then
        Accept:=true;
     if(Accept) then begin
          mm:=(Sender as TMemo);
          pos:=Rect(0,Height-mm.Height, Width, Height);
          mm.Dock(Self,pos);
     end;
end;

procedure TfmMCX.plSettingDockOver(Sender: TObject; Source: TDragDockObject; X,
  Y: Integer; State: TDragState; var Accept: Boolean);
begin
end;

procedure TfmMCX.pMCXReadData(Sender: TObject);
begin
     mmOutput.Lines.Text:=GetMCXOutput(mmOutput.Lines.Text);
     mmOutput.TopLine:=mmOutput.Tag;
     if not (pMCX.Running) then
         pMCXTerminate(Sender);
end;

procedure TfmMCX.pMCXTerminate(Sender: TObject);
begin
     mcxdoStop.Enabled:=false;
     if(mcxdoVerify.Enabled) then
         mcxdoRun.Enabled:=true;
     sbInfo.Panels[0].Text := 'Status: idle';
     sbInfo.Color := clBtnFace;
     AddLog('Task complete');
     UpdateMCXActions(acMCX,'','Run');
end;

procedure TfmMCX.RadioButton1Change(Sender: TObject);
var
   btn: TRadioButton;
begin
   if(Sender is TRadioButton) then
   begin
        btn:= Sender as TRadioButton;
        if(btn.Checked) then
        begin
            edConfigFile.Enabled:=true;
            tabInputData.Enabled:=false;
        end else begin
            edConfigFile.Enabled:=false;
            tabInputData.Enabled:=true;
        end;
   end;

end;

procedure TfmMCX.sgMediaEditingDone(Sender: TObject);
var
   grid: TStringGrid;
   val: Extended;
   ss: string;
   rowid, colid: integer;
begin
  grid:= Sender as TStringGrid;
  if(grid = nil) then exit;
  try
    try
        ss:=grid.Cells[grid.Col,grid.Row];
        rowid:=grid.Row;
        colid:=grid.Col;
        if(Length(ss)>0) then
            val := StrToFloat(ss);
    except
        raise Exception.Create('Input is not a number!');
    end;
  except
    On E : Exception do
    begin
      ShowMessage(E.Message);
      grid.Row:=rowid;
      grid.Col:=colid;
      grid.EditorMode:=true;
    end;
  end;
end;

procedure TfmMCX.sgMediaGetEditMask(Sender: TObject; ACol, ARow: Integer;
  var Value: string);
begin
end;

procedure TfmMCX.StaticText1DblClick(Sender: TObject);
begin
  mmOutput.Lines.Add(GridToJSONStruct(sgConfig));
end;

procedure TfmMCX.StaticText2Click(Sender: TObject);
begin

end;

procedure TfmMCX.StaticText2DblClick(Sender: TObject);
begin
  mmOutput.Lines.Add(GridToJSONArray(sgMedia));
end;

procedure TfmMCX.StaticText3DblClick(Sender: TObject);
begin
  mmOutput.Lines.Add(GridToJSONArray(sgDet));
end;

procedure TfmMCX.ToolButton14Click(Sender: TObject);
begin

end;

procedure TfmMCX.ValueListEditor1Click(Sender: TObject);
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
    Sleep(100);
end;

procedure TfmMCX.SaveTasksToIni(fname: string);
var
   inifile: TIniFile;
   i,j: integer;
begin
     DeleteFile(fname);
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
     AddLog(Format('Successfully loaded project %s. Please double click on session list to edit.',[fname]));
end;


procedure TfmMCX.VerifyInput;
var
    nthread, nblock: integer;
    radius,nphoton: extended;
    exepath: string;
begin
  try
    if(Length(edConfigFile.FileName)=0) then
        raise Exception.Create('Config file must be specified');
    if(not FileExists(edConfigFile.FileName)) then
        raise Exception.Create('Config file does not exist, please check the path');
    try
        nthread:=StrToInt(edThread.Text);
        nphoton:=StrToFloat(edPhoton.Text);
        radius:=StrToFloat(edBubble.Text);
        nblock:=StrToInt(edBlockSize.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread (-t), block (-T), photon (-n) and cache radius (-R)');
    end;
    if(nthread<512) then
       AddLog('Warning: using over 20000 threads (-t) can usually give you the best speed');
    if(nthread>10000) then
       AddLog('Warning: you can try Cached MCX to improve accuracy near the source');
    if(nphoton>1e9) then
       AddLog('Warning: you can increase respin number (-r) to get more photons');
    if(nblock<0) then
       raise Exception.Create('Thread block number (-T) can not be negative');
//    if(radius<0) then
//       raise Exception.Create('Cache radius (-R) can not be negative');

    exepath:=SearchForExe(CreateCmdOnly);
    if(exepath='') then
       raise Exception.Create('Can not find mcx executable in the search path');

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
    Result:=cmd;
end;

function TfmMCX.CreateCmd:string;
var
    nthread, nblock,gpuid,hitmax,seed,reseed: integer;
    bubbleradius,unitinmm,nphoton: extended;
    cmd: string;
begin
//    cmd:='"'+Config.MCXExe+'" ';
    cmd:=CreateCmdOnly;
    if(Length(edSession.Text)>0) then
       cmd:=cmd+' --session "'+Trim(edSession.Text)+'" ';
    if(Length(edConfigFile.FileName)>0) then
       cmd:=cmd+' --input "'+Trim(edConfigFile.FileName)
         +'" --root "'+ExcludeTrailingPathDelimiter(ExtractFilePath(edConfigFile.FileName))+'" ';
    try
        nthread:=StrToInt(edThread.Text);
        nphoton:=StrToFloat(edPhoton.Text);
        nblock:=StrToInt(edBlockSize.Text);
        bubbleradius:=StrToFloat(edBubble.Text);
        gpuid:=StrToInt(edGPUID.Text);
        unitinmm:=StrToFloat(edUnitInMM.Text);
        hitmax:=StrToInt(edDetectedNum.Text);
        seed:=StrToInt(edSeed.Text);
        reseed:=StrToInt(edReseed.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, block, photon and time gate settings');
    end;

    if(ckAutopilot.Checked) then begin
      cmd:=cmd+Format(' --gpu %d --autopilot 1 --photon %.0f --repeat %d --array %d --skipradius %f ',
        [gpuid,nphoton,edRespin.Value,grArray.ItemIndex,bubbleradius]);
    end else begin
      cmd:=cmd+Format(' --thread %d --blocksize %d --photon %.0f --repeat %d --array %d --skipradius %f ',
        [nthread,nblock,nphoton,edRespin.Value,grArray.ItemIndex,bubbleradius]);
    end;
    cmd:=cmd+Format(' --normalize %d --save2pt %d --reflect %d --savedet %d --maxdetphoton %d --unitinmm %f',
      [Integer(ckNormalize.Checked),Integer(ckSaveData.Checked),Integer(ckReflect.Checked),
      Integer(ckSaveDetector.Checked),hitmax,unitinmm]);
    if(seed<>0) then
      cmd:=cmd+Format(' --seed %d',[seed]);
    if(reseed <> 10000000) then
      cmd:=cmd+Format(' --reseed %d',[reseed]);
    if(ckSkipVoid.Checked) then
      cmd:=cmd+' --skipvoid 1';

    Result:=cmd;
    AddLog('Command:');
    AddLog(cmd);
end;

function TfmMCX.GridToJSONArray(grid:TStringGrid):string;
var
    i: integer;
    json: TStrings;
begin
  json := TStringList.Create;
  Result:='';
  try
      for i := grid.FixedRows to grid.RowCount - 1 do
      begin
          if (Length(grid.Cells[0,i])=0) then break;
          json.Add('['+grid.Rows[i].CommaText+']');
      end;
      json.Delimiter:=',';
      json.QuoteChar:=' ';
      Result:='['+json.DelimitedText+']';
  finally
      json.Free;
  end;
end;


function TfmMCX.GridToJSONStruct(grid:TStringGrid):string;
var
    i, j: integer;
    json: TStrings;
begin
  json := TStringList.Create;
  Result:='';
  try
      try
          for i := grid.FixedRows to grid.RowCount - 1 do
          begin
              if (Length(grid.Cells[0,i])=0) and (Length(grid.Cells[2,i])=0) then continue;
              //if(Length(grid.Cells[2,i])=0) then
              //   raise Exception.Create('Field '+grid.Cells[0,i]+'::'+grid.Cells[1,i]+' can not be empty');
              json.Add(''''+grid.Cells[0,i]+'::'+grid.Cells[1,i]+''': '+grid.Cells[2,i]);
          end;
      except
          On E : Exception do
              ShowMessage(E.Message);
      end;
      json.Delimiter:=',';
      json.QuoteChar:=' ';
      Result:='{'+json.DelimitedText+'}';
    finally
        json.Free;
    end;
end;

procedure TfmMCX.PanelToList2(node:TListItem);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    cg: TCheckBox;
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
        if(plSetting.Controls[i] is TCheckBox) then begin
           cg:=plSetting.Controls[i] as TCheckBox;
           idx:=MapList.IndexOf(cg.Hint);
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

