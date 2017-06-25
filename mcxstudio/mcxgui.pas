unit mcxgui;
{==============================================================================
    Monte Carlo eXtreme (MCX) Studio
-------------------------------------------------------------------------------
    Author: Qianqian Fang
    Email : q.fang at neu.edu
    Web   : http://mcx.space
    License: GNU General Public License version 3 (GPLv3)
===============================================================================}
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, process, FileUtil, TAGraph, SynEdit,
  SynHighlighterAny, SynHighlighterPerl, synhighlighterunixshellscript,
  LResources, Forms, Controls, Graphics, Dialogs, StdCtrls, Menus, ComCtrls,
  ExtCtrls, Spin, EditBtn, Buttons, ActnList, lcltype, AsyncProcess,
  Grids, CheckLst, inifiles, fpjson, jsonparser, mcxabout, mcxshape;

type

  { TfmMCX }

  TfmMCX = class(TForm)
    acEditShape: TActionList;
    ILJSON: TImageList;
    shapePrint: TAction;
    shapeEdit: TAction;
    shapeAddCylinder: TAction;
    shapeAddZSlabs: TAction;
    shapeAddYSlabs: TAction;
    shapeAddXSlabs: TAction;
    shapeAddZLayers: TAction;
    shapeAddYLayers: TAction;
    shapeAddXLayers: TAction;
    shapeAddUpperSpace: TAction;
    shapeAddSubgrid: TAction;
    shapeAddBox: TAction;
    shapeAddSphere: TAction;
    shapeAddGrid: TAction;
    shapeAddOrigin: TAction;
    shapeAddName: TAction;
    shapeDelete: TAction;
    shapeReset: TAction;
    Button1: TButton;
    ckSaveSeed: TCheckBox;
    edGPUID: TCheckListBox;
    ckAutopilot: TCheckBox;
    ckNormalize: TCheckBox;
    ckReflect: TCheckBox;
    ckSaveData: TCheckBox;
    ckSaveDetector: TCheckBox;
    ckSaveExit: TCheckBox;
    ckSaveRef: TCheckBox;
    ckSkipVoid: TCheckBox;
    ckSrcFrom0: TCheckBox;
    edBlockSize: TComboBox;
    edBubble: TEdit;
    edConfigFile: TFileNameEdit;
    edDetectedNum: TEdit;
    edGate: TSpinEdit;
    edPhoton: TEdit;
    edReseed: TEdit;
    edRespin: TSpinEdit;
    edSeed: TEdit;
    edSession: TEdit;
    edThread: TComboBox;
    edUnitInMM: TEdit;
    edWorkload: TEdit;
    grAdvSettings: TGroupBox;
    grArray: TRadioGroup;
    grBasic: TGroupBox;
    grGPU: TGroupBox;
    grSwitches: TGroupBox;
    HeaderControl1: THeaderControl;
    ImageList2: TImageList;
    Label1: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    Label14: TLabel;
    Label15: TLabel;
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
    rbUseFile: TRadioButton;
    rbUseDesigner: TRadioButton;
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
    tabVolumeDesigner: TTabSheet;
    tbtRun: TToolButton;
    tbtStop: TToolButton;
    tbtVerify: TToolButton;
    ToolBar1: TToolBar;
    ToolBar2: TToolBar;
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
    ToolButton20: TToolButton;
    ToolButton21: TToolButton;
    ToolButton22: TToolButton;
    ToolButton23: TToolButton;
    ToolButton24: TToolButton;
    ToolButton25: TToolButton;
    ToolButton26: TToolButton;
    ToolButton27: TToolButton;
    ToolButton28: TToolButton;
    ToolButton29: TToolButton;
    ToolButton3: TToolButton;
    ToolButton31: TToolButton;
    ToolButton32: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    ToolButton7: TToolButton;
    ToolButton8: TToolButton;
    ToolButton9: TToolButton;
    tvShapes: TTreeView;
    procedure grAdvSettingsClick(Sender: TObject);
    procedure grAdvSettingsDblClick(Sender: TObject);
    procedure grAdvSettingsEnter(Sender: TObject);
    procedure lvJobsChange(Sender: TObject; Item: TListItem; Change: TItemChange
      );
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoDefaultExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure mcxdoHelpExecute(Sender: TObject);
    procedure mcxdoHelpOptionsExecute(Sender: TObject);
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
    procedure miClearLogClick(Sender: TObject);
    procedure plOutputDockOver(Sender: TObject; Source: TDragDockObject; X,
      Y: Integer; State: TDragState; var Accept: Boolean);
    procedure pMCXReadData(Sender: TObject);
    procedure pMCXTerminate(Sender: TObject);
    procedure rbUseFileChange(Sender: TObject);
    procedure sgConfigClick(Sender: TObject);
    procedure sgConfigDblClick(Sender: TObject);
    procedure sgMediaEditingDone(Sender: TObject);
    procedure shapeAddBoxExecute(Sender: TObject);
    procedure shapeAddCylinderExecute(Sender: TObject);
    procedure shapeAddGridExecute(Sender: TObject);
    procedure shapeAddNameExecute(Sender: TObject);
    procedure shapeAddOriginExecute(Sender: TObject);
    procedure shapeAddSphereExecute(Sender: TObject);
    procedure shapeAddSubgridExecute(Sender: TObject);
    procedure shapeAddUpperSpaceExecute(Sender: TObject);
    procedure shapeAddXLayersExecute(Sender: TObject);
    procedure shapeAddXSlabsExecute(Sender: TObject);
    procedure shapeAddYLayersExecute(Sender: TObject);
    procedure shapeAddYSlabsExecute(Sender: TObject);
    procedure shapeAddZLayersExecute(Sender: TObject);
    procedure shapeAddZSlabsExecute(Sender: TObject);
    procedure shapePrintExecute(Sender: TObject);
    procedure shapeResetExecute(Sender: TObject);
    procedure shapeDeleteExecute(Sender: TObject);
    procedure StaticText1DblClick(Sender: TObject);
    procedure StaticText2DblClick(Sender: TObject);
    procedure ToolButton22Click(Sender: TObject);
    procedure tvShapesDeletion(Sender: TObject; Node: TTreeNode);
    procedure tvShapesEdited(Sender: TObject; Node: TTreeNode; var S: string);
    procedure tvShapesSelectionChanged(Sender: TObject);
  private
    { private declarations }
  public
    { public declarations }
    MapList, ConfigData, JSONstr : TStringList;
    JSONdata : TJSONData;
    function CreateCmd:string;
    function CreateCmdOnly:string;
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
    function CreateWorkFolder:string;
    procedure SaveJSONConfig(filename: string);
    function CheckListToStr(list: TCheckListBox) : string;
    function GridToStr(grid:TStringGrid) :string;
    procedure StrToGrid(str: string; grid:TStringGrid);
    procedure ShowJSONData(AParent : TTreeNode; Data : TJSONData);
    procedure AddShapesWindow(shapeid: string; defaultval: TStringList; node: TTreeNode);
    procedure AddShapes(shapeid: string; defaultval: string);
    procedure RebuildShapeJSON(root: TTreeNode);
  end;

var
  fmMCX: TfmMCX;
  ProfileChanged: Boolean;
  MaxWait: integer;
  TaskFile: string;




implementation

Const
  ImageTypeMap : Array[TJSONtype] of Integer =
//      jtUnknown, jtNumber, jtString, jtBoolean, jtNull, jtArray, jtObject
     (-1,8,9,7,6,5,4);
  JSONTypeNames : Array[TJSONtype] of string =
     ('Unknown','Number','String','Boolean','Null','Array','Object');

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
    ckb: TCheckListBox;
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
    end else if(Sender is TCheckListBox) then begin
       ckb:=Sender as TCheckListBox;
       idx:=MapList.IndexOf(ckb.Hint);
       if(idx>=0) then begin
           node.SubItems.Strings[idx]:=CheckListToStr(ckb);
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
          sbInfo.Panels[0].Text := 'Status: querying command line options';
          sbInfo.Tag:=-2;
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
      if(edGPUID.Items.Count>0) then edGPUID.Checked[0]:=true;
      edDetectedNum.Text:='10000000';
      edSeed.Text:='0';
      edReseed.Text:='10000000';
      rbUseDesigner.Checked:=true;
      sgMedia.RowCount:=1;
      sgMedia.RowCount:=129;
      sgMedia.Rows[1].CommaText:='0,0,1,1';
      sgDet.RowCount:=1;
      sgDet.RowCount:=129;
      sgConfig.ColCount:=2;
      sgConfig.ColCount:=3;
      sgConfig.Cols[2].CommaText:=ConfigData.CommaText;
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

procedure TfmMCX.grAdvSettingsClick(Sender: TObject);
begin
end;

procedure TfmMCX.grAdvSettingsDblClick(Sender: TObject);
var
    i: integer;
begin
     if(grAdvSettings.Tag<0) then begin
       for i:=20 to plSetting.ClientHeight do begin
            grAdvSettings.Height:=i;
            Sleep(100);
       end;
       grAdvSettings.Tag:=0;
     end else begin
       for i:=plSetting.ClientHeight downto 20 do begin
              grAdvSettings.Height:=i;
              Sleep(100);
       end;
       grAdvSettings.Tag:=-1;
     end;
end;

procedure TfmMCX.grAdvSettingsEnter(Sender: TObject);
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
          sbInfo.Panels[0].Text := 'Status: querying GPU';
          sbInfo.Tag:=-1;
          AddLog('"-- Printing GPU Information --"');
          mmOutput.Tag:=mmOutput.Lines.Count;
          pMCX.Execute;

          UpdateMCXActions(acMCX,'Run','');
    end;
end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          //pMCX.CommandLine:='du /usr/ --max-depth=1';
          pMCX.CommandLine:=CreateCmd;
          AddLog('"-- Executing MCX --"');
          mmOutput.Tag:=mmOutput.Lines.Count;
          pMCX.Execute;
          mcxdoStop.Enabled:=true;
          mcxdoRun.Enabled:=false;
          sbInfo.Panels[0].Text := 'Status: running MCX';
          sbInfo.Tag:=-10;
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
          AddLog('"-- Terminated MCX --"');
     end
end;

procedure TfmMCX.mcxdoVerifyExecute(Sender: TObject);
begin
    VerifyInput;
end;

procedure TfmMCX.FormCreate(Sender: TObject);
var
   i: integer;
   shaperoot: TTreeNode;
begin
    MapList:=TStringList.Create();
    MapList.Clear;
    for i:=1 to lvJobs.Columns.Count-1 do begin
        MapList.Add(lvJobs.Columns.Items[i].Caption);
    end;

    ConfigData:=TStringList.Create();
    ConfigData.Clear;
    ConfigData.CommaText:=sgConfig.Cols[2].CommaText;

    ProfileChanged:=false;
    if not (SearchForExe(CreateCmdOnly) = '') then begin
        mcxdoQuery.Enabled:=true;
        mcxdoHelpOptions.Enabled:=true;
    end;

    tvShapes.Items.BeginUpdate;
    tvShapes.Items.Clear;
    tvShapes.Items.EndUpdate;
    shaperoot:=tvShapes.Items.Add(nil,'Shapes');
    FreeAndNil(JSONdata);
    JSONdata:=GetJSON('[{"Grid":{"Tag":1,"Size":[60,60,60]}}]');
    ShowJSONData(shaperoot,JSONdata);
    tvShapes.FullExpand;
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
     Result := SearchForExe('google-chrome');  {KDE browser}
   if Result = '' then
     Result := SearchForExe('konqueror');  {GNOME browser}
   if Result = '' then
     Result := SearchForExe('epiphany');
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
     sbInfo.Tag:=0;
     sbInfo.Color := clBtnFace;
     AddLog('"-- Task completed --"');
     UpdateMCXActions(acMCX,'','Run');
end;

procedure TfmMCX.rbUseFileChange(Sender: TObject);
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

procedure TfmMCX.sgConfigClick(Sender: TObject);
begin
end;

procedure TfmMCX.sgConfigDblClick(Sender: TObject);
var
   sg: TStringGrid;
begin
   if(not (Sender is TStringGrid)) then exit;
   sg:=Sender as TStringGrid;
   if (sg.Row=1) and (sg.Col=2) then begin
         if(sg.Cells[sg.Col,sg.Row]='See Volume Designer...') then
              pcSimuEditor.ActivePage:=tabVolumeDesigner;
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
  UpdateMCXActions(acMCX,'','Work');
  UpdateMCXActions(acMCX,'','Run');
  mcxdoSave.Enabled:=true;
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

procedure TfmMCX.shapeAddBoxExecute(Sender: TObject);
begin
  AddShapes('Box','Tag=1|O=[30,30,30]|Size=[10,10,10]');
end;

procedure TfmMCX.shapeAddCylinderExecute(Sender: TObject);
begin
  AddShapes('Cylinder','Tag=1|C0=[30,30,0]|C1=[30,30,60]|R=5');
end;

procedure TfmMCX.AddShapesWindow(shapeid: string; defaultval: TStringList; node: TTreeNode);
var
   fmshape:TfmShapeEditor;
   ss: string;
begin
   if(node = nil) then exit;
   if(defaultval.Count=1) then begin
        ss:=InputBox('Edit Shape',shapeid, defaultval.Strings[0]);
        ss:=Trim(ss);
        if(Pos(',',ss)=0) then
            ss:='{"'+shapeid+'":"'+ss+'"}'
        else
            ss:='{"'+shapeid+'":'+ss+'}';
        ShowJSONData(node,GetJSON(ss));
        exit;
   end;

   fmshape:=TfmShapeEditor.Create(Application, defaultval);
   fmshape.Caption:='Add Shape: '+shapeid;
   if(Pos('Layer',shapeid)=2) or (Pos('Slab',shapeid)=2) then
        fmshape.plEditor.Options:=fmshape.plEditor.Options+[goAutoAddRows];
   if(fmshape.ShowModal= mrOK) then begin
        ss:=Trim(fmshape.JSON.DelimitedText);
        if(Pos('[',ss)=1) then
             ss:='{"'+shapeid+'":['+ss+']}'
        else
             ss:='{"'+shapeid+'":{'+ss+'}}';
        ShowJSONData(node,GetJSON(ss));
   end;
   fmshape.Free;
end;

procedure TfmMCX.AddShapes(shapeid: string; defaultval: string);
var
    fs: TStringList;
begin
  try
    fs:=TStringList.Create;
    fs.StrictDelimiter := true;
    fs.Delimiter:='|';
    fs.DelimitedText:=defaultval;
    AddShapesWindow(shapeid, fs, tvShapes.Items[0]);
  finally
    fs.Free;
  end;
end;

procedure TfmMCX.shapeAddGridExecute(Sender: TObject);
begin
     AddShapes('Grid','Tag=1|Size=[60,60,60]');
end;

procedure TfmMCX.shapeAddNameExecute(Sender: TObject);
begin
   AddShapes('Name','mcxdomain');
end;

procedure TfmMCX.shapeAddOriginExecute(Sender: TObject);
begin
   AddShapes('Origin','[0,0,0]');
end;

procedure TfmMCX.shapeAddSphereExecute(Sender: TObject);
begin
  AddShapes('Sphere','Tag=1|O=[30,30,30]|R=10');
end;

procedure TfmMCX.shapeAddSubgridExecute(Sender: TObject);
begin
  AddShapes('Subgrid','Tag=1|O=[30,30,30]|Size=[10,10,10]');
end;

procedure TfmMCX.shapeAddUpperSpaceExecute(Sender: TObject);
begin
  AddShapes('UpperSpace','Tag=1|Coef=[1,-1,0,0]');
end;

procedure TfmMCX.shapeAddXLayersExecute(Sender: TObject);
begin
   AddShapes('XLayers','Layer 1=[1,10,1]|Layer 2=[11,30,2]|Layer 3=[31,50,3]');
end;

procedure TfmMCX.shapeAddXSlabsExecute(Sender: TObject);
begin
  AddShapes('XSlabs','Tag=1|Bound=[1,10]');
end;

procedure TfmMCX.shapeAddYLayersExecute(Sender: TObject);
begin
   AddShapes('YLayers','Layer 1=[1,10,1]|Layer 2=[11,30,2]|Layer 3=[31,50,3]');
end;

procedure TfmMCX.shapeAddYSlabsExecute(Sender: TObject);
begin
   AddShapes('YSlabs','Tag=1|Bound=[1,10]');
end;

procedure TfmMCX.shapeAddZLayersExecute(Sender: TObject);
begin
  AddShapes('ZLayers','Layer 1=[1,10,1]|Layer 2=[11,30,2]|Layer 3=[31,50,3]');
end;

procedure TfmMCX.shapeAddZSlabsExecute(Sender: TObject);
begin
  AddShapes('ZSlabs','Tag=1|Bound=[1,10]');
end;

procedure TfmMCX.RebuildShapeJSON(root: TTreeNode);
var
     i: integer;
     jdata: TJSONData;
begin
     jdata:=GetJSON('{"Shapes": []}');
     for i:=0 to root.Count-1 do begin
         if(Assigned(root.Items[i].Data)) then
            TJSONArray(jdata.Items[0]).Add(TJSONObject(root.Items[i].Data));
     end;
     root.Data:=jdata;
end;

procedure TfmMCX.shapePrintExecute(Sender: TObject);
begin
   //AddLog(JSONdata.FormatJSON);
    if(tvShapes.Selected <> nil) then
       if(tvShapes.Selected=tvShapes.Items[0]) then begin
           RebuildShapeJSON(tvShapes.Selected);
           AddLog(TJSONData(tvShapes.Selected.Data).FormatJSON(AsJSONFormat));
       end else begin
           if(tvShapes.Selected.Data <> nil) then
               AddLog(TJSONData(tvShapes.Selected.Data).FormatJSON(AsJSONFormat));
       end;
end;



procedure TfmMCX.shapeResetExecute(Sender: TObject);
var
  ret:integer;
  shaperoot: TTreeNode;
begin
  ret:=Application.MessageBox('The current shape has not been saved, are you sure you want to clear?',
    'Confirm', MB_YESNOCANCEL);
  if (ret=IDYES) then begin
    tvShapes.Items.BeginUpdate;
    tvShapes.Items.Clear;
    tvShapes.Items.EndUpdate;
    shaperoot:=tvShapes.Items.Add(nil,'Shapes');
    FreeAndNil(JSONdata);
    JSONdata:=GetJSON('[{"Grid":{"Tag":1,"Size":[60,60,60]}}]');
    ShowJSONData(shaperoot,JSONdata);
    tvShapes.FullExpand;
  end;
  if (ret=IDCANCEL) then
       exit;
end;

procedure TfmMCX.shapeDeleteExecute(Sender: TObject);
begin
  if(tvShapes.Selected <> nil) then
       tvShapes.Selected.Delete;
end;

procedure TfmMCX.StaticText1DblClick(Sender: TObject);
begin
end;

procedure TfmMCX.StaticText2DblClick(Sender: TObject);
begin
    SaveJSONConfig('');
end;

procedure TfmMCX.ToolButton22Click(Sender: TObject);
begin

end;

procedure TfmMCX.tvShapesDeletion(Sender: TObject; Node: TTreeNode);
begin
  RebuildShapeJSON(tvShapes.Items[0]);
end;

procedure TfmMCX.tvShapesEdited(Sender: TObject; Node: TTreeNode; var S: string
  );
var
     val: extended;
begin
     if(Node.Parent= nil) then exit;

     if(Node.ImageIndex=ImageTypeMap[jtNumber]) then begin
         try
               val:=StrToFloat(S);
               if (Pos(Node.Parent.Text,'Layer ')=0 ) then begin
                    if(Frac(val)=0) then
                       TJSONArray(Node.Parent.Data).Integers[Node.Index]:=Round(val)
                    else
                       TJSONArray(Node.Parent.Data).Floats[Node.Index]:=val;
               end else begin
                    if(Frac(val)=0) then
                       TJSONArray(TJSONArray(Node.Parent.Data).Items[Node.Parent.Index]).Integers[Node.Index]:=Round(val)
                    else
                       TJSONArray(TJSONArray(Node.Parent.Data).Items[Node.Parent.Index]).Floats[Node.Index]:=val;
               end;

         except
               ShowMessage('The field must be a number');
               S:= Node.Text;
         end;
     end else if(Node.ImageIndex=ImageTypeMap[jtString]) then begin
          if(Length(S)=0) then begin
               ShowMessage('Input string can not be empty');
               S:=Node.Text;
               exit;
          end;

          if(Node.Parent <> nil) then
             TJSONArray(Node.Parent.Data).Strings[Node.Index]:=S;
     end;
end;

procedure TfmMCX.tvShapesSelectionChanged(Sender: TObject);
begin
    if(tvShapes.Selected <> nil) then  begin
       shapeDelete.Enabled:=tvShapes.Selected.Level=1;
       shapeEdit.Enabled:=tvShapes.Selected.Level=1;
       if(tvShapes.Selected.Count=0) then begin
           tvShapes.Options:=tvShapes.Options-[tvoReadOnly];
       end else begin
           tvShapes.Options:=tvShapes.Options+[tvoReadOnly];
       end;
    end;
end;

function TfmMCX.GetMCXOutput (outputstr: string) : string;
var
    Buffer: string;
    BytesAvailable: DWord;
    BytesRead:LongInt;
    list: TStringList;
    i, idx, total, namepos: integer;
    gpuname, ss: string;
begin
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
    if(sbInfo.Tag=-1) then begin
        list:=TStringList.Create;
        list.StrictDelimiter := true;
        list.Delimiter:=AnsiChar(#10);
        list.DelimitedText:=Result;
        for i:=0 to list.Count-1 do begin
          ss:= list.Strings[i];
          if(sscanf(ss,'Device %d of %d:%s', [@idx, @total, @gpuname])=3) then
          begin
                 if(idx=1) then
                     edGPUID.Items.Clear;
                 namepos := Pos(gpuname, ss);
                 edGPUID.Items.Add(Trim(copy(ss, namepos, Length(ss)-namepos)));
          end;
        end;
        if(edGPUID.Items.Count>0) then
            edGPUID.Checked[0]:=true;
    end;
    Sleep(100);
    Result:= outputstr + Result;
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
    if rbUseFile.Checked and (Length(edConfigFile.FileName)=0) then
        raise Exception.Create('Config file must be specified');
    if rbUseFile.Checked and (not FileExists(edConfigFile.FileName)) then
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
procedure TfmMCX.SaveJSONConfig(filename: string);
var
    nthread, nblock,hitmax,seed,reseed, i: integer;
    bubbleradius,unitinmm,nphoton: extended;
    gpuid, section, key, val: string;
    json, jobj, jmedium, jdet, jforward, joptode : TJSONObject;
    jdets, jmedia: TJSONArray;
    jsonlist: TStringList;
begin
  try
      nthread:=StrToInt(edThread.Text);
      nphoton:=StrToFloat(edPhoton.Text);
      nblock:=StrToInt(edBlockSize.Text);
      bubbleradius:=StrToFloat(edBubble.Text);
      gpuid:=CheckListToStr(edGPUID);
      unitinmm:=StrToFloat(edUnitInMM.Text);
      hitmax:=StrToInt(edDetectedNum.Text);
      seed:=StrToInt(edSeed.Text);
      reseed:=StrToInt(edReseed.Text);
  except
      raise Exception.Create('Invalid numbers: check the values for thread, block, photon and time gate settings');
  end;

  json:=TJSONObject.Create;

  if(json.Find('Session') = nil) then
      json.Objects['Session']:=TJSONObject.Create;
  jobj:= json.Objects['Session'];
  jobj.Floats['Photons']:=nphoton;
  jobj.Integers['RNGSeed']:=seed;
  jobj.Strings['ID']:=edSession.Text;
  jobj.Integers['DoMismatch']:=Integer(ckReflect.Checked);
  jobj.Integers['DoNormalize']:=Integer(ckNormalize.Checked);
  jobj.Integers['DoPartialPath']:=Integer(ckSaveDetector.Checked);
  jobj.Integers['DoSaveExit']:=Integer(ckSaveExit.Checked);
  jobj.Integers['DoSaveSeed']:=Integer(ckSaveSeed.Checked);
  jobj.Integers['DoSaveRef']:=Integer(ckSaveRef.Checked);
  jobj.Integers['ReseedLimit']:=reseed;
  //jobj.Strings['OutputType']:=edOutputType.Text;

  if(json.Find('Domain') = nil) then
      json.Objects['Domain']:=TJSONObject.Create;
  jobj:= json.Objects['Domain'];
  jobj.Integers['OriginType']:=Integer(ckSrcFrom0.Checked);
  jobj.Floats['LengthUnit']:=unitinmm;

  jmedia:=TJSONArray.Create;
  for i := sgMedia.FixedRows to sgMedia.RowCount - 1 do
  begin
          if (Length(sgMedia.Cells[0,i])=0) then break;
          jmedium:=TJSONObject.Create;
          jmedium.Add('mua',sgMedia.Cells[0,i]);
          jmedium.Add('mus',sgMedia.Cells[1,i]);
          jmedium.Add('g',sgMedia.Cells[2,i]);
          jmedium.Add('n',sgMedia.Cells[3,i]);
          jmedia.Add(jmedium);
  end;
  if(jmedia.Count>0) then
     jobj.Arrays['Media']:=jmedia;

  jdets:=TJSONArray.Create;
  for i := sgDet.FixedRows to sgDet.RowCount - 1 do
  begin
          if (Length(sgDet.Cells[0,i])=0) then break;
          jdet:=TJSONObject.Create;
          jdet.Arrays['Pos']:=TJSONArray.Create;
          jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[0,i]));
          jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[1,i]));
          jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[2,i]));
          jdet.Add('R',sgDet.Cells[3,i]);
          jdets.Add(jdet);
  end;
  if(json.Find('Optode') = nil) then
     json.Objects['Optode']:=TJSONObject.Create;

  joptode:=json.Objects['Optode'];

  if(jdets.Count>0) then
      joptode.Arrays['Detector']:=jdets;
  joptode.Objects['Source']:=TJSONObject.Create;

  jforward:=TJSONObject.Create;
  for i := sgConfig.FixedRows to sgConfig.RowCount - 1 do
  begin
          if(Length(sgConfig.Cells[0,i])=0) then break;
          val:=sgConfig.Cells[2,i];
          if(Length(val)=0) then continue;
          section:= sgConfig.Cells[0,i];
          key:=sgConfig.Cells[1,i];
          if(section = 'Forward') then begin
              jforward.Floats[key]:=StrToFloat(val);
          end else if(section = 'Session') then begin
              json.Objects['Session'].Strings[key]:=val;
          end else if(section = 'Domain') then begin
              if (key = 'VolumeFile') and (val='See Volume Designer...') then begin
                  RebuildShapeJSON(tvShapes.Items[0]);
                  json.Objects['Shapes']:=TJSONObject(TJSONObject(tvShapes.Items[0].Data).Items[0]);
              end else begin
                  json.Objects['Domain'].Objects[key]:=TJSONObject(GetJSON(val));
              end;
          end else if(section = 'Optode.Source') then begin
              if (key = 'Type') then begin
                  joptode.Objects['Source'].Strings[key]:=val;
              end else begin
                  joptode.Objects['Source'].Objects[key]:=TJSONObject(GetJSON(val));
              end;
          end;
  end;
  json.Objects['Forward']:=jforward;

  AddLog(json.FormatJSON);

  if(Length(filename)>0) then begin
      jsonlist:=TStringList.Create;
      jsonlist.Text:=json.FormatJSON;
      try
          jsonlist.SaveToFile(filename);
      finally
          jsonlist.Free;
      end;
  end;
end;

function TfmMCX.CreateWorkFolder : string;
var
    path: string;
begin
    path:=ExtractFileDir(Application.ExeName)
       +DirectorySeparator+'..'+DirectorySeparator+'sessions'+DirectorySeparator+edSession.Text;
    Result:=path;
    try
      if(not DirectoryExists(path)) then
           if( not ForceDirectories(path) ) then
               raise Exception.Create('Can not create session output folder');
    except
      On E : Exception do
          ShowMessage(E.Message);
    end;
    AddLog(Result);
end;

function TfmMCX.CheckListToStr(list: TCheckListBox) : string;
var
    i: integer;
begin
    Result:='';
     for i:=0 to list.Items.Count-1 do begin
          Result:=Result + IntToStr(Integer(list.Checked[i]));
     end;
end;

function TfmMCX.CreateCmd:string;
var
    nthread, nblock,hitmax,seed,reseed: integer;
    bubbleradius,unitinmm,nphoton: extended;
    cmd, jsonfile, gpuid: string;
    shellscript: TStringList;
begin
//    cmd:='"'+Config.MCXExe+'" ';
    cmd:=CreateCmdOnly;
    if(Length(edSession.Text)>0) then
       cmd:=cmd+' --session "'+Trim(edSession.Text)+'" ';
    if rbUseFile.Checked and (Length(edConfigFile.FileName)>0) then
    begin
       cmd:=cmd+' --input "'+Trim(edConfigFile.FileName)
         +'" --root "'+ExcludeTrailingPathDelimiter(ExtractFilePath(edConfigFile.FileName))+'" ';
    end else begin
        jsonfile:=CreateWorkFolder+DirectorySeparator+Trim(edSession.Text)+'.json';
        SaveJSONConfig(jsonfile);
        cmd:=cmd+' --input "'+Trim(jsonfile)
          +'" --root "'+ExcludeTrailingPathDelimiter(ExtractFilePath(jsonfile))+'" ';

    end;
    try
        nthread:=StrToInt(edThread.Text);
        nphoton:=StrToFloat(edPhoton.Text);
        nblock:=StrToInt(edBlockSize.Text);
        bubbleradius:=StrToFloat(edBubble.Text);
        gpuid:=CheckListToStr(edGPUID);
        unitinmm:=StrToFloat(edUnitInMM.Text);
        hitmax:=StrToInt(edDetectedNum.Text);
        seed:=StrToInt(edSeed.Text);
        reseed:=StrToInt(edReseed.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread, block, photon and time gate settings');
    end;

    if(ckAutopilot.Checked) then begin
      cmd:=cmd+' --gpu '+gpuid+ Format(' --autopilot 1 --photon %.0f --repeat %d --array %d --skipradius %f ',
        [nphoton,edRespin.Value,grArray.ItemIndex,bubbleradius]);
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

    if(Length(jsonfile)>0) then begin
         shellscript:=TStringList.Create;
         shellscript.Add('#!/bin/sh');
         shellscript.Add(cmd);
         shellscript.SaveToFile(ChangeFileExt(jsonfile,'.sh'));
         shellscript.SaveToFile(ChangeFileExt(jsonfile,'.bat'));
         shellscript.Free;
    end;
    Result:=cmd;
    AddLog('Command:');
    AddLog(cmd);
end;

function TfmMCX.GridToStr(grid:TStringGrid):string;
var
    i: integer;
    json: TStrings;
begin
  json := TStringList.Create;
  Result:='';
  try
      try
          for i := grid.FixedRows to grid.RowCount - 1 do
          begin
              if (Length(grid.Cells[0,i])=0) and (Length(grid.Cells[2,i])=0) then break;
              json.Add(grid.Rows[i].CommaText);
          end;
      except
          On E : Exception do
              ShowMessage(E.Message);
      end;
      json.Delimiter:='|';
      //json.QuoteChar:=' ';
      Result:=json.DelimitedText;
    finally
        json.Free;
    end;
end;


procedure TfmMCX.StrToGrid(str: string; grid:TStringGrid);
var
    i, j: integer;
    json: TStrings;
begin
  json := TStringList.Create;
  json.Delimiter:='|';
  json.DelimitedText:=str;

  try
      try
          if(grid.RowCount < json.Count) then
              grid.RowCount:= json.Count;
          for i := 0 to json.Count do begin
              grid.Rows[i].CommaText:=json.Strings[i];
          end;
      except
          On E : Exception do
              ShowMessage(E.Message);
      end;
    finally
        json.Free;
    end;
end;
procedure TfmMCX.ShowJSONData(AParent : TTreeNode; Data : TJSONData);

Var
  N,N2 : TTreeNode;
  I : Integer;
  D : TJSONData;
  C : String;
  S : TStringList;
begin
  N:=nil;
  if Assigned(Data) then
    begin
    case Data.JSONType of
      jtArray,
      jtObject:
        begin
        if (Data.JSONType=jtArray) then begin
          C:='Array (%d elements)';
        end else begin
           C:='Object (%d members)';
        end;
        //N:=AParent.TreeView.Items.AddChild(AParent,Format(C,[Data.Count]));
        if(N =nil) then N:=AParent;
        S:=TstringList.Create;
        try
          for I:=0 to Data.Count-1 do
            if Data.JSONtype=jtArray then
              S.AddObject('_ARRAY_',Data.items[i])
            else
              S.AddObject(TJSONObject(Data).Names[i],Data.items[i]);
          for I:=0 to S.Count-1 do
            begin
            if(not (S[i] = '_ARRAY_')) then begin
               N2:=AParent.TreeView.Items.AddChild(N,S[i]);
            end else begin
               N2:=N;
            end;
            D:=TJSONData(S.Objects[i]);
            N2.ImageIndex:=ImageTypeMap[D.JSONType];
            N2.SelectedIndex:=ImageTypeMap[D.JSONType];
            N2.Data:=Data;
            ShowJSONData(N2,D);
            end
        finally
          S.Free;
        end;
        end;
      jtNull:
        N:=AParent.TreeView.Items.AddChild(AParent,'null');
    else
      N:=AParent.TreeView.Items.AddChild(AParent,Data.AsString);
    end;
    If Assigned(N) then
      begin
      N.ImageIndex:=ImageTypeMap[Data.JSONType];
      N.SelectedIndex:=ImageTypeMap[Data.JSONType];
      //N.Data:=Data;
      end;
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
    ckb: TCheckListBox;
    sg: TStringGrid;
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
        if(plSetting.Controls[i] is TCheckListBox) then begin
           ckb:=plSetting.Controls[i] as TCheckListBox;
           idx:=MapList.IndexOf(ckb.Hint);
           if(idx>=0) then node.SubItems.Strings[idx]:=CheckListToStr(ckb);
           continue;
        end;
        except
        end;
    end;
    for i:=0 to tabInputData.ControlCount-1 do
    begin
        try
          if(tabInputData.Controls[i] is TStringGrid) then begin
             sg:=tabInputData.Controls[i] as TStringGrid;
             idx:=MapList.IndexOf(sg.Hint);
             if(idx>=0) then node.SubItems.Strings[idx]:=GridToStr(sg);
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
    ckb: TCheckListBox;
    sg: TStringGrid;
    fed:TFileNameEdit;
    i,j,idx: integer;
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
        if(plSetting.Controls[i] is TCheckListBox) then begin
           ckb:=plSetting.Controls[i] as TCheckListBox;
           idx:=MapList.IndexOf(ck.Hint);
           if(idx>=0) then begin
               ckb.Items.Clear;
               for j:=1 to Length(node.SubItems.Strings[idx]) do begin
                   ckb.Items.Add('GPU#'+IntToStr(j));
                   if(node.SubItems.Strings[idx][j]='1') then
                       ckb.Checked[j]:=true;
               end;
           end;
           continue;
        end;
    end;
    for i:=0 to tabInputData.ControlCount-1 do
    begin
        try
          if(tabInputData.Controls[i] is TStringGrid) then begin
             sg:=tabInputData.Controls[i] as TStringGrid;
             idx:=MapList.IndexOf(sg.Hint);
             if(idx>=0) then
                 StrToGrid(node.SubItems.Strings[idx],sg);
             continue;
          end;
        finally
        end;
    end;
end;
initialization
  {$I mcxgui.lrs}
  {$I mcxdefaultinput.lrs}
end.

