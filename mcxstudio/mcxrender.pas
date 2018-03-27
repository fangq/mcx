unit mcxrender;

{==============================================================================
    Monte Carlo eXtreme (MCX) Studio - Domain Renderer
-------------------------------------------------------------------------------
    Author: Qianqian Fang
    Email : q.fang at neu.edu
    Web   : http://mcx.space
    License: GNU General Public License version 3 (GPLv3)
===============================================================================}

interface

uses
  SysUtils, Classes, Graphics, Controls, Forms, Dialogs, GLScene, GLObjects,
  ExtCtrls, ComCtrls, ActnList, ExtDlgs, SynEdit, SynHighlighterJScript,
  synhighlighterunixshellscript, GLBehaviours, GLTexture, GLVectorGeometry,
  GLLCLViewer, GLGeomObjects, GLCoordinates, GLCrossPlatform, GLGraphics,
  GLMaterial, GLColor, GLState, GLSkydome, GLMesh, Types, strutils, fpjson,
  jsonparser;

type

  { TfmDomain }

  TfmDomain = class(TForm)
    acShapeRender: TActionList;
    acResetCamera: TAction;
    acHideBBX: TAction;
    acRender: TAction;
    acSaveImage: TAction;
    acExit: TAction;
    acLoadJSON: TAction;
    acSaveJSON: TAction;
    glCanvas: TGLSceneViewer;
    glDomain: TGLCube;
    glLight2: TGLLightSource;
    glOrigin: TGLPoints;
    glShape: TGLScene;
    glCamera: TGLCamera;
    glLight1: TGLLightSource;
    glShapes: TGLDummyCube;
    ImageList3: TImageList;
    dlOpenFile: TOpenDialog;
    plEditor: TPanel;
    dlSaveScreen: TSavePictureDialog;
    dlSaveFile: TSaveDialog;
    Splitter1: TSplitter;
    mmShapeJSON: TSynEdit;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    glSpace: TGLDummyCube;
    ToolBar1: TToolBar;
    ToolButton1: TToolButton;
    btPin: TToolButton;
    ToolButton11: TToolButton;
    ToolButton2: TToolButton;
    ToolButton3: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    ToolButton7: TToolButton;
    ToolButton8: TToolButton;
    ToolButton9: TToolButton;
    procedure acExitExecute(Sender: TObject);
    procedure acHideBBXExecute(Sender: TObject);
    procedure acLoadJSONExecute(Sender: TObject);
    procedure acRenderExecute(Sender: TObject);
    procedure acResetCameraExecute(Sender: TObject);
    procedure acSaveImageExecute(Sender: TObject);
    procedure acSaveJSONExecute(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure glCanvasMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure glCanvasMouseMove(Sender: TObject; Shift: TShiftState; X,
      Y: Integer);
    procedure glCanvasMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    procedure AddSphere(jobj: TJSONData);
    procedure AddOrigin(jobj: TJSONData);
    procedure AddGrid(jobj: TJSONData);
    procedure AddCylinder(jobj: TJSONData);
    procedure AddBox(jobj: TJSONData; isbox: Boolean);
    procedure AddLayers(jobj: TJSONData; dim: integer);
    procedure AddSlabs(jobj: TJSONData; dim: integer);
    procedure AddName(jobj: TJSONObject);
    procedure AddSource(jobj: TJSONData);
    procedure AddDiskSource(jobj: TJSONData);
    procedure AddPlanarSource(jobj: TJSONData; isorth: boolean=false);
    procedure AddPattern3DSource(jobj: TJSONData);
    procedure AddDetector(jobj: TJSONData);
    procedure plEditorMouseEnter(Sender: TObject);
    procedure plEditorMouseLeave(Sender: TObject);
    procedure ShowJSON(root: TJSONData; rootstr: string);
    procedure LoadJSONShape(shapejson: AnsiString);
    procedure Splitter1Moved(Sender: TObject);
  private
    mdx, mdy : Integer;
    editorwidth: integer;
    JSONdata : TJSONData;
    colormap: array [0..1023,0..2] of extended;
  public

  end;

var
  fmDomain: TfmDomain;

implementation

{$R *.lfm}

procedure TfmDomain.glCanvasMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
begin
  glCamera.AdjustDistanceToTarget(Power(1.1, WheelDelta/1200.0));
end;

procedure TfmDomain.LoadJSONShape(shapejson: AnsiString);
begin
    FreeAndNil(JSONData);
    glSpace.DeleteChildren;
    glCamera.TargetObject:=glDomain;
    JSONData:=GetJSON(shapejson);
end;

procedure TfmDomain.Splitter1Moved(Sender: TObject);
begin
    editorwidth:=plEditor.Width;
    glCamera.TargetObject:=glDomain;
end;

procedure TfmDomain.AddGrid(jobj: TJSONData);
var
     objtag: integer;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('Size')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Grid shape construct', mtError, [mbOK],0);
        exit;
     end;

     glDomain.DeleteChildren; // grid object reset the domain

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;

     //obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);

     data:=TJSONArray(jobj.FindPath('Size'));
     glDomain.CubeWidth:=data.Items[0].AsFloat;
     glDomain.CubeDepth:=data.Items[1].AsFloat;
     glDomain.CubeHeight:=data.Items[2].AsFloat;
     glDomain.Position.X:=glDomain.CubeWidth*0.5;
     glDomain.Position.Y:=glDomain.CubeDepth*0.5;
     glDomain.Position.Z:=glDomain.CubeHeight*0.5;
end;

procedure TfmDomain.AddName(jobj: TJSONObject);
begin
     Caption:='MCX Domain Renderer ('+jobj.Strings['Name']+')';
end;

procedure TfmDomain.AddOrigin(jobj: TJSONData);
var
     obj: TGLPoints;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if not (jobj is TJSONArray) then begin
        MessageDlg('Warning', 'Malformed JSON Origin shape construct', mtError, [mbOK],0);
        exit;
     end;

     //obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);

     data:=TJSONArray(jobj);

     obj:=TGLPoints.Create(Self);
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;
     obj.Style:=psSquareAdditive;
     obj.Size:=20;

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddBox(jobj: TJSONData; isbox: Boolean);
var
     objtag: integer;
     obj: TGLCube;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('Size')=nil) or (jobj.FindPath('O')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Box shape construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLCube.Create(Self);

     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;
     obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     obj.Material.BlendingMode:=bmTransparency;

     data:=TJSONArray(jobj.FindPath('O'));
     obj.Position.X:=data.Items[0].AsFloat+Integer(isbox)*0.5;
     obj.Position.Y:=data.Items[1].AsFloat+Integer(isbox)*0.5;
     obj.Position.Z:=data.Items[2].AsFloat+Integer(isbox)*0.5;

     data:=TJSONArray(jobj.FindPath('Size'));
     obj.CubeWidth:=data.Items[0].AsFloat;
     obj.CubeDepth:=data.Items[1].AsFloat;
     obj.CubeHeight:=data.Items[2].AsFloat;

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddSphere(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLSphere;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('R')=nil) or (jobj.FindPath('O')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Sphere shape construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLSphere.Create(Self);
     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;
     obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     obj.Material.BlendingMode:=bmTransparency;

     obj.Radius:=jobj.FindPath('R').AsFloat;

     data:=TJSONArray(jobj.FindPath('O'));
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;
     obj.Slices:=64;

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddCylinder(jobj: TJSONData);
var
     objtag: integer;
     x,y,z: extended;
     obj: TGLCylinder;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('C0')=nil) or (jobj.FindPath('C1')=nil) or (jobj.FindPath('R')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Cylinder shape construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLCylinder.Create(Self);
     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);
     obj.Alignment:=caBottom;

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;
     obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     obj.Material.BlendingMode:=bmTransparency;

     data:=TJSONArray(jobj.FindPath('C0'));
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;

     data:=TJSONArray(jobj.FindPath('C1'));
     x:=data.Items[0].AsFloat;
     y:=data.Items[1].AsFloat;
     z:=data.Items[2].AsFloat;

     obj.Height:=sqrt((x-obj.Position.X)*(x-obj.Position.X)+
                      (y-obj.Position.Y)*(y-obj.Position.Y)+
                      (z-obj.Position.Z)*(z-obj.Position.Z));

     obj.Up.X:=(x-obj.Position.X)/obj.Height;
     obj.Up.Y:=(y-obj.Position.Y)/obj.Height;
     obj.Up.Z:=(z-obj.Position.Z)/obj.Height;

     obj.BottomRadius:=jobj.FindPath('R').AsFloat;
     obj.TopRadius:=obj.BottomRadius;

     obj.Slices:=64;

     glSpace.AddChild(obj);
end;
procedure TfmDomain.AddLayers(jobj: TJSONData; dim: integer);
var
     objtag, i: integer;
     obj: TGLCube;
     data: TJSONArray;
     elem: TJSONData;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if not (jobj is TJSONArray) then begin
        MessageDlg('Warning', 'Malformed JSON ?Layers shape construct', mtError, [mbOK],0);
        exit;
     end;

     data:=TJSONArray(jobj);
     for i:=0 to jobj.Count-1 do begin
       obj:=TGLCube.Create(Self);
       if (data.Items[i].Count = 0) then begin
           elem:=data;
       end else begin
           elem:=data.Items[i];
       end;
       if (elem.Count <> 3) then begin
          MessageDlg('Warning', 'Malformed JSON ?Layers shape element', mtError, [mbOK],0);
          exit;
       end;

       objtag:=elem.Items[2].AsInteger mod 1024;

       obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
       //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
       obj.Material.BlendingMode:=bmTransparency;

       data:=TJSONArray(jobj);

       obj.Up.SetVector(0,0,1);
       obj.Direction.SetVector(0,1,0);
       if(dim=1) then begin
             obj.CubeWidth:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;
             obj.CubeDepth:=glDomain.CubeDepth;
             obj.CubeHeight:=glDomain.CubeHeight;

             obj.Position.X:=elem.Items[0].AsFloat-1+obj.CubeWidth*0.5;
             obj.Position.Y:=obj.CubeDepth*0.5;
             obj.Position.Z:=obj.CubeHeight*0.5;
       end else if (dim=2) then begin
             obj.CubeWidth:=glDomain.CubeWidth;
             obj.CubeDepth:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;
             obj.CubeHeight:=glDomain.CubeHeight;

             obj.Position.X:=obj.CubeWidth*0.5;
             obj.Position.Y:=elem.Items[0].AsFloat-1+obj.CubeDepth*0.5;
             obj.Position.Z:=obj.CubeHeight*0.5;
       end else if(dim=3) then begin
             obj.CubeWidth:=glDomain.CubeWidth;
             obj.CubeDepth:=glDomain.CubeDepth;
             obj.CubeHeight:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;

             obj.Position.X:=obj.CubeWidth*0.5;
             obj.Position.Y:=obj.CubeDepth*0.5;
             obj.Position.Z:=elem.Items[0].AsFloat-1+obj.CubeHeight*0.5;
       end;

       glSpace.AddChild(obj);
       if (data.Items[i].Count = 0) then exit;
     end;
end;

procedure TfmDomain.AddSlabs(jobj: TJSONData; dim: integer);
var
     objtag, i: integer;
     obj: TGLCube;
     data: TJSONArray;
     elem: TJSONData;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('Bound')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON ?Slabs shape construct', mtError, [mbOK],0);
        exit;
     end;

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;

     jobj:=TJSONArray(jobj.FindPath('Bound'));
     for i:=0 to jobj.Count-1 do begin
       data:=TJSONArray(jobj.Items[i]);

       if (data.Count = 0) then begin
           elem:=jobj;
       end else begin
           elem:=data.Items[i];
       end;

       obj:=TGLCube.Create(Self);
       if (elem.Count <> 2) then begin
          MessageDlg('Warning', 'Malformed JSON ?Slabs shape element', mtError, [mbOK],0);
          exit;
       end;

       obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
       //obj.Material.FrontProperties.Emission.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
       obj.Material.BlendingMode:=bmTransparency;

       obj.Up.SetVector(0,0,1);
       obj.Direction.SetVector(0,1,0);
       if(dim=1) then begin
             obj.CubeWidth:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;
             obj.CubeDepth:=glDomain.CubeDepth;
             obj.CubeHeight:=glDomain.CubeHeight;

             obj.Position.X:=elem.Items[0].AsFloat-1+obj.CubeWidth*0.5;
             obj.Position.Y:=obj.CubeDepth*0.5;
             obj.Position.Z:=obj.CubeHeight*0.5;
       end else if (dim=2) then begin
             obj.CubeWidth:=glDomain.CubeWidth;
             obj.CubeDepth:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;
             obj.CubeHeight:=glDomain.CubeHeight;

             obj.Position.X:=obj.CubeWidth*0.5;
             obj.Position.Y:=elem.Items[0].AsFloat-1+obj.CubeDepth*0.5;
             obj.Position.Z:=obj.CubeHeight*0.5;
       end else if(dim=3) then begin
             obj.CubeWidth:=glDomain.CubeWidth;
             obj.CubeDepth:=glDomain.CubeDepth;
             obj.CubeHeight:=elem.Items[1].AsFloat-elem.Items[0].AsFloat+1;

             obj.Position.X:=obj.CubeWidth*0.5;
             obj.Position.Y:=obj.CubeDepth*0.5;
             obj.Position.Z:=elem.Items[0].AsFloat-1+obj.CubeHeight*0.5;
       end;
       glSpace.AddChild(obj);
       if (data.Count = 0) then exit;
     end;
end;

procedure TfmDomain.AddPlanarSource(jobj: TJSONData; isorth: boolean=false);
var
     objtag: integer;
     obj: TGLMesh;
     v0,v1,v2: TAffineVector;
     data,data1,data2,dir: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Param1')=nil) or (jobj.FindPath('Param2')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Disk Source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLMesh.Create(Self);

     obj.Up.SetVector(0,1,0);
     obj.Direction.SetVector(0,0,1);

     data:=TJSONArray(jobj.FindPath('Pos'));
     data1:=TJSONArray(jobj.FindPath('Param1'));
     data2:=TJSONArray(jobj.FindPath('Param2'));
     dir:=TJSONArray(jobj.FindPath('Dir'));

     obj.Mode:=mmQuads;
     obj.VertexMode:=vmV;

     obj.Vertices.Clear;

     v0:= AffineVectorMake(data.Items[0].AsFloat, data.Items[1].AsFloat, data.Items[2].AsFloat);
     v1:= AffineVectorMake(data1.Items[0].AsFloat, data1.Items[1].AsFloat, data1.Items[2].AsFloat);
     v2:= AffineVectorMake(data2.Items[0].AsFloat, data2.Items[1].AsFloat, data2.Items[2].AsFloat);

     if(isorth) then
         v2:=VectorCrossProduct(AffineVectorMake(dir.Items[0].AsFloat, dir.Items[1].AsFloat, dir.Items[2].AsFloat),v1);

     obj.Vertices.AddVertex(v0,nullvector,clrYellow);
     obj.Vertices.AddVertex(VectorAdd(v0,v1),nullvector,clrYellow);
     obj.Vertices.AddVertex(VectorAdd(v0,VectorAdd(v1,v2)),nullvector,clrYellow);
     obj.Vertices.AddVertex(VectorAdd(v0,v2),nullvector,clrYellow);

     obj.Material.FrontProperties.Diffuse.SetColor(1.0,1.0,0.0,1);
     obj.Material.BackProperties.Diffuse.SetColor(1.0,1.0,0.0,1);

     glSpace.AddChild(obj);
end;
procedure TfmDomain.AddDiskSource(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLDisk;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Param1')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Disk Source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLDisk.Create(Self);

     obj.Up.SetVector(0,0,1);

     data:=TJSONArray(jobj.FindPath('Pos'));
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;
     obj.Material.FrontProperties.Diffuse.SetColor(1.0,1.0,0.0,1);
     obj.Material.BackProperties.Diffuse.SetColor(1.0,1.0,0.0,1);
     //obj.NormalDirection:=ndInside;

     data:=TJSONArray(jobj.FindPath('Param1'));
     obj.OuterRadius:=data.Items[0].AsFloat;

     data:=TJSONArray(jobj.FindPath('Dir'));
     obj.Direction.SetVector(data.Items[0].AsFloat,data.Items[1].AsFloat,data.Items[2].AsFloat);

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddPattern3DSource(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLCube;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Param1')=nil ) then begin
        MessageDlg('Warning', 'Malformed JSON Pattern3D source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLCube.Create(Self);

     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);

     obj.Material.FrontProperties.Diffuse.SetColor(1.0,1.0,0.0,0.5);
     obj.Material.BlendingMode:=bmTransparency;

     data:=TJSONArray(jobj.FindPath('Param1'));
     obj.CubeWidth:=data.Items[0].AsFloat;
     obj.CubeDepth:=data.Items[1].AsFloat;
     obj.CubeHeight:=data.Items[2].AsFloat;

     data:=TJSONArray(jobj.FindPath('Pos'));
     obj.Position.X:=data.Items[0].AsFloat+obj.CubeWidth*0.5;
     obj.Position.Y:=data.Items[1].AsFloat+obj.CubeDepth*0.5;
     obj.Position.Z:=data.Items[2].AsFloat+obj.CubeHeight*0.5;

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddSource(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLPoints;
     dir: TGLArrowLine;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Pos')=nil) or (jobj.FindPath('Dir')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLPoints.Create(Self);

     obj.Up.SetVector(0,0,1);

     data:=TJSONArray(jobj.FindPath('Pos'));
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;
     obj.Size:=4;
     obj.Style:=psRound;
     obj.Material.FrontProperties.Diffuse.SetColor(1.0,1.0,0.0,0.5);

     glSpace.AddChild(obj);

     dir:=TGLArrowLine.Create(Self);
     data:=TJSONArray(jobj.FindPath('Dir'));
     dir.Position:=obj.Position;
     dir.Direction.SetVector(data.Items[0].AsFloat,data.Items[1].AsFloat,data.Items[2].AsFloat);
     dir.Height:=5;
     dir.TopRadius:=1;
     dir.BottomRadius:=dir.TopRadius;
     dir.TopArrowHeadRadius:=3;
     dir.TopArrowHeadHeight:=4;
     dir.BottomArrowHeadHeight:=0;
     dir.Material.FrontProperties.Diffuse.SetColor(1.0,0.0,0.0,0.5);

     dir.Position.X:=dir.Position.X+dir.Direction.X*dir.Height*0.5;
     dir.Position.Y:=dir.Position.Y+dir.Direction.Y*dir.Height*0.5;
     dir.Position.Z:=dir.Position.Z+dir.Direction.Z*dir.Height*0.5;

     glSpace.AddChild(dir);

     if(jobj.FindPath('Type') <> nil) then begin
         Case AnsiIndexStr(jobj.FindPath('Type').AsString, ['gaussian','disk','zgaussian', 'planar', 'pattern', 'fourier',
            'fourierx', 'fourierx2d','pattern3d']) of
              0..2:  AddDiskSource(jobj);      //Origin
              3..5:  AddPlanarSource(jobj, false);    //Planar Source
              6..7:  AddPlanarSource(jobj, true);    //Planar Source
              8:     AddPattern3DSource(jobj); //Pattern3D source
           else
           end;
     end;
end;

procedure TfmDomain.AddDetector(jobj: TJSONData);
var
     i: integer;
     obj: TGLSphere;
     data: TJSONArray;
     elem: TJSONData;
begin
     for i:=0 to jobj.Count-1 do begin;

       if(jobj.JSONType=jtObject) then begin
           elem:=jobj;
       end else begin
           elem:=jobj.Items[i];
       end;

       if(elem.FindPath('Pos')=nil) or (elem.FindPath('R')=nil) then begin
          MessageDlg('Warning', 'Malformed JSON Detector construct', mtError, [mbOK],0);
          exit;
       end;

       obj:=TGLSphere.Create(Self);

       obj.Up.SetVector(0,0,1);

       data:=TJSONArray(elem.FindPath('Pos'));
       obj.Position.X:=data.Items[0].AsFloat;
       obj.Position.Y:=data.Items[1].AsFloat;
       obj.Position.Z:=data.Items[2].AsFloat;
       obj.Radius:=elem.FindPath('R').AsFloat;
       obj.Slices:=64;
       obj.Material.FrontProperties.Diffuse.SetColor(0.0,1.0,0.0,0.5);
       glSpace.AddChild(obj);
       if(jobj.JSONType=jtObject) then exit;
     end;
end;

procedure TfmDomain.plEditorMouseEnter(Sender: TObject);
begin
    plEditor.Width:=editorwidth;
    glCamera.TargetObject:=glDomain;
end;

procedure TfmDomain.plEditorMouseLeave(Sender: TObject);
begin
    if(not btPin.Down) then begin
        plEditor.Width:=40;
        glCamera.TargetObject:=glDomain;
    end;
end;

procedure TfmDomain.ShowJSON(root: TJSONData; rootstr: string);
var
     i: integer;
     jobj: TJSONObject;
     ss, objname: string;
begin
     ss:= root.AsJSON;
     if(root.FindPath(rootstr) <> nil) then
         root:=root.FindPath(rootstr);

     if(rootstr = 'Shapes') and (root.JSONType <> jtArray) then begin
        MessageDlg('JSON Error','Shape data root node should always be an array', mtError, [mbOK],0);
        exit;
     end;
     for i:=0 to root.Count-1 do begin
       jobj:=TJSONObject(root.Items[i]);
       if(root.JSONType = jtArray) then begin
           objname:=jobj.Names[0];
       end else begin
           objname:=TJSONObject(root).Names[i];
       end;
       ss:=jobj.AsJSON;
       Case AnsiIndexStr(objname, ['Origin','Grid', 'Box', 'Subgrid', 'Sphere',
          'Cylinder', 'XLayers','YLayers','ZLayers','XSlabs','YSlabs','ZSlabs',
          'Name','Source','Detector']) of
          0: AddOrigin(jobj);      //Origin
          1: AddGrid(jobj);        //Grid
          2: AddBox(jobj, jobj.Names[0]<>'Box');    //box
          3: AddBox(jobj, jobj.Names[0]<>'Box');    //Subgrid
          4: AddSphere(jobj);      //Sphere
          5: AddCylinder(jobj);    //Cylinder
          6: AddLayers(jobj,1);    //XLayers
          7: AddLayers(jobj,2);    //YLayers
          8: AddLayers(jobj,3);    //ZLayers
          9: AddSlabs(jobj,1);     //XLayers
          10: AddSlabs(jobj,2);    //YLayers
          11: AddSlabs(jobj,3);    //ZLayers
          12: AddName(jobj);       //Name
          13: AddSource(jobj);     //Source
          14: AddDetector(jobj);   //Detector
         -1: ShowMessage('Unsupported Shape Keyword'); // not present in array
       else
          ShowMessage('Shape keyword '+ jobj.Names[0]+' is not supported'); // present, but not handled above
       end;
     end;
end;

procedure TfmDomain.glCanvasMouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
var
	dx, dy : Integer;
	v : TVector;
begin
	// calculate delta since last move or last mousedown
	dx:=(mdx-x); dy:=(mdy-y);
	mdx:=x; mdy:=y;
	if ssLeft in Shift then begin
        if ssShift in Shift then begin
                // right button with shift rotates the teapot
                // (rotation happens around camera's axis)
	   	glCamera.RotateObject(glSpace, dy, dx);
        end else begin
   		// right button without shift changes camera angle
	   	// (we're moving around the parent and target dummycube)
		glCamera.MoveAroundTarget(dy, dx)
        end;
	end else if Shift=[ssRight] then begin
		// left button moves our target and parent dummycube
		v:=glCamera.ScreenDeltaToVectorXY(dx, -dy,
		  0.12*glCamera.DistanceToTarget/glCamera.FocalLength);
		glSpace.Position.Translate(v);
		// notify camera that its position/target has been changed
		glCamera.TransformationChanged;
	end;
end;

procedure TfmDomain.glCanvasMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
	// store mouse coordinates when a button went down
	mdx:=x; mdy:=y;
end;

procedure TfmDomain.FormShow(Sender: TObject);
begin
    glCanvas.Invalidate;
    plEditor.Width:=40;
    acRenderExecute(Sender);
end;

procedure TfmDomain.FormCreate(Sender: TObject);
var
   i: integer;
begin
   for i:=0 to 1023 do begin
      colormap[i][0]:=random;
      colormap[i][1]:=random;
      colormap[i][2]:=random;
   end;
   editorwidth:=plEditor.Width;
end;

procedure TfmDomain.FormDestroy(Sender: TObject);
begin
   glSpace.DeleteChildren;
   FreeAndNil(JSONData);
end;

procedure TfmDomain.FormResize(Sender: TObject);
begin
  glCamera.TargetObject:=glDomain;
end;

procedure TfmDomain.acResetCameraExecute(Sender: TObject);
begin
  glCamera.Position.X:=80;
  glCamera.Position.Y:=-100;
  glCamera.Position.Z:=100;
end;

procedure TfmDomain.acSaveImageExecute(Sender: TObject);
var
   bm : TBitmap;
   bmp32 : TGLBitmap32;
begin
     bmp32:=glCanvas.Buffer.CreateSnapShot;
     try
        bm:=bmp32.Create32BitsBitmap;
        try
           dlSaveScreen.DefaultExt := GraphicExtension(TBitmap);
           dlSaveScreen.Filter := GraphicFilter(TBitmap);
           if dlSaveScreen.Execute then
              bm.SaveToFile(dlSaveScreen.FileName);
        finally
           bm.Free;
        end;
     finally
        bmp32.Free;
     end;
end;

procedure TfmDomain.acSaveJSONExecute(Sender: TObject);
begin
    if(dlSaveFile.Execute) then begin
        mmShapeJSON.Lines.SaveToFile(dlSaveFile.FileName);
    end;
end;

procedure TfmDomain.acHideBBXExecute(Sender: TObject);
begin
    if(glDomain.Material.PolygonMode=pmFill) then begin
        glDomain.Material.PolygonMode:=pmLines;
    end else begin
        glDomain.Material.PolygonMode:=pmFill;
    end;
end;

procedure TfmDomain.acLoadJSONExecute(Sender: TObject);
begin
    if(dlOpenFile.Execute) then begin
      mmShapeJSON.Lines.LoadFromFile(dlOpenFile.FileName);
    end;
end;

procedure TfmDomain.acExitExecute(Sender: TObject);
begin
    Close;
end;

procedure TfmDomain.acRenderExecute(Sender: TObject);
begin
  LoadJSONShape(mmShapeJSON.Lines.Text);
  ShowJSON(JSONdata,'Shapes');
  ShowJSON(JSONdata,'Optode');
end;

end.
