unit mcxrender;

{==============================================================================
    Monte Carlo eXtreme (MCX) Studio - Domain Renderer
-------------------------------------------------------------------------------
    Author: Qianqian Fang
    Email : q.fang at neu.edu
    Web   : https://mcx.space
    License: GNU General Public License version 3 (GPLv3)
===============================================================================}

interface

uses
  SysUtils, Classes, Graphics, Controls, Forms, Dialogs, GLScene, GLObjects,
  ExtCtrls, ComCtrls, ActnList, ExtDlgs, Buttons, SynEdit,
  SynHighlighterJScript, synhighlighterunixshellscript, GLBehaviours, GLTexture,
  GLVectorGeometry, GLLCLViewer, GLGeomObjects, GLCoordinates, GLCrossPlatform,
  GLGraphics, GLMaterial, GLColor, GLState, GLSkydome, GLMesh, Types, strutils,
  fpjson, jsonparser, LCLType, GLWindowsFont, GLBitmapFont, GLGraph, OpenGLTokens;

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
    btTogglePersp: TToolButton;
    btBackground: TColorButton;
    glCanvas: TGLSceneViewer;
    glDomain: TGLCube;
    glLight2: TGLLightSource;
    glShape: TGLScene;
    glCamera: TGLCamera;
    glLight1: TGLLightSource;
    glShapes: TGLDummyCube;
    DCCoordsZ: TGLDummyCube;
    DCCoordsY: TGLDummyCube;
    DCCoordsX: TGLDummyCube;
    GLWinBmpFont: TGLWindowsBitmapFont;
    XYGrid: TGLXYZGrid;
    YZGrid: TGLXYZGrid;
    XZGrid: TGLXYZGrid;
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
    procedure btBackgroundColorChanged(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure glCanvasMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure glCanvasMouseUp(Sender: TObject; Button: TMouseButton;
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
    procedure AddConeSource(jobj: TJSONData);
    procedure AddLineSource(jobj: TJSONData);
    procedure AddPlanarSource(jobj: TJSONData; isorth: boolean=false);
    procedure AddPattern3DSource(jobj: TJSONData);
    procedure AddDetector(jobj: TJSONData);
    procedure plEditorMouseEnter(Sender: TObject);
    procedure plEditorMouseLeave(Sender: TObject);
    procedure ShowJSON(root: TJSONData; rootstr: string);
    procedure LoadJSONShape(shapejson: AnsiString);
    procedure Splitter1Moved(Sender: TObject);
    procedure DrawAxis(Sender : TObject);
    procedure SelectObject(obj: TGLBaseSceneObject);
    procedure DeselectObject;
    procedure CreateWireOverlay(src: TGLBaseSceneObject);
    procedure CreateAxisGizmo(src: TGLBaseSceneObject);
    procedure UpdateGizmoPosition;
    procedure DeleteSelectedObject;
    procedure UpdateJSONFromScene;
    procedure glCanvasKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
    function PickObjectAt(mx, my: integer): TGLBaseSceneObject;
    function PickAxisAt(mx, my: integer): integer;
    procedure acTogglePerspExecute(Sender: TObject);
  private
    mdx, mdy : Integer;
    editorwidth: integer;
    JSONdata : TJSONData;
    colormap: array [0..1023,0..2] of extended;
    FSelectedObj: TGLBaseSceneObject;
    FWireOverlay: TGLBaseSceneObject;
    FAxisGizmo: array[0..2] of TGLArrowLine;
    FDragging: Boolean;
    FResizing: Boolean;
    FClickedOnObj: Boolean;
    FDragAxis: Integer;  { -1=none, 0=X, 1=Y, 2=Z }
    FLastPickX, FLastPickY: Integer;
    FPickCycleIdx: Integer;
    acTogglePersp: TAction;
  public

  end;

var
  fmDomain: TfmDomain;

implementation

{$R *.lfm}

const
  AxisStep :  TGLFloat =  10;

procedure TfmDomain.glCanvasMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
var
  f: single;
begin
  f := Power(1.1, WheelDelta/200.0);
  if glCamera.CameraStyle = csOrthogonal then begin
    glCamera.DepthOfView := glCamera.DepthOfView / f;
    glCamera.FocalLength := glCamera.FocalLength * f;
  end else begin
    glCamera.AdjustDistanceToTarget(f);
  end;
end;

procedure TfmDomain.LoadJSONShape(shapejson: AnsiString);
begin
    FreeAndNil(JSONData);
    FWireOverlay := nil;
    FSelectedObj := nil;
    FAxisGizmo[0] := nil;
    FAxisGizmo[1] := nil;
    FAxisGizmo[2] := nil;
    FDragAxis := -1;
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
     gridstep: double;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONData(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('Size')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Grid shape construct', mtError, [mbOK],0);
        exit;
     end;

     glDomain.DeleteChildren; // grid object reset the domain

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;

     data:=TJSONArray(jobj.FindPath('Size'));
     glDomain.CubeWidth:=data.Items[0].AsFloat;
     glDomain.CubeDepth:=data.Items[1].AsFloat;
     glDomain.CubeHeight:=data.Items[2].AsFloat;
     glDomain.Position.X:=glDomain.CubeWidth*0.5;
     glDomain.Position.Y:=glDomain.CubeDepth*0.5;
     glDomain.Position.Z:=glDomain.CubeHeight*0.5;

     DCCoordsX.Position.X:=glDomain.CubeWidth*0.5;
     DCCoordsX.Position.Y:=glDomain.CubeDepth*0.5;
     DCCoordsX.Position.Z:=glDomain.CubeHeight*0.5;
     DCCoordsY.Position.X:=glDomain.CubeWidth*0.5;
     DCCoordsY.Position.Y:=glDomain.CubeDepth*0.5;
     DCCoordsY.Position.Z:=glDomain.CubeHeight*0.5;
     DCCoordsZ.Position.X:=glDomain.CubeWidth*0.5;
     DCCoordsZ.Position.Y:=glDomain.CubeDepth*0.5;
     DCCoordsZ.Position.Z:=glDomain.CubeHeight*0.5;

     XYGrid.Scale.X:=glDomain.CubeWidth;
     XYGrid.Scale.Y:=glDomain.CubeDepth;
     XYGrid.Scale.Z:=glDomain.CubeHeight;
     YZGrid.Scale.X:=glDomain.CubeWidth;
     YZGrid.Scale.Y:=glDomain.CubeDepth;
     YZGrid.Scale.Z:=glDomain.CubeHeight;
     XZGrid.Scale.X:=glDomain.CubeWidth;
     XZGrid.Scale.Y:=glDomain.CubeDepth;
     XZGrid.Scale.Z:=glDomain.CubeHeight;

     gridstep:=10.0/glDomain.CubeWidth;
     XYGrid.XSamplingScale.Step:=gridstep;
     YZGrid.XSamplingScale.Step:=gridstep;
     XZGrid.XSamplingScale.Step:=gridstep;
     gridstep:=10.0/glDomain.CubeDepth;
     XYGrid.YSamplingScale.Step:=gridstep;
     YZGrid.YSamplingScale.Step:=gridstep;
     XZGrid.YSamplingScale.Step:=gridstep;
     gridstep:=10.0/glDomain.CubeHeight;
     XYGrid.ZSamplingScale.Step:=gridstep;
     YZGrid.ZSamplingScale.Step:=gridstep;
     XZGrid.ZSamplingScale.Step:=gridstep;

     DrawAxis(nil);
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
         jobj:=TJSONData(jobj.Items[0]);
     if not (jobj is TJSONArray) then begin
        MessageDlg('Warning', 'Malformed JSON Origin shape construct', mtError, [mbOK],0);
        exit;
     end;

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
         jobj:=TJSONData(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('Size')=nil) or (jobj.FindPath('O')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Box shape construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLCube.Create(Self);

     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;
     obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     obj.Material.FrontProperties.Specular.SetColor(0.8,0.8,0.8,1.0);
     obj.Material.FrontProperties.Shininess:=64;
     obj.Material.BlendingMode:=bmTransparency;
     obj.Material.DepthProperties.DepthWrite := False;

     data:=TJSONArray(jobj.FindPath('Size'));
     obj.CubeWidth:=data.Items[0].AsFloat;
     obj.CubeDepth:=data.Items[1].AsFloat;
     obj.CubeHeight:=data.Items[2].AsFloat;

     data:=TJSONArray(jobj.FindPath('O'));
     obj.Position.X:=data.Items[0].AsFloat+obj.CubeWidth*0.5+Integer(isbox)*0.5;
     obj.Position.Y:=data.Items[1].AsFloat+obj.CubeDepth*0.5+Integer(isbox)*0.5;
     obj.Position.Z:=data.Items[2].AsFloat+obj.CubeHeight*0.5+Integer(isbox)*0.5;

     obj.TagFloat := glSpace.Count;  { track index for JSON sync }
     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddSphere(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLSphere;
     data: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONData(jobj.Items[0]);
     if(jobj.FindPath('Tag')=nil) or (jobj.FindPath('R')=nil) or (jobj.FindPath('O')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Sphere shape construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLSphere.Create(Self);
     obj.Up.SetVector(0,0,1);
     obj.Direction.SetVector(0,1,0);

     objtag:=jobj.FindPath('Tag').AsInteger mod 1024;
     obj.Material.FrontProperties.Diffuse.SetColor(colormap[objtag][0],colormap[objtag][1],colormap[objtag][2],0.5);
     obj.Material.FrontProperties.Specular.SetColor(0.8,0.8,0.8,1.0);
     obj.Material.FrontProperties.Shininess:=64;
     obj.Material.BlendingMode:=bmTransparency;
     obj.Material.DepthProperties.DepthWrite := False;

     obj.Radius:=jobj.FindPath('R').AsFloat;

     data:=TJSONArray(jobj.FindPath('O'));
     obj.Position.X:=data.Items[0].AsFloat;
     obj.Position.Y:=data.Items[1].AsFloat;
     obj.Position.Z:=data.Items[2].AsFloat;
     obj.Slices:=64;

     obj.TagFloat := glSpace.Count;
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
         jobj:=TJSONData(jobj.Items[0]);
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
     obj.Material.FrontProperties.Specular.SetColor(0.8,0.8,0.8,1.0);
     obj.Material.FrontProperties.Shininess:=64;
     obj.Material.BlendingMode:=bmTransparency;
     obj.Material.DepthProperties.DepthWrite := False;

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

     obj.TagFloat := glSpace.Count;
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
         jobj:=TJSONData(jobj.Items[0]);
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
       obj.Material.FrontProperties.Specular.SetColor(0.8,0.8,0.8,1.0);
       obj.Material.FrontProperties.Shininess:=64;
       obj.Material.BlendingMode:=bmTransparency;
       obj.Material.DepthProperties.DepthWrite := False;

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
         jobj:=TJSONData(jobj.Items[0]);
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
       obj.Material.FrontProperties.Specular.SetColor(0.8,0.8,0.8,1.0);
       obj.Material.FrontProperties.Shininess:=64;
       obj.Material.BlendingMode:=bmTransparency;
       obj.Material.DepthProperties.DepthWrite := False;

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
         jobj:=TJSONData(jobj.Items[0]);
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
         jobj:=TJSONData(jobj.Items[0]);
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

     data:=TJSONArray(jobj.FindPath('Param1'));
     obj.OuterRadius:=data.Items[0].AsFloat;

     data:=TJSONArray(jobj.FindPath('Dir'));
     obj.Direction.SetVector(data.Items[0].AsFloat,data.Items[1].AsFloat,data.Items[2].AsFloat);

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddPattern3DSource(jobj: TJSONData);
var
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
     obj.Material.DepthProperties.DepthWrite := False;

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

procedure TfmDomain.AddLineSource(jobj: TJSONData);
var
     obj: TGLLines;
     data,param: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Param1')=nil) then begin
        MessageDlg('Warning', 'Malformed JSON Line Source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLLines.Create(Self);

     obj.Up.SetVector(1,0,0);
     obj.Direction.SetVector(0,1,0);
     obj.LineColor.SetColor(1,0,0);
     obj.NodesAspect:=lnaDodecahedron;
     obj.LineWidth:=2;

     data:=TJSONArray(jobj.FindPath('Pos'));
     obj.AddNode(data.Items[0].AsFloat, data.Items[1].AsFloat,data.Items[2].AsFloat);
     param:=TJSONArray(jobj.FindPath('Param1'));
     obj.AddNode(data.Items[0].AsFloat+param.Items[0].AsFloat, data.Items[1].AsFloat+param.Items[1].AsFloat,data.Items[2].AsFloat+param.Items[2].AsFloat);

     glSpace.AddChild(obj);
end;

procedure TfmDomain.AddConeSource(jobj: TJSONData);
var
     objtag: integer;
     obj: TGLCone;
     data, param: TJSONArray;
begin
     if(jobj.Count=1) and (jobj.Items[0].Count>0) then
         jobj:=TJSONObject(jobj.Items[0]);
     if(jobj.FindPath('Param1')=nil ) then begin
        MessageDlg('Warning', 'Malformed JSON cone source construct', mtError, [mbOK],0);
        exit;
     end;
     obj:=TGLCone.Create(Self);

     obj.Up.SetVector(0,0,1);

     obj.Material.FrontProperties.Diffuse.SetColor(1.0,1.0,0.0,0.5);
     obj.Material.BlendingMode:=bmTransparency;
     obj.Material.DepthProperties.DepthWrite := False;

     param:=TJSONArray(jobj.FindPath('Param1'));
     obj.Height:=20;
     obj.BottomRadius:=obj.Height*tan(param.Items[0].AsFloat);

     data:=TJSONArray(jobj.FindPath('Pos'));
     param:=TJSONArray(jobj.FindPath('Dir'));
     obj.Position.X:=data.Items[0].AsFloat+param.Items[0].AsFloat*obj.Height*0.5;
     obj.Position.Y:=data.Items[1].AsFloat+param.Items[1].AsFloat*obj.Height*0.5;
     obj.Position.Z:=data.Items[2].AsFloat+param.Items[2].AsFloat*obj.Height*0.5;

     obj.Up.SetVector(-param.Items[0].AsFloat,-param.Items[1].AsFloat,-param.Items[2].AsFloat);

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
            'fourierx', 'fourierx2d','pattern3d','line','slit','cone']) of
              0..2:  AddDiskSource(jobj);
              3..5:  AddPlanarSource(jobj, false);
              6..7:  AddPlanarSource(jobj, true);
              8:     AddPattern3DSource(jobj);
              9..10: AddLineSource(jobj);
              11:    AddConeSource(jobj);
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

{ === Selection, picking, wireframe overlay, delete === }

function TfmDomain.PickObjectAt(mx, my: integer): TGLBaseSceneObject;
var
  objWorldPos: TVector;
  i, hitCount: integer;
  child: TGLBaseSceneObject;
  radius, halfW, halfD, halfH, t: single;
  hits: array[0..63] of record
    obj: TGLBaseSceneObject;
    dist: single;
  end;
  tmp: TGLBaseSceneObject;
  tmpDist: single;
  j: integer;
  objScreen, edgeScreen: TAffineVector;
  camRight: TVector;
  edgeWorld: TAffineVector;
  sx, sy, dx2, dy2, scrRadius: single;
begin
  Result := nil;
  hitCount := 0;

  for i := 0 to glSpace.Count - 1 do begin
    child := glSpace.Children[i];
    if child = FWireOverlay then continue;
    if not (child is TGLCustomSceneObject) then continue;
    { skip axis gizmo arrows }
    if (child = FAxisGizmo[0]) or (child = FAxisGizmo[1]) or (child = FAxisGizmo[2]) then continue;

    objWorldPos := child.AbsolutePosition;

    { determine bounding radius }
    radius := 0;
    if child is TGLSphere then
      radius := TGLSphere(child).Radius
    else if child is TGLCylinder then begin
      radius := TGLCylinder(child).BottomRadius;
      if TGLCylinder(child).Height * 0.5 > radius then
        radius := TGLCylinder(child).Height * 0.5;
    end else if child is TGLCube then begin
      halfW := TGLCube(child).CubeWidth * 0.5;
      halfD := TGLCube(child).CubeDepth * 0.5;
      halfH := TGLCube(child).CubeHeight * 0.5;
      radius := sqrt(halfW*halfW + halfD*halfD + halfH*halfH);
    end else
      continue;

    { project center to screen }
    objScreen := glCanvas.Buffer.WorldToScreen(
      AffineVectorMake(objWorldPos.V[0], objWorldPos.V[1], objWorldPos.V[2]));

    { skip behind camera }
    if (objScreen.V[2] < 0) or (objScreen.V[2] > 1) then continue;

    { convert GL coords (origin bottom-left) to widget coords (origin top-left) }
    sx := objScreen.V[0];
    sy := glCanvas.Height - objScreen.V[1];

    { project edge point to get pixel radius }
    camRight := glCamera.AbsoluteRight;
    edgeWorld := AffineVectorMake(
      objWorldPos.V[0] + camRight.V[0] * radius,
      objWorldPos.V[1] + camRight.V[1] * radius,
      objWorldPos.V[2] + camRight.V[2] * radius);
    edgeScreen := glCanvas.Buffer.WorldToScreen(edgeWorld);

    scrRadius := sqrt(
      (edgeScreen.V[0] - objScreen.V[0]) * (edgeScreen.V[0] - objScreen.V[0]) +
      (edgeScreen.V[1] - objScreen.V[1]) * (edgeScreen.V[1] - objScreen.V[1]));
    if scrRadius < 5 then scrRadius := 5;

    dx2 := mx - sx;
    dy2 := my - sy;
    if (dx2*dx2 + dy2*dy2) <= (scrRadius * scrRadius) then begin
      t := VectorLength(VectorSubtract(objWorldPos, glCamera.AbsolutePosition));
      if hitCount < 64 then begin
        hits[hitCount].obj := child;
        hits[hitCount].dist := t;
        Inc(hitCount);
      end;
    end;
  end;

  if hitCount = 0 then exit;

  { sort by distance, nearest first }
  for i := 0 to hitCount - 2 do
    for j := i + 1 to hitCount - 1 do
      if hits[j].dist < hits[i].dist then begin
        tmp := hits[i].obj; tmpDist := hits[i].dist;
        hits[i].obj := hits[j].obj; hits[i].dist := hits[j].dist;
        hits[j].obj := tmp; hits[j].dist := tmpDist;
      end;

  { cycle through hits on repeated clicks at same position }
  if (abs(mx - FLastPickX) < 3) and (abs(my - FLastPickY) < 3) then
    FPickCycleIdx := (FPickCycleIdx + 1) mod hitCount
  else
    FPickCycleIdx := 0;

  FLastPickX := mx;
  FLastPickY := my;
  Result := hits[FPickCycleIdx].obj;
end;

procedure TfmDomain.SelectObject(obj: TGLBaseSceneObject);
begin
     if obj = FSelectedObj then exit;
     DeselectObject;
     FSelectedObj := obj;
     CreateWireOverlay(obj);
     CreateAxisGizmo(obj);
     Caption := 'MCX Domain Renderer - Selected: ' + obj.ClassName;
end;

procedure TfmDomain.DeselectObject;
var
     i: integer;
begin
     if FWireOverlay <> nil then begin
        FWireOverlay.Free;
        FWireOverlay := nil;
     end;
     for i := 0 to 2 do begin
        if FAxisGizmo[i] <> nil then begin
           FAxisGizmo[i].Free;
           FAxisGizmo[i] := nil;
        end;
     end;
     FSelectedObj := nil;
     FDragging := False;
     FResizing := False;
     FDragAxis := -1;
     Caption := 'MCX Domain Renderer';
end;

procedure TfmDomain.CreateWireOverlay(src: TGLBaseSceneObject);
var
     wire: TGLSceneObject;
begin
     if FWireOverlay <> nil then begin
        FWireOverlay.Free;
        FWireOverlay := nil;
     end;

     if src is TGLSphere then begin
        wire := TGLSphere.Create(Self);
        TGLSphere(wire).Radius := TGLSphere(src).Radius * 1.01;
        TGLSphere(wire).Slices := TGLSphere(src).Slices;
        TGLSphere(wire).Stacks := TGLSphere(src).Stacks;
     end else if src is TGLCylinder then begin
        wire := TGLCylinder.Create(Self);
        TGLCylinder(wire).TopRadius := TGLCylinder(src).TopRadius * 1.01;
        TGLCylinder(wire).BottomRadius := TGLCylinder(src).BottomRadius * 1.01;
        TGLCylinder(wire).Height := TGLCylinder(src).Height;
        TGLCylinder(wire).Slices := TGLCylinder(src).Slices;
        TGLCylinder(wire).Alignment := TGLCylinder(src).Alignment;
     end else if src is TGLCube then begin
        wire := TGLCube.Create(Self);
        TGLCube(wire).CubeWidth := TGLCube(src).CubeWidth * 1.01;
        TGLCube(wire).CubeDepth := TGLCube(src).CubeDepth * 1.01;
        TGLCube(wire).CubeHeight := TGLCube(src).CubeHeight * 1.01;
     end else
        exit;

     wire.Position.AsVector := src.Position.AsVector;
     wire.Up.AsVector := src.Up.AsVector;
     wire.Direction.AsVector := src.Direction.AsVector;
     wire.Material.FrontProperties.Diffuse.SetColor(1.0, 1.0, 0.0, 1.0);
     wire.Material.FrontProperties.Emission.SetColor(1.0, 1.0, 0.0, 1.0);
     wire.Material.PolygonMode := pmLines;
     wire.Material.FrontProperties.Ambient.SetColor(1.0, 1.0, 0.0, 1.0);
     glSpace.AddChild(wire);
     FWireOverlay := wire;
end;

procedure TfmDomain.CreateAxisGizmo(src: TGLBaseSceneObject);
const
     ArrowLen = 8;
     ArrowRad = 0.3;
     HeadRad = 1.0;
     HeadLen = 2.0;
     Colors: array[0..2,0..2] of single = (
       (1.0, 0.0, 0.0),   { X = Red }
       (0.0, 1.0, 0.0),   { Y = Green }
       (0.0, 0.0, 1.0)    { Z = Blue }
     );
var
     i: integer;
     a: TGLArrowLine;
begin
     for i := 0 to 2 do begin
        if FAxisGizmo[i] <> nil then begin
           FAxisGizmo[i].Free;
           FAxisGizmo[i] := nil;
        end;

        a := TGLArrowLine.Create(Self);
        a.Position.AsVector := src.Position.AsVector;
        a.Height := ArrowLen;
        a.TopRadius := ArrowRad;
        a.BottomRadius := ArrowRad;
        a.TopArrowHeadRadius := HeadRad;
        a.TopArrowHeadHeight := HeadLen;
        a.BottomArrowHeadHeight := 0;
        a.Material.FrontProperties.Diffuse.SetColor(Colors[i][0], Colors[i][1], Colors[i][2], 1.0);
        a.Material.FrontProperties.Emission.SetColor(Colors[i][0]*0.3, Colors[i][1]*0.3, Colors[i][2]*0.3, 1.0);

        case i of
          0: begin { X axis }
            a.Direction.SetVector(1, 0, 0);
            a.Up.SetVector(0, 0, 1);
            a.Position.X := a.Position.X + ArrowLen * 0.5;
          end;
          1: begin { Y axis }
            a.Direction.SetVector(0, 1, 0);
            a.Up.SetVector(0, 0, 1);
            a.Position.Y := a.Position.Y + ArrowLen * 0.5;
          end;
          2: begin { Z axis }
            a.Direction.SetVector(0, 0, 1);
            a.Up.SetVector(1, 0, 0);
            a.Position.Z := a.Position.Z + ArrowLen * 0.5;
          end;
        end;

        glSpace.AddChild(a);
        FAxisGizmo[i] := a;
     end;
end;

procedure TfmDomain.UpdateGizmoPosition;
const
     ArrowLen = 8;
var
     i: integer;
begin
     if FSelectedObj = nil then exit;
     for i := 0 to 2 do begin
        if FAxisGizmo[i] = nil then continue;
        FAxisGizmo[i].Position.AsVector := FSelectedObj.Position.AsVector;
        case i of
          0: FAxisGizmo[i].Position.X := FAxisGizmo[i].Position.X + ArrowLen * 0.5;
          1: FAxisGizmo[i].Position.Y := FAxisGizmo[i].Position.Y + ArrowLen * 0.5;
          2: FAxisGizmo[i].Position.Z := FAxisGizmo[i].Position.Z + ArrowLen * 0.5;
        end;
     end;
end;

function TfmDomain.PickAxisAt(mx, my: integer): integer;
var
     i: integer;
     arrowPos: TVector;
     arrowScreen: TAffineVector;
     sx, sy, dx2, dy2, dist, bestDist: single;
begin
     Result := -1;
     bestDist := 900;

     for i := 0 to 2 do begin
        if FAxisGizmo[i] = nil then continue;
        arrowPos := FAxisGizmo[i].AbsolutePosition;
        arrowScreen := glCanvas.Buffer.WorldToScreen(
          AffineVectorMake(arrowPos.V[0], arrowPos.V[1], arrowPos.V[2]));

        if (arrowScreen.V[2] < 0) or (arrowScreen.V[2] > 1) then continue;

        sx := arrowScreen.V[0];
        sy := glCanvas.Height - arrowScreen.V[1];
        dx2 := mx - sx;
        dy2 := my - sy;
        dist := dx2*dx2 + dy2*dy2;
        if (dist < bestDist) then begin
           bestDist := dist;
           Result := i;
        end;
     end;
end;

procedure TfmDomain.DeleteSelectedObject;
var
     idx: integer;
begin
     if FSelectedObj = nil then exit;
     idx := glSpace.IndexOfChild(FSelectedObj);
     DeselectObject;
     if idx >= 0 then
        glSpace.Children[idx].Free;
     UpdateJSONFromScene;
end;

procedure TfmDomain.UpdateJSONFromScene;
var
     shapes, jobj, inner: TJSONData;
     oarr: TJSONArray;
     objname: string;
     i, idx: integer;
     child: TGLBaseSceneObject;
begin
     if JSONData = nil then exit;
     if FSelectedObj = nil then exit;

     child := FSelectedObj;
     shapes := JSONData.FindPath('Shapes');
     if shapes = nil then exit;

     { find the JSON entry by scanning shapes for matching type and approximate position }
     for i := 0 to shapes.Count - 1 do begin
       jobj := shapes.Items[i];
       if jobj.Count = 0 then continue;
       objname := TJSONObject(jobj).Names[0];
       inner := TJSONObject(jobj).Items[0];

       if (child is TGLSphere) and (objname = 'Sphere') and (inner.FindPath('O') <> nil) then begin
         oarr := TJSONArray(inner.FindPath('O'));
         oarr.Items[0] := TJSONFloatNumber.Create(TGLSphere(child).Position.X);
         oarr.Items[1] := TJSONFloatNumber.Create(TGLSphere(child).Position.Y);
         oarr.Items[2] := TJSONFloatNumber.Create(TGLSphere(child).Position.Z);
         if inner.FindPath('R') <> nil then
           TJSONObject(inner).Delete('R');
         TJSONObject(inner).Add('R', TJSONFloatNumber.Create(TGLSphere(child).Radius));
         break;
       end else if (child is TGLCylinder) and (objname = 'Cylinder') and (inner.FindPath('C0') <> nil) then begin
         oarr := TJSONArray(inner.FindPath('C0'));
         oarr.Items[0] := TJSONFloatNumber.Create(TGLCylinder(child).Position.X);
         oarr.Items[1] := TJSONFloatNumber.Create(TGLCylinder(child).Position.Y);
         oarr.Items[2] := TJSONFloatNumber.Create(TGLCylinder(child).Position.Z);
         oarr := TJSONArray(inner.FindPath('C1'));
         oarr.Items[0] := TJSONFloatNumber.Create(
           TGLCylinder(child).Position.X + TGLCylinder(child).Up.X * TGLCylinder(child).Height);
         oarr.Items[1] := TJSONFloatNumber.Create(
           TGLCylinder(child).Position.Y + TGLCylinder(child).Up.Y * TGLCylinder(child).Height);
         oarr.Items[2] := TJSONFloatNumber.Create(
           TGLCylinder(child).Position.Z + TGLCylinder(child).Up.Z * TGLCylinder(child).Height);
         if inner.FindPath('R') <> nil then
           TJSONObject(inner).Delete('R');
         TJSONObject(inner).Add('R', TJSONFloatNumber.Create(TGLCylinder(child).BottomRadius));
         break;
       end else if (child is TGLCube) and ((objname = 'Box') or (objname = 'Subgrid')) and (inner.FindPath('O') <> nil) then begin
         oarr := TJSONArray(inner.FindPath('O'));
         oarr.Items[0] := TJSONFloatNumber.Create(TGLCube(child).Position.X - TGLCube(child).CubeWidth * 0.5);
         oarr.Items[1] := TJSONFloatNumber.Create(TGLCube(child).Position.Y - TGLCube(child).CubeDepth * 0.5);
         oarr.Items[2] := TJSONFloatNumber.Create(TGLCube(child).Position.Z - TGLCube(child).CubeHeight * 0.5);
         oarr := TJSONArray(inner.FindPath('Size'));
         oarr.Items[0] := TJSONFloatNumber.Create(TGLCube(child).CubeWidth);
         oarr.Items[1] := TJSONFloatNumber.Create(TGLCube(child).CubeDepth);
         oarr.Items[2] := TJSONFloatNumber.Create(TGLCube(child).CubeHeight);
         break;
       end;
     end;

     mmShapeJSON.Lines.Text := JSONData.FormatJSON;
     Caption := 'MCX Domain Renderer - JSON updated';
end;

procedure TfmDomain.glCanvasKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
     case Key of
       46: begin DeleteSelectedObject; Key := 0; end;
       27: begin DeselectObject; Key := 0; end;
     end;
end;

procedure TfmDomain.acTogglePerspExecute(Sender: TObject);
var
     dist: single;
begin
     if glCamera.CameraStyle = csPerspective then begin
        { save current distance, switch to ortho, set appropriate view width }
        dist := glCamera.DistanceToTarget;
        glCamera.CameraStyle := csOrthogonal;
        { in ortho mode, DepthOfView controls the visible width }
        glCamera.DepthOfView := dist * 2;
        btTogglePersp.Caption := 'Persp';
        Caption := 'MCX Domain Renderer [Orthographic]';
     end else begin
        glCamera.CameraStyle := csPerspective;
        glCamera.DepthOfView := 1000;
        btTogglePersp.Caption := 'Ortho';
        Caption := 'MCX Domain Renderer [Perspective]';
     end;
end;

{ === End selection code === }

procedure TfmDomain.ShowJSON(root: TJSONData; rootstr: string);
var
     i: integer;
     jobj: TJSONData;
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
       jobj:=root.Items[i];
       if(root.JSONType = jtArray) then begin
           objname:=TJSONObject(jobj).Names[0];
       end else begin
           objname:=TJSONObject(root).Names[i];
       end;
       ss:=jobj.AsJSON;
       Case AnsiIndexStr(objname, ['Origin','Grid', 'Box', 'Subgrid', 'Sphere',
          'Cylinder', 'XLayers','YLayers','ZLayers','XSlabs','YSlabs','ZSlabs',
          'Name','Source','Detector']) of
          0: AddOrigin(jobj);
          1: AddGrid(jobj);
          2: AddBox(jobj, objname<>'Box');
          3: AddBox(jobj, objname<>'Box');
          4: AddSphere(jobj);
          5: AddCylinder(jobj);
          6: AddLayers(jobj,1);
          7: AddLayers(jobj,2);
          8: AddLayers(jobj,3);
          9: AddSlabs(jobj,1);
          10: AddSlabs(jobj,2);
          11: AddSlabs(jobj,3);
          12: AddName(TJSONObject(jobj));
          13: AddSource(jobj);
          14: AddDetector(jobj);
         -1: ShowMessage('Unsupported Shape Keyword');
       else
          ShowMessage('Shape keyword '+ objname+' is not supported');
       end;
     end;
end;

procedure TfmDomain.glCanvasMouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
var
	dx, dy : Integer;
	v : TVector;
	scaleFactor, moveAmt : Single;
begin
	dx:=(mdx-x); dy:=(mdy-y);
	mdx:=x; mdy:=y;
	if ssLeft in Shift then begin
		if ssShift in Shift then begin
			glCamera.RotateObject(glSpace, dy, dx);
		end else if (ssCtrl in Shift) and (FSelectedObj <> nil) then begin
			{ Ctrl+drag = resize }
			FResizing := True;
			scaleFactor := 1.0 + dy * 0.01;
			if FSelectedObj is TGLSphere then
				TGLSphere(FSelectedObj).Radius := TGLSphere(FSelectedObj).Radius * scaleFactor
			else if FSelectedObj is TGLCylinder then begin
				TGLCylinder(FSelectedObj).TopRadius := TGLCylinder(FSelectedObj).TopRadius * scaleFactor;
				TGLCylinder(FSelectedObj).BottomRadius := TGLCylinder(FSelectedObj).BottomRadius * scaleFactor;
			end else if FSelectedObj is TGLCube then begin
				TGLCube(FSelectedObj).CubeWidth := TGLCube(FSelectedObj).CubeWidth * scaleFactor;
				TGLCube(FSelectedObj).CubeDepth := TGLCube(FSelectedObj).CubeDepth * scaleFactor;
				TGLCube(FSelectedObj).CubeHeight := TGLCube(FSelectedObj).CubeHeight * scaleFactor;
			end;
			CreateWireOverlay(FSelectedObj);
			UpdateGizmoPosition;
		end else if (FDragAxis >= 0) and (FSelectedObj <> nil) then begin
			{ dragging along a specific axis }
			FDragging := True;
			moveAmt := (-dx + dy) * 0.12 * glCamera.DistanceToTarget / glCamera.FocalLength;
			case FDragAxis of
				0: FSelectedObj.Position.X := FSelectedObj.Position.X + moveAmt;
				1: FSelectedObj.Position.Y := FSelectedObj.Position.Y + moveAmt;
				2: FSelectedObj.Position.Z := FSelectedObj.Position.Z + moveAmt;
			end;
			UpdateGizmoPosition;
			if FWireOverlay <> nil then
				FWireOverlay.Position.AsVector := FSelectedObj.Position.AsVector;
		end else begin
			{ default: orbit camera }
			glCamera.MoveAroundTarget(dy, dx);
		end;
	end else if Shift=[ssRight] then begin
		v:=glCamera.ScreenDeltaToVectorXY(dx, -dy,
		  0.12*glCamera.DistanceToTarget/glCamera.FocalLength);
		glSpace.Position.Translate(v);
		glCamera.TransformationChanged;
	end;
end;

procedure TfmDomain.glCanvasMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  pick: TGLBaseSceneObject;
  axis: Integer;
begin
	mdx:=x; mdy:=y;
	FDragging := False;
	FResizing := False;
	FClickedOnObj := False;
	FDragAxis := -1;
	if (Button = TMouseButton(0)) and not (ssShift in Shift) then begin
		{ first check if clicking on an axis arrow of selected object }
		if FSelectedObj <> nil then begin
			axis := PickAxisAt(x, y);
			if axis >= 0 then begin
				FDragAxis := axis;
				FClickedOnObj := True;
				exit;
			end;
		end;
		{ check if clicking on a shape body }
		if not (ssCtrl in Shift) then begin
			pick := PickObjectAt(x, y);
			if (pick <> nil) then begin
				SelectObject(pick);
				FClickedOnObj := True;
			end else begin
				DeselectObject;
			end;
		end;
	end;
end;

procedure TfmDomain.glCanvasMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
	if FDragging or FResizing then begin
		UpdateJSONFromScene;
	end;
	FDragging := False;
	FResizing := False;
end;

procedure TfmDomain.FormShow(Sender: TObject);
begin
    glCanvas.Invalidate;
    plEditor.Width:=40;
    acRenderExecute(Sender);
    glDomain.Material.DepthProperties.DepthWrite := False;
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
   if(Application.HasOption('f','json')) then begin
      mmShapeJSON.Lines.LoadFromFile(Application.GetOptionValue('f', 'json'));
   end;
   glCanvas.OnKeyDown := {$IFDEF FPC}@{$ENDIF}glCanvasKeyDown;
   glCanvas.OnMouseUp := {$IFDEF FPC}@{$ENDIF}glCanvasMouseUp;
   glCanvas.TabStop := True;
   FSelectedObj := nil;
   FWireOverlay := nil;
   FAxisGizmo[0] := nil;
   FAxisGizmo[1] := nil;
   FAxisGizmo[2] := nil;
   FDragging := False;
   FResizing := False;
   FDragAxis := -1;
   FLastPickX := -1;
   FLastPickY := -1;
   FPickCycleIdx := 0;
   acTogglePersp := TAction.Create(Self);
   acTogglePersp.Caption := 'Ortho';
   acTogglePersp.Hint := 'Toggle Orthographic/Perspective';
   acTogglePersp.OnExecute := {$IFDEF FPC}@{$ENDIF}acTogglePerspExecute;
   btTogglePersp := TToolButton.Create(Self);
   btTogglePersp.Parent := ToolBar1;
   btTogglePersp.Action := acTogglePersp;
   btTogglePersp.Caption := 'Ortho';
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

procedure TfmDomain.btBackgroundColorChanged(Sender: TObject);
begin
   glCanvas.Buffer.BackgroundColor:=btBackground.ButtonColor;
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

Procedure TfmDomain.DrawAxis(Sender : TObject);
Var
  ScaleFactor : TGLFloat;
  CurrentXCoord: TGLFloat;
  AxisMini :  TGLFloat;

  CurrentYCoord: TGLFloat;
  CurrentZCoord: TGLFloat;
  CurrentFlatText: TGLFlatText;
Begin
  DCCoordsX.DeleteChildren;
  DCCoordsY.DeleteChildren;
  DCCoordsZ.DeleteChildren;
  ScaleFactor := 0.2;
  { Draw X }
  AxisMini:=-glDomain.CubeWidth*0.5;
  CurrentXCoord := -glDomain.CubeWidth*0.5;
  CurrentYCoord := -glDomain.CubeDepth*0.5;
  CurrentZCoord := -glDomain.CubeHeight*0.5;
  while CurrentXCoord <= glDomain.CubeWidth*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsX);
    with DCCoordsX do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, -1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom;
        ModulateColor.AsWinColor := clRed;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentXCoord-AxisMini));
      end;
    end;
    CurrentXCoord := CurrentXCoord + AxisStep;
  end;
  CurrentXCoord := AxisMini;
  while CurrentXCoord <= glDomain.CubeWidth*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsX);
    with DCCoordsX do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, 1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom;
        ModulateColor.AsWinColor := clRed;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentXCoord-AxisMini));
      end;
    end;
    CurrentXCoord := CurrentXCoord + AxisStep;
  end;
  { Draw Y }
  AxisMini:=-glDomain.CubeDepth*0.5;
  CurrentXCoord := -glDomain.CubeWidth*0.5;
  CurrentYCoord := -glDomain.CubeDepth*0.5;
  CurrentZCoord := -glDomain.CubeHeight*0.5;
  while CurrentYCoord <= glDomain.CubeDepth*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsY);
    with DCCoordsY do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(1, 0, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom;
        ModulateColor.AsWinColor := clLime;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentYCoord-AxisMini));
      end;
    end;
    CurrentYCoord := CurrentYCoord + AxisStep;
  end;
  CurrentYCoord := AxisMini;
  while CurrentYCoord <= glDomain.CubeDepth*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsY);
    with DCCoordsY do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(-1, 0, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom;
        ModulateColor.AsWinColor := clLime;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentYCoord-AxisMini));
      end;
    end;
    CurrentYCoord := CurrentYCoord + AxisStep;
  end;
  { Draw Z }
  AxisMini:=-glDomain.CubeHeight*0.5;
  CurrentXCoord := -glDomain.CubeWidth*0.5;
  CurrentYCoord := -glDomain.CubeDepth*0.5;
  CurrentZCoord := -glDomain.CubeHeight*0.5;
  while CurrentZCoord <= glDomain.CubeHeight*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsZ);
    with DCCoordsZ do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, -1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlCenter;
        ModulateColor.AsWinColor := clBlue;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentZCoord-AxisMini));
      end;
    end;
    CurrentZCoord := CurrentZCoord + AxisStep;
  end;
  CurrentZCoord := AxisMini;
  while CurrentZCoord <= glDomain.CubeHeight*0.5 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsZ);
    with DCCoordsZ do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, 1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlCenter;
        ModulateColor.AsWinColor := clBlue;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentZCoord-AxisMini));
      end;
    end;
    CurrentZCoord := CurrentZCoord + AxisStep;
  end;
end;

end.