//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   "Alternate" OpenGL functions to handle multi-texturing.

   Using this functions allows specifying none/one/multiple ARB multi-texture
   coordinates with standard texture specification call.

   Before using any of the xglTexCoordXxxx fonctions, call one of the
   xglMapTexCoordToXxxx functions to establish the redirectors.

   This unit is Open-Source under MPL 
   Copyright 2001 - Eric Grange (egrange@glscene.org) 
   http://glscene.org

    History :  
       25/11/10 - Yar - Wrapped multitexturing in TGLMultitextureCoordinator class
       23/08/10 - Yar - Added OpenGLTokens to uses, replaced OpenGL1x functions to OpenGLAdapter
       29/03/10 - Yar - Replaced MULTITHREADOPENGL to GLS_MULTITHREAD (thanks Controler)
       16/03/07 - DaStr - Dropped Kylix support in favor of FPC
                             (thanks Burkhard Carstens) (BugTracekrID=1681585)
       08/07/04 - LR - Removed ../ from the GLScene.inc
       23/05/03 - EG - Support for arbitrary (complex) mappings
       01/02/03 - EG - Added State stack
       01/07/02 - EG - Added mtcmUndefined, fixed initial state
       03/01/02 - EG - Added xglDisableClientState
       26/01/02 - EG - Added xglBegin/EndUpdate mechanism
       21/12/01 - EG - Fixed xglTexCoordPointer and xglEnableClientState
       18/12/01 - EG - Added xglEnableClientState
       24/08/01 - EG - Now supports MULTITHREADOPENGL (same as OpenGL1x)
       17/08/01 - EG - Made declarations Kylix compatible (cdecl vs stdcall)
       16/08/01 - EG - Renamed xglMapTextCoordMode to xglMapTexCoordMode
       14/08/01 - EG - Added xglMapTexCoordToSecond
       21/02/01 - EG - Added TexGen and vertex arrays mappings
    
}
unit XOpenGL;

interface

{$I GLScene.inc}

uses
  OpenGLTokens,
  GLContext;

type
  TMapTexCoordMode = (mtcmUndefined, mtcmNull, mtcmMain, mtcmDual, mtcmSecond,
    mtcmArbitrary);

  TGLMultitextureCoordinator = class(TAbstractMultitextureCoordinator)
  private
    FMapTexCoordMode: TMapTexCoordMode;
    FSecondTextureUnitForbidden: Boolean;

    FUpdCount: Integer;
    FUpdNewMode: TMapTexCoordMode;
    FStateStack: array of TMapTexCoordMode;
    FComplexMapping: array of Cardinal;
    FComplexMappingN: Integer;
  public
    // Explicit texture coordinates specification
    TexCoord2f: procedure(s, t: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexCoord2fv: procedure(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexCoord3f: procedure(s, t, r: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexCoord3fv: procedure(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexCoord4f: procedure(s, t, r, q: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexCoord4fv: procedure(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}

    // TexGen texture coordinates specification
    TexGenf: procedure(coord, pname: TGLEnum; param: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexGenfv: procedure(coord, pname: TGLEnum; params: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexGeni: procedure(coord, pname: TGLEnum; param: TGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    TexGeniv: procedure(coord, pname: TGLEnum; params: PGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}

    // Vertex Arrays texture coordinates specification
    TexCoordPointer: procedure(size: TGLint; atype: TGLEnum; stride: TGLsizei;
      data: pointer);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    EnableClientState: procedure(aarray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    DisableClientState: procedure(aarray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}

    // Misc
    Enable: procedure(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
    Disable: procedure(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}

    constructor Create(AOwner: TGLContext); override;

    { TexCoord functions will be ignored. }
    procedure MapTexCoordToNull;
    { TexCoord functions will define the main texture coordinates. }
    procedure MapTexCoordToMain;
    { TexCoord functions will define the second texture unit coordinates. }
    procedure MapTexCoordToSecond;
    { TexCoord functions will define the two first texture units coordinates. }
    procedure MapTexCoordToDual;
    { TexCoord functions will define the specified texture units coordinates. }
    procedure MapTexCoordToArbitrary(const units: array of Cardinal); overload;
    procedure MapTexCoordToArbitrary(const bitWiseUnits: Cardinal); overload;
    procedure MapTexCoordToArbitraryAdd(const bitWiseUnits: Cardinal);

    { Defers Map calls execution until EndUpdate is met.
       Calls to Begin/EndUpdate may be nested. }
    procedure BeginUpdate;
    { Applies Map calls if there were any since BeginUpdate was invoked.
       Calls to Begin/EndUpdate may be nested. }
    procedure EndUpdate;

    { Saves XOpenGL State on the stack. }
    procedure PushState;
    { Restores XOpenGL State from the stack. }
    procedure PopState;

    { Whenever called, 2nd texture units changes will be forbidden to .
       Use this function when you're using the 2nd texture unit for your own
       purposes and don't want XOpenGL to alter it. }
    procedure ForbidSecondTextureUnit;
    { Allow XOpenGL to use the second texture unit again. }
    procedure AllowSecondTextureUnit;
    { Returns the complex mapping in bitwise form. }
    function GetBitWiseMapping: Cardinal;

    property MapTexCoordMode: TMapTexCoordMode read FMapTexCoordMode write FMapTexCoordMode;
    property SecondTextureUnitForbidden: Boolean read FSecondTextureUnitForbidden;
  end;

function xgl(): TGLMultitextureCoordinator;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

{$IFNDEF GLS_MULTITHREAD}
var
{$ELSE}
threadvar
{$ENDIF}
  vMTC : TGLMultitextureCoordinator;

function xgl(): TGLMultitextureCoordinator;
var
  RC: TGLContext;
begin
  RC := SafeCurrentGLContext;
  if not Assigned(vMTC) or (vMTC.FOwner <> RC) then
  begin
    vMTC := TGLMultitextureCoordinator(RC.MultitextureCoordinator);
  end;
  Result := vMTC;
end;

  // ------------------------------------------------------------------
  // Multitexturing coordinates duplication functions
  // ------------------------------------------------------------------

  // --------- Complex (arbitrary) mapping

procedure TexCoord2f_Arbitrary(s, t: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord2f(xgl.FComplexMapping[i], s, t);
end;

procedure TexCoord2fv_Arbitrary(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord2fv(xgl.FComplexMapping[i], v);
end;

procedure TexCoord3f_Arbitrary(s, t, r: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord3f(xgl.FComplexMapping[i], s, t, r);
end;

procedure TexCoord3fv_Arbitrary(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord3fv(xgl.FComplexMapping[i], v);
end;

procedure TexCoord4f_Arbitrary(s, t, r, q: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord4f(xgl.FComplexMapping[i], s, t, r, q);
end;

procedure TexCoord4fv_Arbitrary(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
    GL.MultiTexCoord4fv(xgl.FComplexMapping[i], v);
end;

procedure TexGenf_Arbitrary(coord, pname: TGLEnum; param: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.TexGenf(coord, pname, param);
  end;
end;

procedure TexGenfv_Arbitrary(coord, pname: TGLEnum; params: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.TexGenfv(coord, pname, params);
  end;
end;

procedure TexGeni_Arbitrary(coord, pname: TGLEnum; param: TGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.TexGeni(coord, pname, param);
  end;
end;

procedure TexGeniv_Arbitrary(coord, pname: TGLEnum; params: PGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.TexGeniv(coord, pname, params);
  end;
end;

procedure Enable_Arbitrary(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.Enable(cap);
  end;
end;

procedure Disable_Arbitrary(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    CurrentGLContext.GLStates.ActiveTexture := xgl.FComplexMapping[i];
    GL.Disable(cap);
  end;
end;

procedure TexCoordPointer_Arbitrary(size: TGLint; atype: TGLEnum; stride:
  TGLsizei; data: pointer);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    GL.ClientActiveTexture(xgl.FComplexMapping[i]);
    GL.TexCoordPointer(size, atype, stride, data);
  end;
end;

procedure EnableClientState_Arbitrary(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    GL.ClientActiveTexture(xgl.FComplexMapping[i]);
    GL.EnableClientState(aArray);
  end;
end;

procedure DisableClientState_Arbitrary(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
var
  i: Integer;
begin
  for i := 0 to xgl.FComplexMappingN do
  begin
    GL.ClientActiveTexture(xgl.FComplexMapping[i]);
    GL.DisableClientState(aArray);
  end;
end;

// --------- Second unit Texturing

procedure TexCoord2f_Second(s, t: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord2f(GL_TEXTURE1, s, t);
end;

procedure TexCoord2fv_Second(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord2fv(GL_TEXTURE1, v);
end;

procedure TexCoord3f_Second(s, t, r: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord3f(GL_TEXTURE1, s, t, r);
end;

procedure TexCoord3fv_Second(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord3fv(GL_TEXTURE1, v);
end;

procedure TexCoord4f_Second(s, t, r, q: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord4f(GL_TEXTURE1, s, t, r, q);
end;

procedure TexCoord4fv_Second(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.MultiTexCoord4fv(GL_TEXTURE1, v);
end;

procedure TexGenf_Second(coord, pname: TGLEnum; param: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.TexGenf(coord, pname, param);
end;

procedure TexGenfv_Second(coord, pname: TGLEnum; params: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.TexGenfv(coord, pname, params);
end;

procedure TexGeni_Second(coord, pname: TGLEnum; param: TGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.TexGeni(coord, pname, param);
end;

procedure TexGeniv_Second(coord, pname: TGLEnum; params: PGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.TexGeniv(coord, pname, params);
end;

procedure Enable_Second(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.Enable(cap);
end;

procedure Disable_Second(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  CurrentGLContext.GLStates.ActiveTexture := 1;
  GL.Disable(cap);
end;

procedure TexCoordPointer_Second(size: TGLint; atype: TGLEnum; stride:
  TGLsizei; data: pointer);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.TexCoordPointer(size, atype, stride, data);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

procedure EnableClientState_Second(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.EnableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

procedure DisableClientState_Second(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.DisableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

// --------- Dual Texturing

procedure TexCoord2f_Dual(s, t: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord2f(s, t);
  GL.MultiTexCoord2f(GL_TEXTURE1, s, t);
end;

procedure TexCoord2fv_Dual(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord2fv(v);
  GL.MultiTexCoord2fv(GL_TEXTURE1, v);
end;

procedure TexCoord3f_Dual(s, t, r: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord3f(s, t, r);
  GL.MultiTexCoord3f(GL_TEXTURE1, s, t, r);
end;

procedure TexCoord3fv_Dual(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord3fv(v);
  GL.MultiTexCoord3fv(GL_TEXTURE1, v);
end;

procedure TexCoord4f_Dual(s, t, r, q: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord4f(s, t, r, q);
  GL.MultiTexCoord4f(GL_TEXTURE1, s, t, r, q);
end;

procedure TexCoord4fv_Dual(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoord4fv(v);
  GL.MultiTexCoord4fv(GL_TEXTURE1, v);
end;

procedure TexGenf_Dual(coord, pname: TGLEnum; param: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.TexGenf(coord, pname, param);
    ActiveTexture := 1;
    GL.TexGenf(coord, pname, param);
  end;
end;

procedure TexGenfv_Dual(coord, pname: TGLEnum; params: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.TexGenfv(coord, pname, params);
    ActiveTexture := 1;
    GL.TexGenfv(coord, pname, params);
  end;
end;

procedure TexGeni_Dual(coord, pname: TGLEnum; param: TGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.TexGeni(coord, pname, param);
    ActiveTexture := 1;
    GL.TexGeni(coord, pname, param);
  end;
end;

procedure TexGeniv_Dual(coord, pname: TGLEnum; params: PGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.TexGeniv(coord, pname, params);
    ActiveTexture := 1;
    GL.TexGeniv(coord, pname, params);
  end;
end;

procedure Enable_Dual(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.Enable(cap);
    ActiveTexture := 1;
    GL.Enable(cap);
  end;
end;

procedure Disable_Dual(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  with CurrentGLContext.GLStates do
  begin
    ActiveTexture := 0;
    GL.Disable(cap);
    ActiveTexture := 1;
    GL.Disable(cap);
  end;
end;

procedure TexCoordPointer_Dual(size: TGLint; atype: TGLEnum; stride:
  TGLsizei; data: pointer);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.TexCoordPointer(size, atype, stride, data);
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.TexCoordPointer(size, atype, stride, data);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

procedure EnableClientState_Dual(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.EnableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.EnableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

procedure DisableClientState_Dual(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
  GL.DisableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE1);
  GL.DisableClientState(aArray);
  GL.ClientActiveTexture(GL_TEXTURE0);
end;

// --------- Null Texturing

procedure TexCoord2f_Null(s, t: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoord2fv_Null(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoord3f_Null(s, t, r: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoord3fv_Null(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoord4f_Null(s, t, r, q: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoord4fv_Null(v: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexGenf_Null(coord, pname: TGLEnum; param: TGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexGenfv_Null(coord, pname: TGLEnum; params: PGLfloat);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexGeni_Null(coord, pname: TGLEnum; param: TGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexGeniv_Null(coord, pname: TGLEnum; params: PGLint);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure Enable_Null(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure Disable_Null(cap: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure TexCoordPointer_Null(size: TGLint; atype: TGLEnum; stride:
  TGLsizei; data: pointer);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure EnableClientState_Null(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

procedure DisableClientState_Null(aArray: TGLEnum);
{$IFDEF MSWINDOWS} stdcall;
{$ENDIF}{$IFDEF unix} cdecl;
{$ENDIF}
begin
end;

// ------------------------------------------------------------------
// Redirections management functions
// ------------------------------------------------------------------

// BeginUpdate
//

procedure TGLMultitextureCoordinator.BeginUpdate;
begin
  if FUpdCount = 0 then
  begin
    FUpdCount := 1;
    FUpdNewMode := MapTexCoordMode;
  end
  else
    Inc(FUpdCount);
end;

// EndUpdate
//

procedure TGLMultitextureCoordinator.EndUpdate;
begin
  Dec(FUpdCount);
  if (FUpdCount = 0) and (FUpdNewMode <> MapTexCoordMode) then
  begin
    case FUpdNewMode of
      mtcmNull: MapTexCoordToNull;
      mtcmMain: MapTexCoordToMain;
      mtcmDual: MapTexCoordToDual;
      mtcmSecond: MapTexCoordToSecond;
      mtcmArbitrary: MapTexCoordToArbitrary(FComplexMapping);
    else
      Assert(False);
    end;
  end;
end;

// PushState
//

procedure TGLMultitextureCoordinator.PushState;
var
  i: Integer;
begin
  Assert(FUpdCount = 0);
  i := Length(FStateStack);
  SetLength(FStateStack, i + 1);
  FStateStack[i] := MapTexCoordMode;
end;

// PopState
//

procedure TGLMultitextureCoordinator.PopState;
var
  i: Integer;
begin
  Assert(FUpdCount = 0);
  i := Length(FStateStack) - 1;
  Assert(i >= 0);
  case FStateStack[i] of
    mtcmNull: MapTexCoordToNull;
    mtcmMain: MapTexCoordToMain;
    mtcmDual: MapTexCoordToDual;
    mtcmSecond: MapTexCoordToSecond;
    mtcmArbitrary: MapTexCoordToArbitrary(FComplexMapping);
  else
    Assert(False);
  end;
  SetLength(FStateStack, i);
end;

// ForbidSecondTextureUnit
//

procedure TGLMultitextureCoordinator.ForbidSecondTextureUnit;
begin
  FSecondTextureUnitForbidden := True;
end;

// AllowSecondTextureUnit
//

procedure TGLMultitextureCoordinator.AllowSecondTextureUnit;
begin
  FSecondTextureUnitForbidden := False;
end;

constructor TGLMultitextureCoordinator.Create(AOwner: TGLContext);
begin
  inherited Create(AOwner);
  FMapTexCoordMode := mtcmUndefined;
  MapTexCoordToNull;
end;

// MapTexCoordToNull
//

procedure TGLMultitextureCoordinator.MapTexCoordToNull;
begin
  if FUpdCount <> 0 then
    FUpdNewMode := mtcmNull
  else if MapTexCoordMode <> mtcmNull then
  begin
    MapTexCoordMode := mtcmNull;

    TexCoord2f := TexCoord2f_Null;
    TexCoord2fv := TexCoord2fv_Null;
    TexCoord3f := TexCoord3f_Null;
    TexCoord3fv := TexCoord3fv_Null;
    TexCoord4f := TexCoord4f_Null;
    TexCoord4fv := TexCoord4fv_Null;

    TexGenf := TexGenf_Null;
    TexGenfv := TexGenfv_Null;
    TexGeni := TexGeni_Null;
    TexGeniv := TexGeniv_Null;

    TexCoordPointer := TexCoordPointer_Null;
    EnableClientState := EnableClientState_Null;
    DisableClientState := DisableClientState_Null;

    Enable := Enable_Null;
    Disable := Disable_Null;
  end;
end;

// TexCoordMapToMain
//

procedure TGLMultitextureCoordinator.MapTexCoordToMain;
begin
  if FUpdCount <> 0 then
    FUpdNewMode := mtcmMain
  else if MapTexCoordMode <> mtcmMain then
  begin
    MapTexCoordMode := mtcmMain;

    TexCoord2f := GL.TexCoord2f;
    TexCoord2fv := GL.TexCoord2fv;
    TexCoord3f := GL.TexCoord3f;
    TexCoord3fv := GL.TexCoord3fv;
    TexCoord4f := GL.TexCoord4f;
    TexCoord4fv := GL.TexCoord4fv;

    TexGenf := GL.TexGenf;
    TexGenfv := GL.TexGenfv;
    TexGeni := GL.TexGeni;
    TexGeniv := GL.TexGeniv;

    TexCoordPointer := GL.TexCoordPointer;
    EnableClientState := GL.EnableClientState;
    DisableClientState := GL.DisableClientState;

    Enable := GL.Enable;
    Disable := GL.Disable;
  end;
end;

// TexCoordMapToSecond
//

procedure TGLMultitextureCoordinator.MapTexCoordToSecond;
begin
  if FSecondTextureUnitForbidden then
  begin
    MapTexCoordToNull;
    Exit;
  end;
  if FUpdCount <> 0 then
    FUpdNewMode := mtcmSecond
  else if MapTexCoordMode <> mtcmSecond then
  begin
    MapTexCoordMode := mtcmSecond;
    Assert(GL.ARB_multitexture);

    TexCoord2f := TexCoord2f_Second;
    TexCoord2fv := TexCoord2fv_Second;
    TexCoord3f := TexCoord3f_Second;
    TexCoord3fv := TexCoord3fv_Second;
    TexCoord4f := TexCoord4f_Second;
    TexCoord4fv := TexCoord4fv_Second;

    TexGenf := TexGenf_Second;
    TexGenfv := TexGenfv_Second;
    TexGeni := TexGeni_Second;
    TexGeniv := TexGeniv_Second;

    TexCoordPointer := TexCoordPointer_Second;
    EnableClientState := EnableClientState_Second;
    DisableClientState := DisableClientState_Second;

    Enable := Enable_Second;
    Disable := Disable_Second;
  end;
end;

// TexCoordMapToDual
//

procedure TGLMultitextureCoordinator.MapTexCoordToDual;
begin
  if FSecondTextureUnitForbidden then
  begin
    MapTexCoordToMain;
    Exit;
  end;
  if FUpdCount <> 0 then
    FUpdNewMode := mtcmDual
  else if MapTexCoordMode <> mtcmDual then
  begin
    MapTexCoordMode := mtcmDual;
    Assert(GL.ARB_multitexture);

    TexCoord2f := TexCoord2f_Dual;
    TexCoord2fv := TexCoord2fv_Dual;
    TexCoord3f := TexCoord3f_Dual;
    TexCoord3fv := TexCoord3fv_Dual;
    TexCoord4f := TexCoord4f_Dual;
    TexCoord4fv := TexCoord4fv_Dual;

    TexGenf := TexGenf_Dual;
    TexGenfv := TexGenfv_Dual;
    TexGeni := TexGeni_Dual;
    TexGeniv := TexGeniv_Dual;

    TexCoordPointer := TexCoordPointer_Dual;
    EnableClientState := EnableClientState_Dual;
    DisableClientState := DisableClientState_Dual;

    Enable := Enable_Dual;
    Disable := Disable_Dual;
  end;
end;

// MapTexCoordToArbitrary (array)
//

procedure TGLMultitextureCoordinator.MapTexCoordToArbitrary(const units: array of Cardinal);
var
  i, j, n: Integer;
begin
  n := Length(units);
  SetLength(FComplexMapping, n);
  j := 0;
  FComplexMappingN := n - 1;
  for i := 0 to FComplexMappingN do
  begin
    if (not FSecondTextureUnitForbidden) or (units[i] <> GL_TEXTURE1) then
    begin
      FComplexMapping[j] := units[i];
      Inc(j);
    end;
  end;

  if FUpdCount <> 0 then
    FUpdNewMode := mtcmArbitrary
  else if MapTexCoordMode <> mtcmArbitrary then
  begin

    MapTexCoordMode := mtcmArbitrary;
    Assert(GL.ARB_multitexture);

    TexCoord2f := TexCoord2f_Arbitrary;
    TexCoord2fv := TexCoord2fv_Arbitrary;
    TexCoord3f := TexCoord3f_Arbitrary;
    TexCoord3fv := TexCoord3fv_Arbitrary;
    TexCoord4f := TexCoord4f_Arbitrary;
    TexCoord4fv := TexCoord4fv_Arbitrary;

    TexGenf := TexGenf_Arbitrary;
    TexGenfv := TexGenfv_Arbitrary;
    TexGeni := TexGeni_Arbitrary;
    TexGeniv := TexGeniv_Arbitrary;

    TexCoordPointer := TexCoordPointer_Arbitrary;
    EnableClientState := EnableClientState_Arbitrary;
    DisableClientState := DisableClientState_Arbitrary;

    Enable := Enable_Arbitrary;
    Disable := Disable_Arbitrary;
  end;
end;

// MapTexCoordToArbitrary (bitwise)
//

procedure TGLMultitextureCoordinator.MapTexCoordToArbitrary(const bitWiseUnits: Cardinal);
var
  i, n: Integer;
  units: array of Cardinal;
begin
  n := 0;
  for i := 0 to 7 do
  begin
    if (bitWiseUnits and (1 shl i)) <> 0 then
      Inc(n);
  end;
  SetLength(units, n);
  n := 0;
  for i := 0 to 7 do
  begin
    if (bitWiseUnits and (1 shl i)) <> 0 then
    begin
      units[n] := GL_TEXTURE0 + i;
      Inc(n);
    end;
  end;
  MapTexCoordToArbitrary(units);
end;

// MapTexCoordToArbitrary (bitwise)
//

procedure TGLMultitextureCoordinator.MapTexCoordToArbitraryAdd(const bitWiseUnits: Cardinal);
var
  n: Cardinal;
begin
  n := GetBitWiseMapping;
  MapTexCoordToArbitrary(n or bitWiseUnits);
end;

// GetBitWiseMapping
//

function TGLMultitextureCoordinator.GetBitWiseMapping: Cardinal;
var
  i, n: Cardinal;
  mode: TMapTexCoordMode;
begin
  if FUpdCount > 0 then
    mode := FUpdNewMode
  else
    mode := MapTexCoordMode;
  n := 0;
  case mode of
    mtcmMain: n := 1;
    mtcmDual: n := 3;
    mtcmSecond: n := 2;
    mtcmArbitrary:
      begin
        for i := 0 to FComplexMappingN do
          n := n or (1 shl (FComplexMapping[i] - GL_TEXTURE0));
      end;
  else
    Assert(False);
  end;
  Result := n;
end;

initialization

  // Register class
  vMultitextureCoordinatorClass := TGLMultitextureCoordinator;

end.
