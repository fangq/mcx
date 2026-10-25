{ mcxstudio2 - asking whether OpenGL 3.3 will be there, before asking for it.

  On X11 the GL control finds out that there is no 3.3 core context -- a
  remote display, a virtual machine, an old Mesa -- only while the widget set
  is realising it, and says so by raising an exception from inside a GTK
  callback.  The LCL turns that into a modal "Press OK to ignore and risk
  data corruption", and an exception unwound through GTK's C frames leaves
  the window in no state to carry on in.

  So the same request is made here first, outside any widget: open the
  display, pick a window-capable RGBA configuration, and ask for a 3.3 core
  context with glXCreateContextAttribsARB, as the control will.  If that is
  refused, the view never creates the control and draws in software from the
  start.  An X error handler is installed for the duration, because a refusal
  can arrive as a BadMatch, and Xlib's default handler for one is to exit.

  Elsewhere -- Windows, macOS -- there is nothing to probe this way and the
  answer is yes; the view still falls back if the control fails later. }
unit mcxglprobe;

{$mode objfpc}{$H+}

interface

{ True if a 3.3 core context can be created; otherwise AWhy says why not. }
function McxProbeGL(out AWhy: string): Boolean;

implementation

{$IF DEFINED(UNIX) AND NOT DEFINED(DARWIN)}
uses
  ctypes, xlib, x, glx;

var
  GXFailed: Boolean;

function OnXError(ADisplay: PDisplay; AEvent: PXErrorEvent): cint; cdecl;
begin
  GXFailed := True;
  Result := 0;
end;

function McxProbeGL(out AWhy: string): Boolean;
const
  CfgAttr: array[0..8] of cint = (
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_RENDER_TYPE, GLX_RGBA_BIT,
    GLX_DOUBLEBUFFER, 1,
    GLX_DEPTH_SIZE, 24,
    0);
  CtxAttr: array[0..6] of cint = (
    GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
    GLX_CONTEXT_MINOR_VERSION_ARB, 3,
    GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
    0);
var
  Dpy: PDisplay;
  Screen, N, ErrBase, EvtBase: cint;
  Cfgs: PGLXFBConfig;
  Ctx: TGLXContext;
  Old: TXErrorHandler;
begin
  Result := False;
  AWhy := '';
  Dpy := XOpenDisplay(nil);
  if Dpy = nil then
  begin
    AWhy := 'no X display to ask';
    Exit;
  end;
  Old := XSetErrorHandler(@OnXError);
  GXFailed := False;
  try
    if not Assigned(glXQueryExtension) or
       not glXQueryExtension(Dpy, ErrBase, EvtBase) then
    begin
      AWhy := 'the X server has no GLX';
      Exit;
    end;
    Screen := DefaultScreen(Dpy);
    if not GLX_version_1_3(Dpy) then
    begin
      AWhy := 'GLX is older than 1.3';
      Exit;
    end;
    if not GLX_ARB_create_context(Dpy, Screen) or
       not GLX_ARB_create_context_profile(Dpy, Screen) then
    begin
      AWhy := 'GLX cannot create a core-profile context';
      Exit;
    end;
    N := 0;
    Cfgs := glXChooseFBConfig(Dpy, Screen, @CfgAttr[0], N);
    if (Cfgs = nil) or (N < 1) then
    begin
      AWhy := 'no double-buffered RGBA configuration with a depth buffer';
      Exit;
    end;
    try
      Ctx := glXCreateContextAttribsARB(Dpy, Cfgs[0], nil, True, @CtxAttr[0]);
      XSync(Dpy, False);
      if (Ctx = nil) or GXFailed then
      begin
        AWhy := 'the driver refused an OpenGL 3.3 core context';
        if Ctx <> nil then glXDestroyContext(Dpy, Ctx);
        Exit;
      end;
      glXDestroyContext(Dpy, Ctx);
      Result := True;
    finally
      XFree(Cfgs);
    end;
  finally
    XSync(Dpy, False);
    XSetErrorHandler(Old);
    XCloseDisplay(Dpy);
  end;
end;

{$ELSE}
function McxProbeGL(out AWhy: string): Boolean;
begin
  AWhy := '';
  Result := True;
end;
{$ENDIF}

end.
