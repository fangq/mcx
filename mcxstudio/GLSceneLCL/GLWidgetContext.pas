//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Widget specific Context.
   GLWidgetContext replaces old GLLinGTKContext.

    History :  
       11/06/10 - Yar - Fixed uses section after lazarus-0.9.29.26033 release
       02/05/10 - Yar - Fixes for Linux x64
       21/04/10 - Yar - Fixed conditions
                           (by Rustam Asmandiarov aka Predator)
       06/04/10 - Yar - Added to GLScene
                           (Created by Rustam Asmandiarov aka Predator)
    
}
unit GLWidgetContext;

interface

{$I GLScene.inc}

uses
  Classes, SysUtils, LCLType,
  GLCrossPlatform, GLContext,
  {$IFDEF GLS_LOGGING}GLSLog,{$ENDIF}

  // Operation System
{$IFDEF MSWINDOWS}
  Windows, GLWin32Context, LMessages, LCLVersion,
{$ENDIF}

{$IFDEF UNIX}
{$IFDEF LINUX}
  GLGLXContext,
{$ENDIF}
{$IFDEF GLS_X11_SUPPORT}
  x, xlib, xutil,
{$ENDIF}
{$IFDEF Darwin}
  GLCarbonContext;
{$ENDIF}
{$IFDEF BSD}
{$MESSAGE Warn 'Needs to be implemented'}
{$ENDIF}
{$IFDEF SUNOS or SOLARIS}
{$MESSAGE Warn 'Needs to be implemented'}
{$ENDIF}
{$ENDIF}
{$IFDEF WINCE}
{$MESSAGE Warn 'Needs to be implemented'}
{$ENDIF}
{$IFDEF OS2}
{$MESSAGE Warn 'Needs to be implemented'}
{$ENDIF}

  //Widgets
{$IF  DEFINED(LCLwin32) or DEFINED(LCLwin64)}
  Controls, WSLCLClasses, Win32Int,
  Win32WSControls, Win32Proc, LCLMessageGlue;
{$ENDIF}

{$IFDEF LCLGTK2}
gtk2proc, gtk2, gdk2, gdk2x, gtk2def;
{$ENDIF}

{$IFDEF LCLGTK}
gtkproc, gtk, gtkdef, gdk;
{$ENDIF}

{$IFDEF LCLQT}
QT4, QTWidgets;
{$ENDIF}
{$IFDEF LCLfpgui}
{$MESSAGE Warn 'LCLfpgui: Needs to be implemented'}
{$ENDIF}
{$IFDEF LCLwinse}
{$MESSAGE Warn 'LCLwinse: Needs to be implemented'}
{$ENDIF}
{$IFDEF LCLcarbon}
//    {$MESSAGE Warn 'LCLcarbon: Needs to be implemented'}
{$ENDIF}

type
  //====================TGLWidgetContext=======================

  // Windows 2000\Xp\Vista\7 x32\64
{$IFDEF MSWINDOWS}
  TGLWidgetContext = class(TGLWin32Context)
  protected
     
    procedure DoGetHandles(outputDevice: HWND; out XWin: HWND);
      override;
  end;
{$ENDIF}
  // MacOS X
{$IFDEF Darwin}
  TGLWidgetContext = class(TGLCarbonContext)
  protected
     
   // procedure DoGetHandles(outputDevice: HWND; out XWin: HWND); override;
  end;
{$ENDIF}
  // Linux Ubuntu, Kubuntu,...
{$IFDEF LINUX}
  TGLWidgetContext = class(TGLGLXContext)
  protected
     
    procedure DoGetHandles(outputDevice: HWND; out XWin: HWND); override;
  end;
{$ENDIF}
{$IF DEFINED(LCLwin32) or DEFINED(LCLwin64)}
  TGLSOpenGLControl = class(TWin32WSWinControl)
  published
    class function CreateHandle(const AWinControl: TWinControl;
      const AParams: TCreateParams): HWND; override;
  end;
procedure GLRegisterWSComponent(aControl: TComponentClass);
{$ENDIF}

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

{$IFDEF MSWINDOWS}

procedure TGLWidgetContext.DoGetHandles(outputDevice: HWND; out XWin: HWND);
begin
{$IF  DEFINED(LCLwin32) or DEFINED(LCLwin64)}
  XWin := outputDevice;
{$IFDEF GLS_LOGGING}
  GLSLogger.LogInfo('GLWidgetContext: Widget->LCLwin32\64');
{$ENDIF}
{$ELSE}
{$MESSAGE Warn 'Needs to be implemented'}
{$ENDIF}
end;
{$ENDIF}

{$IFDEF LINUX}

procedure TGLWidgetContext.DoGetHandles(outputDevice: HWND; out XWin: HWND);
{$IF DEFINED(LCLGTK2) or DEFINED(LCLGTK)}
var
  vGTKWidget: PGTKWidget;
  ptr: Pointer;
{$ENDIF}
begin
{$IF DEFINED(LCLGTK2) or DEFINED(LCLGTK)}
  vGTKWidget := TGtkDeviceContext(outputDevice).Widget;
  if Assigned(vGTKWidget) then
    ptr := Pointer(vGTKWidget)
  else
    ptr := Pointer(outputDevice);
  vGTKWidget := GetFixedWidget(ptr);
  // Dirty workaround: force realize
  gtk_widget_realize(vGTKWidget);
{$ENDIF}
{$IFDEF LCLGTK2}
  gtk_widget_set_double_buffered(vGTKWidget, False);
  XWin := GDK_WINDOW_XWINDOW(PGdkDrawable(vGTKWidget^.window));
{$IFDEF GLS_LOGGING}
  GLSLogger.LogInfo('GLWidgetContext: Widget->LCLGTK2');
{$ENDIF}
{$ENDIF}
{$IFDEF LCLGTK}
  XWin := GDK_WINDOW_XWINDOW(PGdkWindowPrivate(vGTKWidget^.window));
{$IFDEF GLS_LOGGING}
  GLSLogger.LogInfo('GLWidgetContext: Widget->LCLGTK');
{$ENDIF}
{$ENDIF}
{$IFDEF LCLQT}
  //Need Test passable problem
  XWin := QWidget_winId(TQTWidget(outputDevice).widget);
{$IFDEF GLS_LOGGING}
  GLSLogger.LogInfo('GLWidgetContext: Widget->LCLQT');
{$ENDIF}
{$ENDIF}
{$IFDEF LCLfpgui}
{$MESSAGE Warn 'LCLfpgui: Needs to be implemented'}
{$ENDIF}
end;
{$ENDIF}

//MacOS X
//
{$IFDEF Darwin}

(*procedure TGLWidgetContext.DoGetHandles(outputDevice: HWND; out XWin: HWND);
begin
{$IFNDEF LCLcarbon}
  XWin := outputDevice;
  {$IFDEF GLS_LOGGING}
  GLSLogger.LogInfo('GLWidgetContext:DoGetHandles->Widget->LCLcarbon');
  {$ENDIF}
{$ENDIF}
end;    *)
{$ENDIF}

{$IF  DEFINED(LCLwin32) or DEFINED(LCLwin64)}
//Need to debug Viewer because there is a black square
//Необходимо для отладки Viewera так как появляется черный квадрат
//Заимствовано из TOpenGLControl

function GlWindowProc(Window: HWnd; Msg: UInt; WParam: Windows.WParam;
  LParam: Windows.LParam): LResult; stdcall;
var
  PaintMsg: TLMPaint;
  winctrl: TWinControl;
begin
  case Msg of
    WM_ERASEBKGND:
      begin
        Result := 0;
      end;
    WM_PAINT:
      begin
        winctrl := GetWin32WindowInfo(Window)^.WinControl;
        if Assigned(winctrl) then
        begin
          FillChar(PaintMsg, SizeOf(PaintMsg), 0);
          PaintMsg.Msg := LM_PAINT;
          PaintMsg.DC := WParam;
          DeliverMessage(winctrl, PaintMsg);
          Result := PaintMsg.Result;
        end
        else
          Result := WindowProc(Window, Msg, WParam, LParam);
      end;
  else
    Result := WindowProc(Window, Msg, WParam, LParam);
  end;
end;

class function TGLSOpenGLControl.CreateHandle(const AWinControl: TWinControl;
  const AParams: TCreateParams): HWND;
var
  Params: TCreateWindowExParams;
begin
  // general initialization of Params
  {$if (lcl_major = 0) and  (lcl_release <= 28) }
  PrepareCreateWindow(AWinControl, Params);
  {$ELSE}
  PrepareCreateWindow(AWinControl, AParams, Params);
  {$ENDIF}
  // customization of Params
  with Params do
  begin
    pClassName := @ClsName;
    WindowTitle := StrCaption;
    SubClassWndProc := @GlWindowProc;
  end;
  // create window
  FinishCreateWindow(AWinControl, Params, false);
  Result := Params.Window;
end;

procedure GLRegisterWSComponent(aControl: TComponentClass);
begin
  RegisterWSComponent(aControl, TGLSOpenGLControl);
end;
{$ENDIF}

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
initialization
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------

  RegisterGLContextClass(TGLWidgetContext);

end.
