//
// This unit is part of the GLScene Project, http://glscene.org
//
{
    Platform independant viewer.

    History:
       13/06/12 - Yar - Bugfix synchronization control for Mac OS (thanks to pchev)	
       23/08/10 - Yar - Replaced OpenGL1x to OpenGLTokens
       30/04/10 - Yar - Added vertical synchronization cntrol for Linux (by Rustam Asmandiarov aka Predato) 
       17/09/07 - DaStr - Replaced $IFNDEF KYLIX to $IFDEF MSWINDOWS in 
                              SetupVSync() because wgl* functions are Windows-specific
       12/09/07 - DaStr - Fixed SetupVSync() function (Bugtracker ID = 1786279)
                             Made cross-platform code easier to read
       12/07/07 - DaStr - Added SetupVSync
       30/03/07 - DaStr - Another update after the previous fix (removed class())
                             Added TVSyncMode type and constants.
       24/03/07 - DaStr - Update for Windows after the previous fix
       21/03/07 - DaStr - Improved Cross-Platform compatibility
                             (thanks Burkhard Carstens) (Bugtracker ID = 1684432)
       17/03/07 - DaStr - Dropped Kylix support in favor of FPC (BugTrackerID=1681585)
       24/01/02 -  EG   - Initial version
}

unit GLViewer;

interface

{$I GLScene.inc}

uses
  GLContext,

  GLLCLViewer;
type

  TGLSceneViewer = GLLCLViewer.TGLSceneViewer;


procedure SetupVSync(const AVSyncMode : TVSyncMode);

implementation

uses
  OpenGLTokens, OpenGLAdapter;

procedure SetupVSync(const AVSyncMode : TVSyncMode);
{$IFDEF MSWINDOWS}
var
  I: Integer;
begin
  if GL.W_EXT_swap_control then
  begin
    I := GL.WGetSwapIntervalEXT;
    case AVSyncMode of
      vsmSync  : if I <> 1 then GL.WSwapIntervalEXT(1);
      vsmNoSync: if I <> 0 then GL.WSwapIntervalEXT(0);
    else
       Assert(False);
    end;
  end;
end;
{$ENDIF}
{$IFDEF Linux}
begin
  if GL.X_SGI_swap_control then
  begin
    case AVSyncMode of
      vsmSync  : GL.XSwapIntervalSGI(GL_True);
      vsmNoSync: GL.XSwapIntervalSGI(GL_False);
    else
       Assert(False);
    end;
  end;
end;
{$ENDIF}
{$IFDEF DARWIN}
var ctx: TAGLContext;
const ISync: Integer = 0;
      INoSync: Integer = 1;
begin
  if Assigned(GL) then
  begin
    ctx := GL.aGetCurrentContext();
    if Assigned(ctx) then
      case AVSyncMode of
        vsmSync  : GL.aSetInteger(ctx, AGL_SWAP_INTERVAL, @ISync); 
        vsmNoSync: GL.aSetInteger(ctx, AGL_SWAP_INTERVAL, @INoSync);
      else
         Assert(False);
      end;
  end;
end;
{$ENDIF}

end.
