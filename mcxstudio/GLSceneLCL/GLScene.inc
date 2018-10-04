//
// Heavily based on Delphi Free Stuff Include File
//                  by Brad Stowers: bstowers@pobox.com
//                  Delphi Free Stuff: http://www.delphifreestuff.com/
//
// History:
// 09/09/14 - PW - Dropped support for compilers older then Delphi7
// 10/11/12 - PW - Added support for Delphi and C++Builder with
//                 GLS_DELPHI and GLS_CPPB defines
// 02/04/10 - Yar - Added GLS_NO_INLINE option
// 13/12/09 - DaStr - Added GLS_MULTITHREAD option (thanks Controller)
// 10/11/09 - DaStr - Added Delphi 2010 support
// 07/11/09 - DaStr - Improved FPC compatibility (BugtrackerID = 2893580)
//                    Added GLS_NO_ASM option
// 12/09/07 - DaStr - Added GLS_DELPHI_OR_CPPB
// 30/03/07 - DaStr - Added GLS_ENABLE_UNSAFE_WARNINGS
// 25/03/07 - DaStr - Added UNIX and KYLIX defines
// 14/01/07 - DaStr - Added GLS_DELPHI/COMPILER_XX_DOWN defines
// 20/12/06 - DaStr - Added GLS_OPTIMIZATIONS option
// 08/03/06 - ur - Added Delphi 2006 detection
// 08/07/04 - LR - BCB5 and BCB6 detection
// 30/08/02 - EG - Delphi 7 detection
// 24/01/02 - EG - Linux/Kylix changes (Puthoon)
// 28/12/01 - EG - Added GLS_Graphics32_SUPPORT
// 27/07/01 - EG - Renamed and made GLScene-specific
// 26/06/01 - EG - Added Delphi6 stuff
//

{$DEFINE GLSCENE}

//
// Additionnal support & dependencies
//

// Activate support for Alex Denissov's Graphics32 library (http://www.g32.org)
{.$DEFINE GLS_Graphics32_SUPPORT}

// Activate combined support for Delphi and C++Builder
{$DEFINE GLS_DELPHI_OR_CPPB}

// Activate support for png images (using pngimage.pas library)
// _PngImage_SUPPORT is automatically defined if Delphi>=2009 (see below)
// Also, Define "Auto" Resize To Nearest 8 Multiple Size
{.$DEFINE GLS_PngImage_SUPPORT}
{.$DEFINE GLS_PngImage_RESIZENEAREST}

// Activate support for some optimizations by disabling checks that occur very often
// Warning: it will disable some options (like ShowAxis and OptSaveGLStack)
{.$DEFINE GLS_OPTIMIZATIONS}

// Enable all "Unsafe Type" / "Unsafe Code" compiler warnings
// May be useful when debugging.
{.$DEFINE GLS_ENABLE_UNSAFE_WARNINGS}

// De-Activate assembly routines in GLVectorGeometry.pas
{.$DEFINE GEOMETRY_NO_ASM}

// De-Activate assembly routines in other GLS units.
{.$DEFINE GLS_NO_ASM}

// De-Activate inline directive in GLS units.
{.$DEFINE GLS_NO_INLINE}

// De-Activate regions in GLS units.
{.$DEFINE GLS_NO_REGIONS}

// Activate Logging
{.$DEFINE GLS_LOGGING}

// Activate OpenGL debug mode
{.$DEFINE GLS_OPENGL_DEBUG}

// Activate Multithread support
{.$DEFINE GLS_MULTITHREAD}

// Enable service rendering context, which automatically activate GLS_MULTITHREAD
// This work only on windows platform and with modern compiler (D2009 and FPC}
{$DEFINE GLS_SERVICE_CONTEXT}

// Activate support X Window System
{.$DEFINE GLS_X11_SUPPORT}

// Activate OpenGL ES
{.$DEFINE GLS_OPENGL_ES}

//
// Platform Detection
//
{$IFDEF WIN64}
  {$DEFINE GLS_NO_ASM}
  {$DEFINE GEOMETRY_NO_ASM}
{$ENDIF}

{$IFNDEF MSWINDOWS}
  {$IFDEF WIN32}
    // MSWINDOWS wasn't defined in Delphi 5, but WIN32 was
    // so need to define MSWINDOWS here to provide Delphi 5 support
    {$DEFINE MSWINDOWS}
  {$ENDIF}
{$ENDIF}

{$IFDEF GLS_MULTITHREAD}
  {$DEFINE MULTITHREADOPENGL}
{$ENDIF}

{$IFDEF LINUX}
  {$DEFINE GEOMETRY_NO_ASM}
  {$DEFINE UNIX}
{$ENDIF}

// FPC
{$IFDEF FPC}
  {$UNDEF GLS_DELPHI_OR_CPPB}
  {.$DEFINE GLS_GENERIC_PREFIX}
  {$DEFINE GEOMETRY_NO_ASM}
  {$DEFINE GLS_NO_ASM}
  {$TYPEINFO ON}
  {$INLINE ON}
  {$H+} // Enable long strings.
  {$IFDEF FPC_HAS_TYPE_EXTENDED}
    {$DEFINE GLS_PLATFORM_HAS_EXTENDED}
  {$ENDIF}
{$ENDIF}


// by default, GLX is not supported on windows, although windows implementations
// do exist
{$IFDEF MSWINDOWS}
  {$DEFINE SUPPORT_WGL}
  {$UNDEF SUPPORT_GLX}
  {$UNDEF GLS_X11_SUPPORT}
{$ENDIF}

{$IFDEF Linux}
  {$UNDEF SUPPORT_WGL}
  {$DEFINE SUPPORT_GLX}
  {$DEFINE GLS_X11_SUPPORT}
{$ENDIF}

{$IFDEF DARWIN}
  {$UNDEF SUPPORT_WGL}
  {$UNDEF SUPPORT_GLX}
  {$UNDEF GLS_X11_SUPPORT}
{$ENDIF}

{$UNDEF GLS_UNSUPPORTED}


{$IFDEF GLS_UNSUPPORTED}
  Error: unsupported version, requires at least Delphi 5
{$ENDIF}

{$IFNDEF GLS_NO_INLINE}
  {$IFDEF GLS_COMPILER_2005_UP}
  {$DEFINE GLS_INLINE}

  {$ENDIF}
  {$IFDEF FPC}
  {$DEFINE GLS_INLINE}
  {$ENDIF}
{$ENDIF}

{$IFNDEF GLS_NO_REGIONS}
  {$IFDEF GLS_COMPILER_2005_UP}
  {$DEFINE GLS_REGIONS}
  {$ENDIF}
  {$IFDEF FPC}
  {$DEFINE GLS_REGIONS}
  {$ENDIF}
{$ENDIF}

// PNG is supported by Delphi since Dephi 2009
{$IFDEF GLS_DELPHI_2009_UP}
  {$DEFINE GLS_PngImage_SUPPORT}
{$ENDIF}

{$IFNDEF FPC}
  {$UNDEF GLS_SERVICE_CONTEXT}
{$ELSE}
  {$IFNDEF MSWINDOWS}
     {$UNDEF GLS_SERVICE_CONTEXT}
  {$ENDIF}
{$ENDIF}

{$IFDEF GLS_SERVICE_CONTEXT}
  {$DEFINE GLS_MULTITHREAD}
{$ENDIF}

