//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Defines base vector types for use in Geometry.pas and OpenGL1x.pas.

   The sole aim of this unit is to limit dependency between the Geometry
   and OpenGL1x units by introducing the base compatibility types
   (and only the *base* types).

   Conventions: 
        d  is used for Double precision floating points values (64 bits)
        f  is used for Single precision floating points values (32 bits)
        i  is used for 32 bits signed integers (longint)
        s  is uses for 16 bits signed integers (smallint)
    

   Note : D3D types untested.

	 History :  
     10/12/14 - PW - Renamed from "VectorTypes.pas" to "GLVectorTypes.pas"
     18/12/12 - PW - Restored CPP compatibility: changed arrays to records
     21/02/11 - Yar - Added half and signed byte vectors
     03/03/07 - DaStr - Added TMatrix2[d/d/i/s/b/e/w/p] types
     13/01/07 - DaStr - Added T[Matrix/Vector][2/3/4][e/w/p] types
     19/12/04 - PhP - Added byte vectors
     02/08/04 - LR, YHC - BCB corrections: use record instead array
                             move PAffineVectorArray, PVectorArray and PMatrixArray
                             from GLVectorGeometry to this Unit
     28/06/04 - LR - Removed ..\ from the GLScene.inc
     24/08/03 - PhP - Added smallint vectors
     04/07/01 - EG - Creation
   
}
unit GLVectorTypes;

interface

{$I GLScene.inc}

uses
  GLCrossPlatform;

type
  //2
  TVector2d = record
    case Integer of
      0 : (V: array[0..1] of double);
      1 : (X: double;
           Y: double);
  end;
  TVector2f = record
    case Integer of
      0 : (V: array[0..1] of single);
      1 : (X,Y: single);
  end;
  TVector2h = record
    case Integer of
      0 : (V: array[0..1] of THalfFloat);
      1 : (X,Y: THalfFloat);
  end;
  TVector2i = record
    case Integer of
      0 : (V: array[0..1] of longint);
      1 : (X,Y: longint);
  end;
  TVector2ui = record
    case Integer of
      0 : (V: array[0..1] of longword);
      1 : (X,Y: longword);
  end;
  TVector2s = record
    case Integer of
      0 : (V: array[0..1] of smallint);
      1 : (X,Y: smallint);
  end;
  TVector2b = record
    case Integer of
      0 : (V: array[0..1] of byte);
      1 : (X,Y: byte);
  end;
  TVector2sb = record
    case Integer of
      0 : (V: array[0..1] of ShortInt);
      1 : (X,Y: ShortInt);
  end;
  TVector2e = record
    case Integer of
      0 : (V: array[0..1] of Extended);
      1 : (X,Y: Extended);
  end;
  TVector2w = record
    case Integer of
      0 : (V: array[0..1] of Word);
      1 : (X,Y: Word);
  end;
  TVector2p = record
    case Integer of
      0 : (V: array[0..1] of Pointer);
      1 : (X,Y: Pointer);
  end;

  //3
  TVector3d = record
    case Integer of
      0 : (V: array[0..2] of double);
      1 : (X,Y,Z: double);
  end;
  TVector3f = record
    case Integer of
      0 : (V: array[0..2] of single);
      1 : (X,Y,Z: single);
  end;
  TVector3h = record
    case Integer of
      0 : (V: array[0..2] of THalfFloat);
      1 : (X,Y,Z: THalfFloat);
  end;
  TVector3i = record
    case Integer of
      0 : (V: array[0..2] of longint);
      1 : (X,Y,Z: longint);
  end;
  TVector3ui = record
    case Integer of
      0 : (V: array[0..2] of Longword);
      1 : (X,Y,Z: Longword);
  end;
  TVector3s = record
    case Integer of
      0 : (V: array[0..2] of smallint);
      1 : (X,Y,Z: smallint);
  end;
  TVector3b = record
    case Integer of
      0 : (V: array[0..2] of byte);
      1 : (X,Y,Z: byte);
  end;
  TVector3sb = record
    case Integer of
      0 : (V: array[0..2] of ShortInt);
      1 : (X,Y,Z: ShortInt);
  end;
  TVector3e = record
    case Integer of
      0 : (V: array[0..2] of Extended);
      1 : (X,Y,Z: Extended);
  end;
  TVector3w = record
    case Integer of
      0 : (V: array[0..2] of Word);
      1 : (X,Y,Z: Word);
  end;
  TVector3p = record
    case Integer of
      0 : (V: array[0..2] of Pointer);
      1 : (X,Y,Z: Pointer);
  end;

  //4
  TVector4d = record
    case Integer of
      0 : (V: array[0..3] of double);
      1 : (X,Y,Z,W: double);
  end;
  TVector4f = record
    case Integer of
      0 : (V: array[0..3] of single);
      1 : (X,Y,Z,W: single);
  end;
  TVector4h = record
    case Integer of
      0 : (V: array[0..3] of THalfFloat);
      1 : (X,Y,Z,W: THalfFloat);
  end;
  TVector4i = record
    case Integer of
      0 : (V: array[0..3] of LongInt);
      1 : (X,Y,Z,W: longint);
  end;
  TVector4ui = record
    case Integer of
      0 : (V: array[0..3] of LongWord);
      1 : (X,Y,Z,W: LongWord);
  end;
  TVector4s = record
    case Integer of
      0 : (V: array[0..3] of SmallInt);
      1 : (X,Y,Z,W: SmallInt);
  end;
  TVector4b = record
    case Integer of
      0 : (V: array[0..3] of Byte);
      1 : (X,Y,Z,W: byte);
  end;
  TVector4sb = record
    case Integer of
      0 : (V: array[0..3] of ShortInt);
      1 : (X,Y,Z,W: ShortInt);
  end;
  TVector4e = record
    case Integer of
      0 : (V: array[0..3] of Extended);
      1 : (X,Y,Z,W: Extended);
  end;
  TVector4w = record
    case Integer of
      0 : (V: array[0..3] of Word);
      1 : (X,Y,Z,W: Word);
  end;
  TVector4p = record
    case Integer of
      0 : (V: array[0..3] of Pointer);
      1 : (X,Y,Z,W: Pointer);
  end;

 TMatrix2d = record
    case Integer of
      0 : (V: array[0..1] of TVector2d);
      1 : (X,Y: TVector2d);
  end;
  TMatrix2f = record
    case Integer of
      0 : (V: array[0..1] of TVector2f);
      1 : (X,Y: TVector2f);
  end;
  TMatrix2i = record
    case Integer of
      0 : (V: array[0..1] of TVector2i);
      1 : (X,Y: TVector2i);
  end;
  TMatrix2s = record
    case Integer of
      0 : (V: array[0..1] of TVector2s);
      1 : (X,Y: TVector2s);
  end;
  TMatrix2b = record
    case Integer of
      0 : (V: array[0..1] of TVector2b);
      1 : (X,Y: TVector2b);
  end;
  TMatrix2e = record
    case Integer of
      0 : (V: array[0..1] of TVector2e);
      1 : (X,Y: TVector2e);
  end;
  TMatrix2w = record
    case Integer of
      0 : (V: array[0..1] of TVector2w);
      1 : (X,Y: TVector2w);
  end;
  TMatrix2p = record
    case Integer of
      0 : (V: array[0..1] of TVector2p);
      1 : (X,Y: TVector2p);
  end;

  TMatrix3d = record
    case Integer of
      0 : (V: array[0..2] of TVector3d);
      1 : (X,Y,Z: TVector3d);
  end;
  TMatrix3f = record
    case Integer of
      0 : (V: array[0..2] of TVector3f);
      1 : (X,Y,Z: TVector3f);
  end;
  TMatrix3i = record
    case Integer of
      0 : (V: array[0..2] of TVector3i);
      1 : (X,Y,Z: TVector3i);
  end;
  TMatrix3s = record
    case Integer of
      0 : (V: array[0..2] of TVector3s);
      1 : (X,Y,Z: TVector3s);
  end;
  TMatrix3b = record
    case Integer of
      0 : (V: array[0..2] of TVector3b);
      1 : (X,Y,Z: TVector3b);
  end;
  TMatrix3e = record
    case Integer of
      0 : (V: array[0..2] of TVector3e);
      1 : (X,Y,Z: TVector3e);
  end;
  TMatrix3w = record
    case Integer of
      0 : (V: array[0..2] of TVector3w);
      1 : (X,Y,Z: TVector3w);
  end;
  TMatrix3p = record
    case Integer of
      0 : (V: array[0..2] of TVector3p);
      1 : (X,Y,Z: TVector3p);
  end;

  TMatrix4d = record
    case Integer of
      0 : (V: array[0..3] of TVector4d);
      1 : (X,Y,Z,W: TVector4d);
  end;
  TMatrix4f = record
    case Integer of
      0 : (V: array[0..3] of TVector4f);
      1 : (X,Y,Z,W: TVector4f);
  end;
  TMatrix4i = record
    case Integer of
      0 : (V: array[0..3] of TVector4i);
      1 : (X,Y,Z,W: TVector4i);
  end;
  TMatrix4s = record
    case Integer of
      0 : (V: array[0..3] of TVector4s);
      1 : (X,Y,Z,W: TVector4s);
  end;
  TMatrix4b = record
    case Integer of
      0 : (V: array[0..3] of TVector4b);
      1 : (X,Y,Z,W: TVector4b);
  end;  
  TMatrix4e = record
    case Integer of
      0 : (V: array[0..3] of TVector4e);
      1 : (X,Y,Z,W: TVector4e);
  end;
  TMatrix4w = record
    case Integer of
      0 : (V: array[0..3] of TVector4w);
      1 : (X,Y,Z,W: TVector4w);
  end;
  TMatrix4p = record
    case Integer of
      0 : (V: array[0..3] of TVector4p);
      1 : (X,Y,Z,W: TVector4p);
  end;

  TD3DVector = packed record
    case Integer of
      0 : (X: single;
           Y: single;
           Z: single);
      1 : (V: TVector3f);
  end;

  TD3DMatrix = packed record
    case Integer of
      0 : (_11, _12, _13, _14: single;
           _21, _22, _23, _24: single;
           _31, _32, _33, _34: single;
           _41, _42, _43, _44: single);
      1 : (M : TMatrix4f);
  end;

implementation

end.

