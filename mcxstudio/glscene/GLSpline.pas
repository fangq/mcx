//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Cubic spline interpolation functions

	 History :  
      10/12/14 - PW - Renamed Spline unit to GLSpline
      30/12/12 - PW - Restored CPP compatibility with record arrays
      08/07/04 - LR - Removed ../ from the GLScene.inc
      16/07/02 - Egg - Added methods to access slope per axis
	    28/05/00 - Egg - Javadocisation, minor changes & optimizations,
                           Renamed TSpline to TCubicSpline, added W component
                           and a bunch of helper methods
	    20/05/00 - RoC - Created, based on the C source code from Eric
	 
}
unit GLSpline;

interface

uses
  GLVectorGeometry;

{$I GLScene.inc}

type

   TCubicSplineMatrix = array of array [0..3] of Single;

   // TCubicSpline
   //
   { 3D cubic spline handler class.
      This class allows to describe and calculate values of a time-based,
      three-dimensionnal cubic spline.
      Cubic spline pass through all given points and tangent on point N is
      given by the (N-1) to (N+1) vector.
      Note : X, Y & Z are actually interpolated independently. }
   TCubicSpline = class (TObject)
      private
          
         matX, matY, matZ, matW : TCubicSplineMatrix;
         FNb : Integer;

      public
          
         { Creates the spline and declares interpolation points.
            Time references go from 0 (first point) to nb-1 (last point), the
            first and last reference matrices respectively are used when T is
            used beyond this range.
            Note : "nil" single arrays are accepted, in this case the axis is
            disabled and calculus will return 0 (zero) for this component. }
         constructor Create(const X, Y, Z, W : PFloatArray; const nb : Integer); {$ifdef CLR}unsafe;{$endif}
         destructor Destroy; override;

         { Calculates X component at time t. }
         function SplineX(const t : Single): Single;
         { Calculates Y component at time t. }
         function SplineY(const t : single): Single;
         { Calculates Z component at time t. }
         function SplineZ(const t : single): Single;
         { Calculates W component at time t. }
         function SplineW(const t : single): Single;

         { Calculates X and Y components at time t. }
         procedure SplineXY(const t : single; var X, Y : Single);
         { Calculates X, Y and Z components at time t. }
         procedure SplineXYZ(const t : single; var X, Y, Z : Single);
         { Calculates X, Y, Z and W components at time t. }
         procedure SplineXYZW(const t : single; var X, Y, Z, W : Single);

         { Calculates affine vector at time t. }
         function SplineAffineVector(const t : single) : TAffineVector; overload;
         { Calculates affine vector at time t. }
         procedure SplineAffineVector(const t : single; var vector : TAffineVector); overload;
         { Calculates vector at time t. }
         function SplineVector(const t : single) : TVector; overload;
         { Calculates vector at time t. }
         procedure SplineVector(const t : single; var vector : TVector); overload;

         { Calculates X component slope at time t. }
         function SplineSlopeX(const t : Single): Single;
         { Calculates Y component slope at time t. }
         function SplineSlopeY(const t : single): Single;
         { Calculates Z component slope at time t. }
         function SplineSlopeZ(const t : single): Single;
         { Calculates W component slope at time t. }
         function SplineSlopeW(const t : single): Single;
         { Calculates the spline slope at time t. }
         function SplineSlopeVector(const t : single) : TAffineVector; overload;

         { Calculates the intersection of the spline with the YZ plane.
            Returns True if an intersection was found. }
         function SplineIntersecYZ(X: Single; var Y, Z: Single): Boolean;
         { Calculates the intersection of the spline with the XZ plane.
            Returns True if an intersection was found. }
         function SplineIntersecXZ(Y: Single; var X, Z: Single): Boolean;
         { Calculates the intersection of the spline with the XY plane.
            Returns True if an intersection was found. }
         function SplineIntersecXY(Z: Single; var X, Y: Single): Boolean;
   end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

// VECCholeskyTriDiagResol
//
procedure VECCholeskyTriDiagResol(const b : array of Single; const nb : Integer;
                                  var Result : array of Single);
var
   Y, LDiag, LssDiag : array of Single;
   i, k, Debut, Fin: Integer;
begin
   Debut:=0;
   Fin:=nb-1;
   Assert(Length(b)>0);
   SetLength(LDiag, nb);
   SetLength(LssDiag, nb-1);
   LDiag[Debut]:=1.4142135; // = sqrt(2)
   LssDiag[Debut]:=1.0/1.4142135;
   for K:=Debut+1 to Fin-1 do begin
      LDiag[K]:=Sqrt(4-LssDiag[K-1]*LssDiag[K-1]);
      LssDiag[K]:=1.0/LDiag[K];
   end;
   LDiag[Fin]:=Sqrt(2-LssDiag[Fin-1]*LssDiag[Fin-1]);
   SetLength(Y, nb);
   Y[Debut]:=B[Debut]/LDiag[Debut];
   for I:=Debut+1 to Fin do
      Y[I]:=(B[I]-Y[I-1]*LssDiag[I-1])/LDiag[I];
   Assert(Length(Result)=nb);
   Result[Fin]:=Y[Fin]/LDiag[Fin];
   for i:=Fin-1 downto Debut do
      Result[I]:=(Y[I]-Result[I+1]*LssDiag[I])/LDiag[I];
end;

// MATInterpolationHermite
//
procedure MATInterpolationHermite(const ordonnees : PFloatArray; const nb : Integer;
                                  var Result : TCubicSplineMatrix); {$ifdef CLR}unsafe;{$endif}
var
   a, b, c, d : Single;
   i, n : Integer;
   bb, deriv : array of Single;
begin
   Result:=nil;
   if Assigned(Ordonnees) and (nb>0) then begin
      n:=nb-1;
      SetLength(bb, nb);
      bb[0]:=3*(ordonnees[1]-ordonnees[0]);
      bb[n]:=3*(ordonnees[n]-ordonnees[n-1]);
      for i:=1 to n-1 do
         bb[I]:=3*(ordonnees[I+1]-ordonnees[I-1]);
      SetLength(deriv, nb);
      VECCholeskyTriDiagResol(bb, nb, deriv);
      SetLength(Result, n);
      for i:=0 to n-1 do begin
         a:=ordonnees[I];
         b:=deriv[I];
         c:=3*(ordonnees[I+1]-ordonnees[I])-2*deriv[I]-deriv[I+1];
         d:=-2*(ordonnees[I+1]-ordonnees[I])+deriv[I]+deriv[I+1];
         Result[I][3]:=a+I*(I*(c-I*d)-b);
         Result[I][2]:=b+I*(3*I*d-2*c);
         Result[I][1]:=c-3*I*d;
         Result[I][0]:=d;
      end;
   end;
end;

// MATValeurSpline
//
function MATValeurSpline(const spline : TCubicSplineMatrix; const x : Single;
                         const nb : Integer) : Single;
var
   i : Integer;
begin
   if Length(Spline)>0 then begin
      if x<=0 then
         i:=0
      else if x>nb-1 then
         i:=nb-1
      else i:=Integer(Trunc(x));
      { TODO : the following line looks like a bug... }
      if i=(nb-1) then Dec(i);
      Result:=((spline[i][0]*x+spline[i][1])*x+spline[i][2])*x+spline[i][3];
   end else Result:=0;
end;

// MATValeurSplineSlope
//
function MATValeurSplineSlope(const spline : TCubicSplineMatrix; const x : Single;
                              const nb : Integer) : Single;
var
   i : Integer;
begin
   if Length(Spline)>0 then begin
      if x<=0 then
         i:=0
      else if x>nb-1 then
         i:=nb-1
      else i:=Integer(Trunc(x));
      { TODO : the following line looks like a bug... }
      if i=(nb-1) then Dec(i);
      Result:=(3*spline[i][0]*x+2*spline[i][1])*x+spline[i][2];
   end else Result:=0;
end;

// ------------------
// ------------------ TCubicSpline ------------------
// ------------------

// Create
//
constructor TCubicSpline.Create(const X, Y, Z, W: PFloatArray; const nb : Integer); {$ifdef CLR}unsafe;{$endif}
begin
   inherited Create;
   MATInterpolationHermite(X, nb, matX);
   MATInterpolationHermite(Y, nb, matY);
   MATInterpolationHermite(Z, nb, matZ);
   MATInterpolationHermite(W, nb, matW);
   FNb:=nb;
end;

// Destroy
//
destructor TCubicSpline.Destroy;
begin
   inherited Destroy;
end;

// SplineX
//
function TCubicSpline.SplineX(const t : single): Single;
begin
   Result:=MATValeurSpline(MatX, t, FNb);
end;

// SplineY
//
function TCubicSpline.SplineY(const t : single): Single;
begin
   Result:=MATValeurSpline(MatY, t, FNb);
end;

// SplineZ
//
function TCubicSpline.SplineZ(const t : single): Single;
begin
   Result:=MATValeurSpline(MatZ, t, FNb);
end;

// SplineW
//
function TCubicSpline.SplineW(const t : single): Single;
begin
   Result:=MATValeurSpline(MatW, t, FNb);
end;

// SplineXY
//
procedure TCubicSpline.SplineXY(const t : single; var X, Y : Single);
begin
   X:=MATValeurSpline(MatX, T, FNb);
   Y:=MATValeurSpline(MatY, T, FNb);
end;

// SplineXYZ
//
procedure TCubicSpline.SplineXYZ(const t : single; var X, Y, Z : Single);
begin
   X:=MATValeurSpline(MatX, T, FNb);
   Y:=MATValeurSpline(MatY, T, FNb);
   Z:=MATValeurSpline(MatZ, T, FNb);
end;

// SplineXYZW
//
procedure TCubicSpline.SplineXYZW(const t : single; var X, Y, Z, W : Single);
begin
   X:=MATValeurSpline(MatX, T, FNb);
   Y:=MATValeurSpline(MatY, T, FNb);
   Z:=MATValeurSpline(MatZ, T, FNb);
   W:=MATValeurSpline(MatW, T, FNb);
end;

// SplineAffineVector
//
function TCubicSpline.SplineAffineVector(const t : single) : TAffineVector;
begin
   Result.V[0]:=MATValeurSpline(MatX, t, FNb);
   Result.V[1]:=MATValeurSpline(MatY, t, FNb);
   Result.V[2]:=MATValeurSpline(MatZ, t, FNb);
end;

// SplineAffineVector
//
procedure TCubicSpline.SplineAffineVector(const t : single; var vector : TAffineVector);
begin
   vector.V[0]:=MATValeurSpline(MatX, t, FNb);
   vector.V[1]:=MATValeurSpline(MatY, t, FNb);
   vector.V[2]:=MATValeurSpline(MatZ, t, FNb);
end;

// SplineVector
//
function TCubicSpline.SplineVector(const t : single) : TVector;
begin
   Result.V[0]:=MATValeurSpline(MatX, t, FNb);
   Result.V[1]:=MATValeurSpline(MatY, t, FNb);
   Result.V[2]:=MATValeurSpline(MatZ, t, FNb);
   Result.V[3]:=MATValeurSpline(MatW, t, FNb);
end;

// SplineVector
//
procedure TCubicSpline.SplineVector(const t : single; var vector : TVector);
begin
   vector.V[0]:=MATValeurSpline(MatX, t, FNb);
   vector.V[1]:=MATValeurSpline(MatY, t, FNb);
   vector.V[2]:=MATValeurSpline(MatZ, t, FNb);
   vector.V[3]:=MATValeurSpline(MatW, t, FNb);
end;

// SplineSlopeX
//
function TCubicSpline.SplineSlopeX(const t : Single): Single;
begin
   Result:=MATValeurSplineSlope(MatX, t, FNb);
end;

// SplineSlopeY
//
function TCubicSpline.SplineSlopeY(const t : single): Single;
begin
   Result:=MATValeurSplineSlope(MatY, t, FNb);
end;

// SplineSlopeZ
//
function TCubicSpline.SplineSlopeZ(const t : single): Single;
begin
   Result:=MATValeurSplineSlope(MatZ, t, FNb);
end;

// SplineSlopeW
//
function TCubicSpline.SplineSlopeW(const t : single): Single;
begin
   Result:=MATValeurSplineSlope(MatW, t, FNb);
end;

// SplineSlopeVector
//
function TCubicSpline.SplineSlopeVector(const t : single) : TAffineVector;
begin
   Result.V[0]:=MATValeurSplineSlope(MatX, t, FNb);
   Result.V[1]:=MATValeurSplineSlope(MatY, t, FNb);
   Result.V[2]:=MATValeurSplineSlope(MatZ, t, FNb);
end;

// SplineIntersecYZ
//
function TCubicSpline.SplineIntersecYZ(X: Single; var Y, Z: Single): Boolean;
var
   Sup, Inf, Mid : Single;
   SSup, Sinf, Smid : Single;
begin
   Result:=False;

   Sup:=FNb;
   Inf:=0.0;

   Ssup:=SplineX(Sup);
   Sinf:=SplineX(Inf);
   if SSup>Sinf then begin
      if (SSup<X) or (Sinf>X) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineX(Mid);
         if X<SMid then begin
            SSup:=SMid;
            Sup:=Mid;
         end else begin
            Sinf:=SMid;
            Inf:=Mid;
         end;
      end;
      Y:=SplineY((Sup+Inf)*0.5);
      Z:=SplineZ((Sup+Inf)*0.5);
   end else begin
      if (Sinf<X) or (SSup>X) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineX(Mid);
         if X<SMid then begin
            Sinf:=SMid;
            Inf:=Mid;
         end else begin
            SSup:=SMid;
            Sup:=Mid;
         end;
      end;
      Y:=SplineY((Sup+Inf)*0.5);
      Z:=SplineZ((Sup+Inf)*0.5);
   end;
   Result:=True;
end;

// SplineIntersecXZ
//
function TCubicSpline.SplineIntersecXZ(Y: Single; var X, Z: Single): Boolean;
var
   Sup, Inf, Mid : Single;
   SSup, Sinf, Smid : Single;
begin
   Result:=False;

   Sup:=FNb;
   Inf:=0.0;

   Ssup:=SplineY(Sup);
   Sinf:=SplineY(Inf);
   if SSup>Sinf then begin
      if (SSup<Y) or (Sinf>Y) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineY(Mid);
         if Y<SMid then begin
            SSup:=SMid;
            Sup:=Mid;
         end else begin
            Sinf:=SMid;
            Inf:=Mid;
         end;
      end;
      X:=SplineX((Sup+Inf)*0.5);
      Z:=SplineZ((Sup+Inf)*0.5);
   end else begin
      if (Sinf<Y) or (SSup>Y) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineY(Mid);
         if Y<SMid then begin
            Sinf:=SMid;
            Inf:=Mid;
         end else begin
            SSup:=SMid;
            Sup:=Mid;
         end;
      end;
      X:=SplineX((Sup+Inf)*0.5);
      Z:=SplineZ((Sup+Inf)*0.5);
   end;
   Result:=True;
end;

// SplineIntersecXY
//
function TCubicSpline.SplineIntersecXY(Z: Single; var X, Y: Single): Boolean;
var
   Sup, Inf, Mid : Single;
   SSup, Sinf, Smid : Single;
begin
   Result:=False;

   Sup:=FNb;
   Inf:=0.0;

   Ssup:=SplineZ(Sup);
   Sinf:=SplineZ(Inf);
   if SSup>Sinf then begin
      if (SSup<Z) or (Sinf>Z) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineZ(Mid);
         if Z<SMid then begin
            SSup:=SMid;
            Sup:=Mid;
         end else begin
            Sinf:=SMid;
            Inf:=Mid;
         end;
      end;
      X:=SplineX((Sup+Inf)*0.5);
      Y:=SplineY((Sup+Inf)*0.5);
   end else begin
      if (Sinf<Z) or (SSup>Z) then Exit;
      while Abs(SSup-Sinf)>1e-4 do begin
         Mid:=(Sup+Inf)*0.5;
         SMid:=SplineZ(Mid);
         if Z<SMid then begin
            Sinf:=SMid;
            Inf:=Mid;
         end else begin
            SSup:=SMid;
            Sup:=Mid;
         end;
      end;
      X:=SplineX((Sup+Inf)*0.5);
      Y:=SplineY((Sup+Inf)*0.5);
   end;
   Result:=True;
end;

end.
