//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Enhanced silhouette classes.

   Introduces more evolved/specific silhouette generation and management
   classes.

   CAUTION : both connectivity classes leak memory.

	 History :  
       04/11/10 - DaStr - Restored Delphi5 and Delphi6 compatibility  
       28/03/07 - DaStr - Renamed parameters in some methods
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678658)
       16/03/07 - DaStr - Added explicit pointer dereferencing
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
       26/09/03 - EG - Improved performance of TConnectivity data construction
       19/06/03 - MF - Split up Connectivity classes
       10/06/03 - EG - Creation (based on code from Mattias Fagerlund)
    
}
unit GLSilhouette;

interface

{$I GLScene.inc}

uses Classes, GLVectorGeometry, GLVectorLists, GLCrossPlatform;

type
   // TGLSilhouetteStyle
   //
   TGLSilhouetteStyle = (ssOmni, ssParallel);

   // TGLSilhouetteParameters
   //
   { Silouhette generation parameters.
      SeenFrom and LightDirection are expected in local coordinates. }
   TGLSilhouetteParameters = packed record
      SeenFrom, LightDirection : TAffineVector;
      Style : TGLSilhouetteStyle;
      CappingRequired : Boolean;
   end;

   // TGLSilhouette
   //
   { Base class storing a volume silhouette.
      Made of a set of indexed vertices defining an outline, and another set
      of indexed vertices defining a capping volume. Coordinates system
      is the object's unscaled local coordinates system. 
      This is the base class, you can use the TGLSilhouette subclass if you
      need some helper methods for generating the indexed sets. }
   TGLSilhouette = class
      private
          
         FVertices : TVectorList;
         FIndices : TIntegerList;
         FCapIndices : TIntegerList;
         FParameters : TGLSilhouetteParameters;

      protected
          
         procedure SetIndices(const value : TIntegerList);
         procedure SetCapIndices(const value : TIntegerList);
         procedure SetVertices(const value : TVectorList);

      public
          
         constructor Create; virtual;
         destructor Destroy; override;

         property Parameters : TGLSilhouetteParameters read FParameters write FParameters;
         property Vertices : TVectorList read FVertices write SetVertices;
         property Indices : TIntegerList read FIndices write SetIndices;
         property CapIndices : TIntegerList read FCapIndices write SetCapIndices;

         procedure Flush;
         procedure Clear;

         procedure ExtrudeVerticesToInfinity(const origin : TAffineVector);

         { Adds an edge (two vertices) to the silhouette.
            If TightButSlow is true, no vertices will be doubled in the
            silhouette list. This should only be used when creating re-usable
            silhouettes, because it's much slower. }
         procedure AddEdgeToSilhouette(const v0, v1 : TAffineVector;
                                       tightButSlow : Boolean);
         procedure AddIndexedEdgeToSilhouette(const Vi0, Vi1 : integer);

         { Adds a capping triangle to the silhouette.
            If TightButSlow is true, no vertices will be doubled in the
            silhouette list. This should only be used when creating re-usable
            silhouettes, because it's much slower. }
         procedure AddCapToSilhouette(const v0, v1, v2 : TAffineVector;
                                      tightButSlow : Boolean);

         procedure AddIndexedCapToSilhouette(const vi0, vi1, vi2 : Integer);
   end;

   // TBaseConnectivity
   //
   TBaseConnectivity = class
       protected
          FPrecomputeFaceNormal: boolean;

          function GetEdgeCount: integer; virtual;
          function GetFaceCount: integer; virtual;
       public
          property EdgeCount : integer read GetEdgeCount;
          property FaceCount : integer read GetFaceCount;

          property PrecomputeFaceNormal : boolean read FPrecomputeFaceNormal;
          procedure CreateSilhouette(const ASilhouetteParameters : TGLSilhouetteParameters; var ASilhouette : TGLSilhouette; AddToSilhouette : boolean); virtual;

          constructor Create(APrecomputeFaceNormal : boolean); virtual;
   end;

   // TConnectivity
   //
   TConnectivity = class(TBaseConnectivity)
       protected
          { All storage of faces and adges are cut up into tiny pieces for a reason,
          it'd be nicer with Structs or classes, but it's actually faster this way.
          The reason it's faster is because of less cache overwrites when we only
          access a tiny bit of a triangle (for instance), not all data.}
          FEdgeVertices : TIntegerList;
          FEdgeFaces : TIntegerList;
          FFaceVisible : TByteList;
          FFaceVertexIndex : TIntegerList;
          FFaceNormal : TAffineVectorList;
          FVertexMemory : TIntegerList;
          FVertices : TAffineVectorList;

          function GetEdgeCount: integer; override;
          function GetFaceCount: integer; override;

          function ReuseOrFindVertexID(const seenFrom : TAffineVector;
                     aSilhouette : TGLSilhouette; index : Integer) : Integer;
       public
          { Clears out all connectivity information. }
          procedure Clear; virtual;

          procedure CreateSilhouette(const silhouetteParameters : TGLSilhouetteParameters; var aSilhouette : TGLSilhouette; AddToSilhouette : boolean); override;

          function AddIndexedEdge(vertexIndex0, vertexIndex1 : integer; FaceID: integer) : integer;
          function AddIndexedFace(vi0, vi1, vi2 : integer) : integer;

          function AddFace(const vertex0, vertex1, vertex2 : TAffineVector) : integer;
          function AddQuad(const vertex0, vertex1, vertex2, vertex3 : TAffineVector) : integer;

          property EdgeCount : integer read GetEdgeCount;
          property FaceCount : integer read GetFaceCount;

          constructor Create(APrecomputeFaceNormal : boolean); override;
          destructor Destroy; override;
   end;

//-------------------------------------------------------------
//-------------------------------------------------------------
//-------------------------------------------------------------
implementation

//-------------------------------------------------------------
//-------------------------------------------------------------
//-------------------------------------------------------------

uses SysUtils;

// ------------------
// ------------------ TGLSilhouette ------------------
// ------------------

// Create
//
constructor TGLSilhouette.Create;
begin
   inherited;
   FVertices:=TVectorList.Create;
   FIndices:=TIntegerList.Create;
   FCapIndices:=TIntegerList.Create;
end;

// Destroy
//
destructor TGLSilhouette.Destroy;
begin
   FCapIndices.Free;
   FIndices.Free;
   FVertices.Free;
   inherited;
end;

// SetIndices
//
procedure TGLSilhouette.SetIndices(const value : TIntegerList);
begin
   FIndices.Assign(value);
end;

// SetCapIndices
//
procedure TGLSilhouette.SetCapIndices(const value : TIntegerList);
begin
   FCapIndices.Assign(value);
end;

// SetVertices
//
procedure TGLSilhouette.SetVertices(const value : TVectorList);
begin
   FVertices.Assign(value);
end;

// Flush
//
procedure TGLSilhouette.Flush;
begin
   FVertices.Flush;
   FIndices.Flush;
   FCapIndices.Flush;
end;

// Clear
//
procedure TGLSilhouette.Clear;
begin
   FVertices.Clear;
   FIndices.Clear;
   FCapIndices.Clear;
end;

// ExtrudeVerticesToInfinity
//
procedure TGLSilhouette.ExtrudeVerticesToInfinity(const origin : TAffineVector);
var
   i, nv, ni, nc, k : Integer;
   vList, vListN : PVectorArray;
   iList, iList2 : PIntegerArray;
begin
   // extrude vertices
   nv:=Vertices.Count;
   Vertices.Count:=2*nv;
   vList:=Vertices.List;
   vListN:=@vList[nv];
   for i:=0 to nv-1 do begin
      vListN^[i].V[3]:=0;
      VectorSubtract(PAffineVector(@vList[i])^, origin, PAffineVector(@vListN[i])^);
   end;
   // change silhouette indices to quad indices
   ni:=Indices.Count;
   Indices.Count:=2*ni;
   iList:=Indices.List;
   i:=ni-2; while i>=0 do begin
      iList2:=@iList^[2*i];
      iList2^[0]:=iList^[i];
      iList2^[1]:=iList^[i+1];
      iList2^[2]:=iList^[i+1]+nv;
      iList2^[3]:=iList^[i]+nv;
      Dec(i, 2);
   end;
   // add extruded triangles to capIndices
   nc:=CapIndices.Count;
   CapIndices.Capacity:=2*nc;
   iList:=CapIndices.List;
   for i:=nc-1 downto 0 do begin
      k:=iList^[i];
      CapIndices.Add(k);
      iList^[i]:=k+nv;
   end;
end;

// ------------------
// ------------------ TGLSilhouette ------------------
// ------------------

// AddEdgeToSilhouette
//
procedure TGLSilhouette.AddEdgeToSilhouette(const v0, v1 : TAffineVector;
                                            tightButSlow : Boolean);
begin
   if tightButSlow then
      Indices.Add(Vertices.FindOrAddPoint(v0),
                  Vertices.FindOrAddPoint(v1))
   else Indices.Add(Vertices.Add(v0, 1),
                    Vertices.Add(v1, 1));
end;

// AddIndexedEdgeToSilhouette
//
procedure TGLSilhouette.AddIndexedEdgeToSilhouette(const Vi0, Vi1 : integer);

begin
   Indices.Add(Vi0, Vi1);
end;

// AddCapToSilhouette
//
procedure TGLSilhouette.AddCapToSilhouette(const v0, v1, v2 : TAffineVector;
                                           tightButSlow : Boolean);
begin
   if tightButSlow then
      CapIndices.Add(Vertices.FindOrAddPoint(v0),
                     Vertices.FindOrAddPoint(v1),
                     Vertices.FindOrAddPoint(v2))
   else CapIndices.Add(Vertices.Add(v0, 1),
                       Vertices.Add(v1, 1),
                       Vertices.Add(v2, 1));
end;

// AddIndexedCapToSilhouette
//
procedure TGLSilhouette.AddIndexedCapToSilhouette(const vi0, vi1, vi2 : Integer);
begin
  CapIndices.Add(vi0, vi1, vi2);
end;

// ------------------
// ------------------ TBaseConnectivity ------------------
// ------------------

{ TBaseConnectivity }

constructor TBaseConnectivity.Create(APrecomputeFaceNormal: boolean);
begin
  FPrecomputeFaceNormal := APrecomputeFaceNormal;
end;

procedure TBaseConnectivity.CreateSilhouette(const ASilhouetteParameters : TGLSilhouetteParameters; var ASilhouette : TGLSilhouette; AddToSilhouette : boolean);
begin
  // Purely virtual!
end;

// ------------------
// ------------------ TConnectivity ------------------
// ------------------

function TBaseConnectivity.GetEdgeCount: integer;
begin
  result := 0;
end;

function TBaseConnectivity.GetFaceCount: integer;
begin
  result := 0;
end;

{ TConnectivity }

constructor TConnectivity.Create(APrecomputeFaceNormal : boolean);
begin
  FFaceVisible := TByteList.Create;

  FFaceVertexIndex := TIntegerList.Create;
  FFaceNormal := TAffineVectorList.Create;

  FEdgeVertices := TIntegerList.Create;
  FEdgeFaces := TIntegerList.Create;

  FPrecomputeFaceNormal := APrecomputeFaceNormal;

  FVertexMemory := TIntegerList.Create;

  FVertices := TAffineVectorList.Create;
end;

destructor TConnectivity.Destroy;
begin
  Clear;

  FFaceVisible.Free;
  FFaceVertexIndex.Free;
  FFaceNormal.Free;

  FEdgeVertices.Free;
  FEdgeFaces.Free;

  FVertexMemory.Free;

  if Assigned(FVertices) then
    FVertices.Free;

  inherited;
end;

procedure TConnectivity.Clear;
begin
  FEdgeVertices.Clear;
  FEdgeFaces.Clear;
  FFaceVisible.Clear;
  FFaceVertexIndex.Clear;
  FFaceNormal.Clear;
  FVertexMemory.Clear;

  if FVertices<>nil then
    FVertices.Clear;
end;

// CreateSilhouette
//
procedure TConnectivity.CreateSilhouette(
            const silhouetteParameters : TGLSilhouetteParameters;
            var aSilhouette : TGLSilhouette; addToSilhouette : boolean);
var
   i : Integer;
   vis : PIntegerArray;
   tVi0, tVi1 : Integer;
   faceNormal : TAffineVector;
   face0ID, face1ID : Integer;
   faceIsVisible : Boolean;
   verticesList : PAffineVectorArray;
begin
   if not Assigned(aSilhouette) then
      aSilhouette:=TGLSilhouette.Create
   else if not AddToSilhouette then
      aSilhouette.Flush;

   // Clear the vertex memory
   FVertexMemory.Flush;

   // Update visibility information for all Faces
   vis:=FFaceVertexIndex.List;
   for i:=0 to FaceCount-1 do begin
      if FPrecomputeFaceNormal then
         faceIsVisible:=(PointProject(silhouetteParameters.SeenFrom,
                                      FVertices.List^[vis^[0]],
                                      FFaceNormal.List^[i])
                         >=0)
      else begin
         verticesList:=FVertices.List;
         faceNormal:=CalcPlaneNormal(verticesList^[vis^[0]],
                                     verticesList^[vis^[1]],
                                     verticesList^[vis^[2]]);
         faceIsVisible:=(PointProject(silhouetteParameters.SeenFrom,
                                      FVertices.List^[vis^[0]], faceNormal)
                         >=0);
      end;

      FFaceVisible[i]:=Byte(faceIsVisible);

      if (not faceIsVisible) and silhouetteParameters.CappingRequired then
         aSilhouette.CapIndices.Add(ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[0]),
                                    ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[1]),
                                    ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[2]));
      vis:=@vis[3];      
   end;

   for i:=0 to EdgeCount-1 do  begin
      face0ID:=FEdgeFaces[i*2+0];
      face1ID:=FEdgeFaces[i*2+1];

      if (face1ID=-1) or (FFaceVisible.List^[face0ID]<>FFaceVisible.List^[face1ID]) then begin
         // Retrieve the two vertice values add add them to the Silhouette list
         vis:=@FEdgeVertices.List[i*2];

         // In this moment, we _know_ what vertex id the vertex had in the old
         // mesh. We can remember this information and re-use it for a speedup
         if FFaceVisible.List^[Face0ID]=0 then begin
            tVi0 := ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[0]);
            tVi1 := ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[1]);
            aSilhouette.Indices.Add(tVi0, tVi1);
         end else if Face1ID>-1 then begin
            tVi0 := ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[0]);
            tVi1 := ReuseOrFindVertexID(silhouetteParameters.SeenFrom, aSilhouette, vis^[1]);
            aSilhouette.Indices.Add(tVi1, tVi0);
         end;
      end;
   end;
end;

function TConnectivity.GetEdgeCount: integer;
begin
  result := FEdgeVertices.Count div 2;
end;

function TConnectivity.GetFaceCount: integer;
begin
  result := FFaceVisible.Count;
end;

// ReuseOrFindVertexID
//
function TConnectivity.ReuseOrFindVertexID(
     const seenFrom : TAffineVector; aSilhouette : TGLSilhouette; index : Integer) : Integer;
var
   pMemIndex : PInteger;
   memIndex, i : Integer;
   oldCount : Integer;
   list : PIntegerArray;
begin
   if index>=FVertexMemory.Count then begin
      oldCount:=FVertexMemory.Count;
      FVertexMemory.Count:=index+1;

      list:=FVertexMemory.List;
      for i:=OldCount to FVertexMemory.Count-1 do
         list^[i]:=-1;
   end;

   pMemIndex:=@FVertexMemory.List[index];

   if pMemIndex^=-1 then begin
      // Add the "near" vertex
      memIndex:=aSilhouette.Vertices.Add(FVertices.List^[index], 1);
      pMemIndex^:=memIndex;
      Result:=memIndex;
   end else Result:=pMemIndex^;
end;

// AddIndexedEdge
//
function TConnectivity.AddIndexedEdge(
            vertexIndex0, vertexIndex1 : Integer; faceID : Integer) : Integer;
var
   i : Integer;
   edgesVertices : PIntegerArray;
begin
   // Make sure that the edge doesn't already exists
   edgesVertices:=FEdgeVertices.List;
   for i:=0 to EdgeCount-1 do begin
      // Retrieve the two vertices in the edge
      if    ((edgesVertices^[0]=vertexIndex0) and (edgesVertices^[1]=vertexIndex1))
         or ((edgesVertices^[0]=vertexIndex1) and (edgesVertices^[1]=vertexIndex0)) then begin
         // Update the second Face of the edge and we're done (this _MAY_
         // overwrite a previous Face in a broken mesh)
         FEdgeFaces[i*2+1]:=faceID;
         Result:=i*2+1;
         Exit;
      end;
      edgesVertices:=@edgesVertices[2];
   end;

   // No edge was found, create a new one
   FEdgeVertices.Add(vertexIndex0, vertexIndex1);
   FEdgeFaces.Add(faceID, -1);

   Result:=EdgeCount-1;
end;

// AddIndexedFace
//
function TConnectivity.AddIndexedFace(vi0, vi1, vi2 : Integer) : Integer;
var
   faceID : integer;
begin
   FFaceVertexIndex.Add(vi0, vi1, vi2);

   if FPrecomputeFaceNormal then
      FFaceNormal.Add(CalcPlaneNormal(FVertices.List^[Vi0],
                                      FVertices.List^[Vi1],
                                      FVertices.List^[Vi2]));

   faceID:=FFaceVisible.Add(0);

   AddIndexedEdge(vi0, vi1, faceID);
   AddIndexedEdge(vi1, vi2, faceID);
   AddIndexedEdge(vi2, vi0, faceID);

   result:=faceID;
end;

function TConnectivity.AddFace(const Vertex0, Vertex1, Vertex2: TAffineVector) : integer;
var
  Vi0, Vi1, Vi2 : integer;
begin
  Vi0 := FVertices.FindOrAdd(Vertex0);
  Vi1 := FVertices.FindOrAdd(Vertex1);
  Vi2 := FVertices.FindOrAdd(Vertex2);

  result := AddIndexedFace(Vi0, Vi1, Vi2);
end;

function TConnectivity.AddQuad(const Vertex0, Vertex1, Vertex2,
  Vertex3: TAffineVector): integer;
var
  Vi0, Vi1, Vi2, Vi3 : integer;
begin
  Vi0 := FVertices.FindOrAdd(Vertex0);
  Vi1 := FVertices.FindOrAdd(Vertex1);
  Vi2 := FVertices.FindOrAdd(Vertex2);
  Vi3 := FVertices.FindOrAdd(Vertex3);

  // First face
  result := AddIndexedFace(Vi0, Vi1, Vi2);

  // Second face
  AddIndexedFace(Vi2, Vi3, Vi0);
end;

end.
