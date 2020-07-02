//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Misc. lists of vectors and entities

    History :  
       10/12/14 - PW - Renamed VectorList unit to GLVectorList
       23/02/11 - Yar - Added Revision mechanism to TAffineVectorList
       15/12/10 - DaStr - Added Min() and Max() for TSingleList and TDoubleList
       04/11/10 - DaStr - Restored Delphi5 and Delphi6 compatibility
       24/08/10 - Yar - Added to T4ByteList more overload of Add method
       11/06/10 - Yar - Bugfixed binary reading TTexPointList for FPC
       20/05/10 - Yar - Fixes for Linux x64
       27/02/10 - Yar - Added TLongWordList
       06/02/10 - Yar - Added methods to TSingleList
                           Added T4ByteList
       25/11/09 - DanB - Fixed FastQuickSortLists for 64bit (thanks YarUnderoaker)
                            ASM code protected with IFDEFs
       16/10/08 - UweR - Compatibility fix for Delphi 2009
       01/03/08 - DaStr - Added Borland-style persistency support to TBaseList
       29/03/07 - DaStr - Added more explicit pointer dereferencing
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
       28/03/07 - DaStr - Renamed parameters in some methods
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678658)
       25/01/07 - DaStr - Reformated code according to VCL standard
                             Added explicit pointer dereferencing
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
       23/01/07 - fig - Added FindOrAdd() or IndexOf() to TTexpointList
       16/01/07 - DaStr - Added TDoubleList
       28/06/04 - LR - Removed ..\ from the GLScene.inc
       03/09/03 - EG - Added TBaseList.Move, faster TIntegerList.Offset
       22/08/03 - EG - Faster FastQuickSortLists
       13/08/03 - SG - Added TQuaternionList
       05/06/03 - EG - Added MinInteger, some TIntegerList optimizations
       03/06/03 - EG - Added TIntegerList.BinarySearch and AddSorted (Mattias Fagerlund)
       22/01/03 - EG - Added AddIntegers
       20/01/03 - EG - Added TIntegerList.SortAndRemoveDuplicates
       22/10/02 - EG - Added TransformXxxx to TAffineVectorList
       04/07/02 - EG - Fixed TIntegerList.Add( 2 at once )
       15/06/02 - EG - Added TBaseListOption stuff
       28/05/02 - EG - TBaseList.SetCount now properly resets new items
       23/02/02 - EG - Added TBaseList.UseMemory
       20/01/02 - EG - Now uses new funcs Add/ScaleVectorArray and VectorArrayAdd
       06/12/01 - EG - Added Sort & MaxInteger to TIntegerList
       04/12/01 - EG - Added TIntegerList.IndexOf
       18/08/01 - EG - Fixed TAffineVectorList.Add (list)
       03/08/01 - EG - Added TIntegerList.AddSerie
       19/07/01 - EG - Added TAffineVectorList.Add (list variant)
       18/03/01 - EG - Additions and enhancements
       16/03/01 - EG - Introduced new PersistentClasses
       04/03/01 - EG - Optimized TAffineVectorList.Normalize (x2 speed on K7)
       26/02/01 - EG - VectorArrayLerp 3DNow optimized (x2 speed on K7)
       08/08/00 - EG - Added TSingleList
     20/07/00 - EG - Creation
  
}
unit GLVectorLists;

interface

{$I GLScene.inc}

uses
  Classes, SysUtils,
  //GLScene
  GLVectorTypes, GLVectorGeometry, GLPersistentClasses, GLCrossPlatform;

type
  // TBaseListOption
  //
  TBaseListOption = (bloExternalMemory, bloSetCountResetsMemory);
  TBaseListOptions = set of TBaseListOption;

  // TBaseList
  //
  { Base class for lists, introduces common behaviours. }
  TBaseList = class(TPersistentObject)
  private
     
    FCount: Integer;
    FCapacity: Integer;
    FGrowthDelta: Integer;
    FBufferItem: PByteArray;
    FOptions: TBaseListOptions;
    FRevision: LongWord;
    FTagString: string;
  protected
     
    // The base list pointer (untyped)
    FBaseList: GLVectorGeometry.PByteArray;
    // Must be defined by all subclasses in their constructor(s)
    FItemSize: Integer;

    procedure SetCount(Val: Integer);
        { Only function where list may be alloc'ed & freed.
           Resizes the array pointed by FBaseList, adjust the subclass's
           typed pointer accordingly if any. }
    procedure SetCapacity(NewCapacity: Integer); virtual;
    function BufferItem: PByteArray;
    function GetSetCountResetsMemory: Boolean;
    procedure SetSetCountResetsMemory(const Val: Boolean);

    // Borland-style persistency support.
    procedure ReadItemsData(AReader : TReader); virtual;
    procedure WriteItemsData(AWriter : TWriter); virtual;
    procedure DefineProperties(AFiler: TFiler); override;
  public
     
    constructor Create; override;
    destructor Destroy; override;
    procedure Assign(Src: TPersistent); override;

    procedure WriteToFiler(writer: TVirtualWriter); override;
    procedure ReadFromFiler(reader: TVirtualReader); override;

    procedure AddNulls(nbVals: Cardinal);
    procedure InsertNulls(Index: Integer; nbVals: Cardinal);

    procedure AdjustCapacityToAtLeast(const size: Integer);
    function DataSize: Integer;
        { Tell the list to use the specified range instead of its own.
           rangeCapacity should be expressed in bytes.
           The allocated memory is NOT managed by the list, current content
           if copied to the location, if the capacity is later changed, regular
           memory will be allocated, and the specified range no longer used. }
    procedure UseMemory(rangeStart: Pointer; rangeCapacity: Integer);
    { Empties the list without altering capacity. }
    procedure Flush;
    { Empties the list and release. }
    procedure Clear;

    procedure Delete(Index: Integer);
    procedure DeleteItems(Index: Integer; nbVals: Cardinal);
    procedure Exchange(index1, index2: Integer);
    procedure Move(curIndex, newIndex: Integer);
    procedure Reverse;

        { Nb of items in the list.
           When assigning a Count, added items are reset to zero. }
    property Count: Integer read FCount write SetCount;
        { Current list capacity.
           Not persistent. }
    property Capacity: Integer read FCapacity write SetCapacity;
        { List growth granularity.
                 Not persistent. }
    property GrowthDelta: Integer read FGrowthDelta write FGrowthDelta;
        { If true (default value) adjusting count will reset added values.
           Switching this option to true will turn off this memory reset,
           which can improve performance is that having empty values isn't
           required. }
    property SetCountResetsMemory: Boolean read GetSetCountResetsMemory write SetSetCountResetsMemory;
    property TagString: string read FTagString write FTagString;
    { Increase by one after every content changes. }
    property Revision: LongWord read FRevision write FRevision;
  end;

  // TBaseVectorList
  //
  { Base class for vector lists, introduces common behaviours. }
  TBaseVectorList = class(TBaseList)
  private
     
  protected
     
    function GetItemAddress(Index: Integer): PFloatArray;

  public
     
    procedure WriteToFiler(writer: TVirtualWriter); override;
    procedure ReadFromFiler(reader: TVirtualReader); override;

    procedure GetExtents(out min, max: TAffineVector); dynamic;
    function Sum: TAffineVector; dynamic;
    procedure Normalize; dynamic;
    function MaxSpacing(list2: TBaseVectorList): Single; dynamic;
    procedure Translate(const delta: TAffineVector); overload; dynamic;
    procedure Translate(const delta: TBaseVectorList); overload; dynamic;
    procedure TranslateInv(const delta: TBaseVectorList); overload; dynamic;

        { Replace content of the list with lerp results between the two given lists.
           Note: you can't Lerp with Self!!! }
    procedure Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single); dynamic; abstract;
        { Replace content of the list with angle lerp between the two given lists.
           Note: you can't Lerp with Self!!! }
    procedure AngleLerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
    procedure AngleCombine(const list1: TBaseVectorList; intensity: Single);
        { Linear combination of Self with another list.
           Self[i]:=Self[i]+list2[i]*factor }
    procedure Combine(const list2: TBaseVectorList; factor: Single); dynamic;

    property ItemAddress[Index: Integer]: PFloatArray read GetItemAddress;

  end;

  // TAffineVectorList
  //
  { A list of TAffineVector.
   Similar to TList, but using TAffineVector as items.
       The list has stack-like push/pop methods. }
  TAffineVectorList = class(TBaseVectorList)
  private
     
    FList: PAffineVectorArray;

  protected
     
    function Get(Index: Integer): TAffineVector;
    procedure Put(Index: Integer; const item: TAffineVector);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: TAffineVector): Integer; overload;
    function Add(const item: TVector): Integer; overload;
    procedure Add(const i1, i2: TAffineVector); overload;
    procedure Add(const i1, i2, i3: TAffineVector); overload;
    function Add(const item: TVector2f): Integer; overload;
    function Add(const item: TTexPoint): Integer; overload;
    function Add(const X, Y: Single): Integer; overload;
    function Add(const X, Y, Z: Single): Integer; overload;
    function Add(const X, Y, Z: Integer): Integer; overload;
    function AddNC(const X, Y, Z: Integer): Integer; overload;
    function Add(const xy: PIntegerArray; const Z: Integer): Integer; overload;
    function AddNC(const xy: PIntegerArray; const Z: Integer): Integer; overload;
    procedure Add(const list: TAffineVectorList); overload;
    procedure Push(const Val: TAffineVector);
    function Pop: TAffineVector;
    procedure Insert(Index: Integer; const item: TAffineVector);
    function IndexOf(const item: TAffineVector): Integer;
    function FindOrAdd(const item: TAffineVector): Integer;

    property Items[Index: Integer]: TAffineVector read Get write Put; default;
    property List: PAffineVectorArray read FList;

    procedure Translate(const delta: TAffineVector); overload; override;
    procedure Translate(const delta: TAffineVector; base, nb: Integer); overload;

    // Translates the given item
    procedure TranslateItem(Index: Integer; const delta: TAffineVector);
    // Translates given items
    procedure TranslateItems(Index: Integer; const delta: TAffineVector; nb: Integer);
    // Combines the given item
    procedure CombineItem(Index: Integer; const vector: TAffineVector; const f: Single);

        { Transforms all items by the matrix as if they were points.
           ie. the translation component of the matrix is honoured. }
    procedure TransformAsPoints(const matrix: TMatrix);
        { Transforms all items by the matrix as if they were vectors.
           ie. the translation component of the matrix is not honoured. }
    procedure TransformAsVectors(const matrix: TMatrix); overload;
    procedure TransformAsVectors(const matrix: TAffineMatrix); overload;

    procedure Normalize; override;
    procedure Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single); override;

    procedure Scale(factor: Single); overload;
    procedure Scale(const factors: TAffineVector); overload;
  end;

  // TVectorList
  //
  { A list of TVector.
   Similar to TList, but using TVector as items.
       The list has stack-like push/pop methods. }
  TVectorList = class(TBaseVectorList)
  private
     
    FList: PVectorArray;

  protected
     
    function Get(Index: Integer): TVector;
    procedure Put(Index: Integer; const item: TVector);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: TVector): Integer; overload;
    function Add(const item: TAffineVector; w: Single): Integer; overload;
    function Add(const X, Y, Z, w: Single): Integer; overload;
    procedure Add(const i1, i2, i3: TAffineVector; w: Single); overload;
    function AddVector(const item: TAffineVector): Integer; overload;
    function AddPoint(const item: TAffineVector): Integer; overload;
    function AddPoint(const X, Y: Single; const Z: Single = 0): Integer; overload;
    procedure Push(const Val: TVector);
    function Pop: TVector;
    function IndexOf(const item: TVector): Integer;
    function FindOrAdd(const item: TVector): Integer;
    function FindOrAddPoint(const item: TAffineVector): Integer;
    procedure Insert(Index: Integer; const item: TVector);

    property Items[Index: Integer]: TVector read Get write Put; default;
    property List: PVectorArray read FList;

    procedure Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single); override;
  end;

  // TTexPointList
  //
  { A list of TTexPoint.
   Similar to TList, but using TTexPoint as items.
       The list has stack-like push/pop methods. }
  TTexPointList = class(TBaseVectorList)
  private
     
    FList: PTexPointArray;

  protected
     
    function Get(Index: Integer): TTexPoint;
    procedure Put(Index: Integer; const item: TTexPoint);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function IndexOf(const item: TTexpoint): Integer;
    function FindOrAdd(const item: TTexpoint): Integer;

    function Add(const item: TTexPoint): Integer; overload;
    function Add(const item: TVector2f): Integer; overload;
    function Add(const texS, Text: Single): Integer; overload;
    function Add(const texS, Text: Integer): Integer; overload;
    function AddNC(const texS, Text: Integer): Integer; overload;
    function Add(const texST: PIntegerArray): Integer; overload;
    function AddNC(const texST: PIntegerArray): Integer; overload;
    procedure Push(const Val: TTexPoint);
    function Pop: TTexPoint;
    procedure Insert(Index: Integer; const item: TTexPoint);

    property Items[Index: Integer]: TTexPoint read Get write Put; default;
    property List: PTexPointArray read FList;

    procedure Translate(const delta: TTexPoint);
    procedure ScaleAndTranslate(const scale, delta: TTexPoint); overload;
    procedure ScaleAndTranslate(const scale, delta: TTexPoint; base, nb: Integer); overload;

    procedure Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single); override;
  end;

  // TIntegerList
  //
  { A list of Integers.
   Similar to TList, but using TTexPoint as items.
       The list has stack-like push/pop methods. }
  TIntegerList = class(TBaseList)
  private
     
    FList: PIntegerArray;

  protected
     
    function Get(Index: Integer): Integer;
    procedure Put(Index: Integer; const item: Integer);
    procedure SetCapacity(newCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(src: TPersistent); override;

    function Add(const item: Integer): Integer; overload;
    function AddNC(const item: Integer): Integer; overload;
    procedure Add(const i1, i2: Integer); overload;
    procedure Add(const i1, i2, i3: Integer); overload;
    procedure Add(const AList: TIntegerList); overload;
    procedure Push(const Val: Integer);
    function Pop: Integer;
    procedure Insert(Index: Integer; const item: Integer);
    procedure Remove(const item: Integer);
    function IndexOf(item: Integer): Integer;

    property Items[Index: Integer]: Integer read Get write Put; default;
    property List: PIntegerArray read FList;

        { Adds count items in an arithmetic serie.
           Items are (aBase), (aBase+aDelta) ... (aBase+(aCount-1)*aDelta) }
    procedure AddSerie(aBase, aDelta, aCount: Integer);
    { Add n integers at the address starting at (and including) first. }
    procedure AddIntegers(const First: PInteger; n: Integer); overload;
    { Add all integers from aList into the list. }
    procedure AddIntegers(const aList: TIntegerList); overload;
    { Add all integers from anArray into the list. }
    procedure AddIntegers(const anArray: array of Integer); overload;

    { Returns the minimum integer item, zero if list is empty. }
    function MinInteger: Integer;
    { Returns the maximum integer item, zero if list is empty. }
    function MaxInteger: Integer;
    { Sort items in ascending order. }
    procedure Sort;
    { Sort items in ascending order and remove duplicated integers. }
    procedure SortAndRemoveDuplicates;

    { Locate a value in a sorted list. }
    function BinarySearch(const Value: Integer): Integer; overload;
        { Locate a value in a sorted list.
           If ReturnBestFit is set to true, the routine will return the position
           of the largest value that's smaller than the sought value. Found will
           be set to True if the exact value was found, False if a "BestFit"
           was found. }
    function BinarySearch(const Value: Integer; returnBestFit: Boolean; var found: Boolean): Integer; overload;

        { Add integer to a sorted list.
           Maintains the list sorted. If you have to add "a lot" of integers
           at once, use the Add method then Sort the list for better performance. }
    function AddSorted(const Value: Integer; const ignoreDuplicates: Boolean = False): Integer;
    { Removes an integer from a sorted list. }
    procedure RemoveSorted(const Value: Integer);

    { Adds delta to all items in the list. }
    procedure Offset(delta: Integer); overload;
    procedure Offset(delta: Integer; const base, nb: Integer); overload;
  end;

  TSingleArrayList = array[0..MaxInt shr 4] of Single;
  PSingleArrayList = ^TSingleArrayList;

  // TSingleList
  //
  { A list of Single.
   Similar to TList, but using Single as items.
       The list has stack-like push/pop methods. }
  TSingleList = class(TBaseList)
  private
     
    FList: PSingleArrayList;

  protected
     
    function Get(Index: Integer): Single;
    procedure Put(Index: Integer; const item: Single);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: Single): Integer; overload;
    procedure Add(const i1, i2: Single); overload;
    procedure AddSingles(const First: PSingle; n: Integer); overload;
    procedure AddSingles(const anArray: array of Single); overload;
    procedure Push(const Val: Single);
    function Pop: Single;
    procedure Insert(Index: Integer; const item: Single);

    property Items[Index: Integer]: Single read Get write Put; default;
    property List: PSingleArrayList read FList;

    procedure AddSerie(aBase, aDelta: Single; aCount: Integer);

    { Adds delta to all items in the list. }
    procedure Offset(delta: Single); overload;

    { Adds to each item the corresponding item in the delta list.
       Performs 'Items[i]:=Items[i]+delta[i]'. 
       If both lists don't have the same item count, an exception is raised. }
    procedure Offset(const delta: TSingleList); overload;

    { Multiplies all items by factor. }
    procedure Scale(factor: Single);

    { Square all items. }
    procedure Sqr;

    { SquareRoot all items. }
    procedure Sqrt;

    { Computes the sum of all elements. }
    function Sum: Single;

    function Min: Single;
    function Max: Single;
  end;

  TDoubleArrayList = array[0..MaxInt shr 4] of Double;
  PDoubleArrayList = ^TDoubleArrayList;

    { A list of Double.
     Similar to TList, but using Double as items.
         The list has stack-like push/pop methods. }
  TDoubleList = class(TBaseList)
  private
     
    FList: PDoubleArrayList;

  protected
     
    function Get(Index: Integer): Double;
    procedure Put(Index: Integer; const item: Double);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: Double): Integer;
    procedure Push(const Val: Double);
    function Pop: Double;
    procedure Insert(Index: Integer; const item: Double);

    property Items[Index: Integer]: Double read Get write Put; default;
    property List: PDoubleArrayList read FList;

    procedure AddSerie(aBase, aDelta: Double; aCount: Integer);

    { Adds delta to all items in the list. }
    procedure Offset(delta: Double); overload;
        { Adds to each item the corresponding item in the delta list.
           Performs 'Items[i]:=Items[i]+delta[i]'. 
           If both lists don't have the same item count, an exception is raised. }
    procedure Offset(const delta: TDoubleList); overload;
    { Multiplies all items by factor. }
    procedure Scale(factor: Double);
    { Square all items. }
    procedure Sqr;
    { SquareRoot all items. }
    procedure Sqrt;

    { Computes the sum of all elements. }
    function Sum: Double;

    function Min: Single;
    function Max: Single;
  end;

  // TByteList
  //
  { A list of bytes.
   Similar to TList, but using Byte as items. }
  TByteList = class(TBaseList)
  private
     
    FList: PByteArray;

  protected
     
    function Get(Index: Integer): Byte;
    procedure Put(Index: Integer; const item: Byte);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: Byte): Integer;
    procedure Insert(Index: Integer; const item: Byte);

    property Items[Index: Integer]: Byte read Get write Put; default;
    property List: PByteArray read FList;

  end;

  // TQuaternionList
  //
  { A list of TQuaternion.
     Similar to TList, but using TQuaternion as items.
        The list has stack-like push/pop methods. }
  TQuaternionList = class(TBaseVectorList)
  private
     
    FList: PQuaternionArray;

  protected
     
    function Get(Index: Integer): TQuaternion;
    procedure Put(Index: Integer; const item: TQuaternion);
    procedure SetCapacity(NewCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function Add(const item: TQuaternion): Integer; overload;
    function Add(const item: TAffineVector; w: Single): Integer; overload;
    function Add(const X, Y, Z, W: Single): Integer; overload;
    procedure Push(const Val: TQuaternion);
    function Pop: TQuaternion;
    function IndexOf(const item: TQuaternion): Integer;
    function FindOrAdd(const item: TQuaternion): Integer;
    procedure Insert(Index: Integer; const item: TQuaternion);

    property Items[Index: Integer]: TQuaternion read Get write Put; default;
    property List: PQuaternionArray read FList;

    { Lerps corresponding quaternions from both lists using QuaternionSlerp. }
    procedure Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single); override;
        { Multiplies corresponding quaternions after the second quaternion is
           slerped with the IdentityQuaternion using factor. This allows for weighted
           combining of rotation transforms using quaternions. }
    procedure Combine(const list2: TBaseVectorList; factor: Single); override;
  end;

  // 4 byte union contain access like Integer, Single and four Byte
	T4ByteData = packed record
    case Byte of
    0 : (Bytes : record Value : array[0..3] of Byte; end);
    1 : (Int   : record Value : Integer; end);
    2 : (UInt  : record Value : Cardinal; end);
    3 : (Float : record Value : Single; end);
    4 : (Word  : record Value : array[0..1] of Word; end);
  end;

  T4ByteArrayList = array[0..MaxInt shr 4] of T4ByteData;
  P4ByteArrayList = ^T4ByteArrayList;

  // T4ByteList
  //
  { A list of T4ByteData. }

  T4ByteList = class(TBaseList)
  private
     
    FList: P4ByteArrayList;
  protected
     
    function  Get(Index: Integer): T4ByteData;
    procedure Put(Index: Integer; const item: T4ByteData);
    procedure SetCapacity(NewCapacity: Integer); override;
  public
     
    constructor Create; override;
    procedure Assign(Src: TPersistent); override;

    function  Add(const item: T4ByteData): Integer; overload;
    procedure Add(const i1: Single); overload;
    procedure Add(const i1, i2: Single); overload;
    procedure Add(const i1, i2, i3: Single); overload;
    procedure Add(const i1, i2, i3, i4: Single); overload;
    procedure Add(const i1: Integer); overload;
    procedure Add(const i1, i2: Integer); overload;
    procedure Add(const i1, i2, i3: Integer); overload;
    procedure Add(const i1, i2, i3, i4: Integer); overload;
    procedure Add(const i1: Cardinal); overload;
    procedure Add(const i1, i2: Cardinal); overload;
    procedure Add(const i1, i2, i3: Cardinal); overload;
    procedure Add(const i1, i2, i3, i4: Cardinal); overload;
    procedure Add(const AList: T4ByteList); overload;
    procedure Push(const Val: T4ByteData);
    function  Pop: T4ByteData;
    procedure Insert(Index: Integer; const item: T4ByteData);

    property Items[Index: Integer]: T4ByteData read Get write Put; default;
    property List: P4ByteArrayList read FList;
  end;

  // TLongWordList
  //
  TLongWordList = class(TBaseList)
  private
     
    FList: PLongWordArray;

  protected
     
    function Get(Index: Integer): LongWord;
    procedure Put(Index: Integer; const item: LongWord);
    procedure SetCapacity(newCapacity: Integer); override;

  public
     
    constructor Create; override;
    procedure Assign(src: TPersistent); override;

    function Add(const item: LongWord): Integer; overload;
    function AddNC(const item: LongWord): Integer; overload;
    procedure Add(const i1, i2: LongWord); overload;
    procedure Add(const i1, i2, i3: LongWord); overload;
    procedure Add(const AList: TLongWordList); overload;
    procedure Push(const Val: LongWord);
    function Pop: LongWord;
    procedure Insert(Index: Integer; const item: LongWord);
    procedure Remove(const item: LongWord);
    function IndexOf(item: Integer): LongWord;

    property Items[Index: Integer]: LongWord read Get write Put; default;
    property List: PLongWordArray read FList;

    { Add n integers at the address starting at (and including) first. }
    procedure AddLongWords(const First: PLongWord; n: Integer); overload;
    { Add all integers from aList into the list. }
    procedure AddLongWords(const aList: TLongWordList); overload;
    { Add all integers from anArray into the list. }
    procedure AddLongWords(const anArray: array of LongWord); overload;
  end;

{ Sort the refList in ascending order, ordering objList (TList) on the way. }
procedure QuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TList); overload;

{ Sort the refList in ascending order, ordering objList (TBaseList) on the way. }
procedure QuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TBaseList); overload;

{ Sort the refList in ascending order, ordering objList on the way.
   Use if, and *ONLY* if refList contains only values superior or equal to 1. }
procedure FastQuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TPersistentObjectList);

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

const
  cDefaultListGrowthDelta = 16;

// QuickSortLists (TList)
//
procedure QuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TList);
var
  I, J: Integer;
  P:    Single;
begin
  if endIndex - startIndex > 1 then
  begin
    repeat
      I := startIndex;
      J := endIndex;
      P := refList.List^[(I + J) shr 1];
      repeat
        while Single(refList.List^[I]) < P do
          Inc(I);
        while Single(refList.List^[J]) > P do
          Dec(J);
        if I <= J then
        begin
          refList.Exchange(I, J);
          objList.Exchange(I, J);
          Inc(I);
          Dec(J);
        end;
      until I > J;
      if startIndex < J then
        QuickSortLists(startIndex, J, refList, objList);
      startIndex := I;
    until I >= endIndex;
  end
  else
  if endIndex - startIndex > 0 then
  begin
    p := refList.List^[startIndex];
    if refList.List^[endIndex] < p then
    begin
      refList.Exchange(startIndex, endIndex);
      objList.Exchange(startIndex, endIndex);
    end;
  end;
end;

// QuickSortLists (TBaseList)
//
procedure QuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TBaseList);
var
  I, J: Integer;
  P:    Single;
begin
  if endIndex - startIndex > 1 then
  begin
    repeat
      I := startIndex;
      J := endIndex;
      P := refList.List^[(I + J) shr 1];
      repeat
        while Single(refList.List^[I]) < P do
          Inc(I);
        while Single(refList.List^[J]) > P do
          Dec(J);
        if I <= J then
        begin
          refList.Exchange(I, J);
          objList.Exchange(I, J);
          Inc(I);
          Dec(J);
        end;
      until I > J;
      if startIndex < J then
        QuickSortLists(startIndex, J, refList, objList);
      startIndex := I;
    until I >= endIndex;
  end
  else
  if endIndex - startIndex > 0 then
  begin
    p := refList.List^[startIndex];
    if refList.List^[endIndex] < p then
    begin
      refList.Exchange(startIndex, endIndex);
      objList.Exchange(startIndex, endIndex);
    end;
  end;
end;

// FastQuickSortLists
//
procedure FastQuickSortLists(startIndex, endIndex: Integer; refList: TSingleList; objList: TPersistentObjectList);
var
  I, J:    Integer;
  p, Temp: Integer;
  ppl:     PIntegerArray;
  oTemp    : Pointer;
  oppl     : PPointerArray;
begin
  // All singles are >=1, so IEEE format allows comparing them as if they were integers
  ppl := PIntegerArray(@refList.List[0]);
  oppl := PPointerArray(objList.List);
  if endIndex > startIndex + 1 then
  begin
    repeat
      I := startIndex;
      J := endIndex;
      p := PInteger(@refList.List[(I + J) shr 1])^;
      repeat
        while ppl^[I] < p do
          Inc(I);
        while ppl^[J] > p do
          Dec(J);
        if I <= J then
        begin
          // swap integers
          Temp := ppl^[I];
          ppl^[I] := ppl^[J];
          ppl^[J] := Temp;
          // swap pointers
          oTemp := oppl^[I];
          oppl^[I] := oppl^[J];
          oppl^[J] := oTemp;
          Inc(I);
          Dec(J);
        end;
      until I > J;
      if startIndex < J then
        FastQuickSortLists(startIndex, J, refList, objList);
      startIndex := I;
    until I >= endIndex;
  end
  else
  if endIndex > startIndex then
  begin
    if ppl^[endIndex] < ppl^[startIndex] then
    begin
      I := endIndex;
      J := startIndex;
      // swap integers
      Temp := ppl^[I];
      ppl^[I] := ppl^[J];
      ppl^[J] := Temp;
      // swap pointers
      oTemp := oppl^[I];
      oppl^[I] := oppl^[J];
      oppl^[J] := oTemp;
    end;
  end;
end;

// ------------------
// ------------------ TBaseList ------------------
// ------------------

// Create
//
constructor TBaseList.Create;
begin
  inherited Create;
  FOptions := [bloSetCountResetsMemory];
end;

// Destroy
//
destructor TBaseList.Destroy;
begin
  Clear;
  if Assigned(FBufferItem) then
    FreeMem(FBufferItem);
  inherited;
end;

 
//
procedure TBaseList.Assign(Src: TPersistent);
begin
  if (Src is TBaseList) then
  begin
    SetCapacity(TBaseList(Src).Count);
    FGrowthDelta := TBaseList(Src).FGrowthDelta;
    FCount := FCapacity;
    FTagString := TBaseList(Src).FTagString;
    Inc(FRevision);
  end
  else
    inherited;
end;

// DefineProperties
procedure TBaseList.DefineProperties(AFiler: TFiler);
begin
  inherited DefineProperties(AFiler);
  AFiler.DefineProperty('Items', ReadItemsData, WriteItemsData, True);
end;

// ReadItemsData
procedure TBaseList.ReadItemsData(AReader: TReader);
var
  lData: AnsiString;
  lOutputText: string;
begin
  lOutputText := AReader.ReadString;
  SetLength(lData, Length(lOutputText) div 2 + 1);
  HexToBin(PChar(lOutputText), PAnsiChar(lData), Length(lData));
  LoadFromString(string(lData));
end;

// WriteItemsData
procedure TBaseList.WriteItemsData(AWriter: TWriter);
var
  lData: AnsiString;
  lOutputText: String;
begin
  lData := AnsiString(SaveToString);
  SetLength(lOutputText, Length(lData) * 2);
  BinToHex(PAnsiChar(lData), PChar(lOutputText), Length(lData));
  AWriter.WriteString(lOutputText);
end;

// WriteToFiler
//
procedure TBaseList.WriteToFiler(writer: TVirtualWriter);
begin
  inherited;
  with writer do
  begin
    WriteInteger(0); // Archive Version 0
    WriteInteger(Count);
    WriteInteger(FItemSize);
    if Count > 0 then
      write(FBaseList[0], Count * FItemSize);
  end;
end;

// ReadFromFiler
//
procedure TBaseList.ReadFromFiler(reader: TVirtualReader);
var
  archiveVersion: Integer;
begin
  inherited;
  archiveVersion := reader.ReadInteger;
  if archiveVersion = 0 then
    with reader do
    begin
      FCount := ReadInteger;
      FItemSize := ReadInteger;
      SetCapacity(Count);
      if Count > 0 then
        read(FBaseList[0], Count * FItemSize);
    end
  else
    RaiseFilerException(archiveVersion);
  Inc(FRevision);
end;

// SetCount
//
procedure TBaseList.SetCount(Val: Integer);
begin
  Assert(Val >= 0);
  if Val > FCapacity then
    SetCapacity(Val);
  if (Val > FCount) and (bloSetCountResetsMemory in FOptions) then
    FillChar(FBaseList[FItemSize * FCount], (Val - FCount) * FItemSize, 0);
  FCount := Val;
  Inc(FRevision);
end;

// SetCapacity
//
procedure TBaseList.SetCapacity(newCapacity: Integer);
begin
  if newCapacity <> FCapacity then
  begin
    if bloExternalMemory in FOptions then
    begin
      Exclude(FOptions, bloExternalMemory);
      FBaseList := nil;
    end;
    ReallocMem(FBaseList, newCapacity * FItemSize);
    FCapacity := newCapacity;
    Inc(FRevision);
  end;
end;

// AddNulls
//
procedure TBaseList.AddNulls(nbVals: Cardinal);
begin
  if Integer(nbVals) + Count > Capacity then
    SetCapacity(Integer(nbVals) + Count);
  FillChar(FBaseList[FCount * FItemSize], Integer(nbVals) * FItemSize, 0);
  FCount := FCount + Integer(nbVals);
  Inc(FRevision);
end;

// InsertNulls
//
procedure TBaseList.InsertNulls(Index: Integer; nbVals: Cardinal);
var
  nc: Integer;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if nbVals > 0 then
  begin
    nc := FCount + Integer(nbVals);
    if nc > FCapacity then
      SetCapacity(nc);
    if Index < FCount then
      System.Move(FBaseList[Index * FItemSize],
        FBaseList[(Index + Integer(nbVals)) * FItemSize],
        (FCount - Index) * FItemSize);
    FillChar(FBaseList[Index * FItemSize], Integer(nbVals) * FItemSize, 0);
    FCount := nc;
    Inc(FRevision);
  end;
end;

// AdjustCapacityToAtLeast
//
procedure TBaseList.AdjustCapacityToAtLeast(const size: Integer);
begin
  if Capacity < size then
    Capacity := size;
end;

// DataSize
//

function TBaseList.DataSize: Integer;
begin
  Result := FItemSize * FCount;
end;

// BufferItem
//
function TBaseList.BufferItem: PByteArray;
begin
  if not Assigned(FBufferItem) then
    GetMem(FBufferItem, FItemSize);
  Result := FBufferItem;
end;

// GetSetCountResetsMemory
//
function TBaseList.GetSetCountResetsMemory: Boolean;
begin
  Result := (bloSetCountResetsMemory in FOptions);
end;

// SetSetCountResetsMemory
//
procedure TBaseList.SetSetCountResetsMemory(const Val: Boolean);
begin
  if Val then
    Include(FOptions, bloSetCountResetsMemory)
  else
    Exclude(FOptions, bloSetCountResetsMemory);
end;

// UseMemory
//
procedure TBaseList.UseMemory(rangeStart: Pointer; rangeCapacity: Integer);
begin
  rangeCapacity := rangeCapacity div FItemSize;
  if rangeCapacity < FCount then
    Exit;
  // transfer data
  System.Move(FBaseList^, rangeStart^, FCount * FItemSize);
  if not (bloExternalMemory in FOptions) then
  begin
    FreeMem(FBaseList);
    Include(FOptions, bloExternalMemory);
  end;
  FBaseList := rangeStart;
  FCapacity := rangeCapacity;
  SetCapacity(FCapacity); // notify subclasses
end;

// Flush
//
procedure TBaseList.Flush;
begin
  if Assigned(Self) then
  begin
    SetCount(0);
  end;
end;

// Clear
//
procedure TBaseList.Clear;
begin
  if Assigned(Self) then
  begin
    SetCount(0);
    SetCapacity(0);
  end;
end;

// Delete
//
procedure TBaseList.Delete(Index: Integer);
begin
{$IFOPT R+}
    Assert(Cardinal(index) < Cardinal(FCount));
{$ENDIF}
  Dec(FCount);
  if Index < FCount then
    System.Move(FBaseList[(Index + 1) * FItemSize],
      FBaseList[Index * FItemSize],
      (FCount - Index) * FItemSize);
  Inc(FRevision);
end;

// DeleteItems
//
procedure TBaseList.DeleteItems(Index: Integer; nbVals: Cardinal);
begin
{$IFOPT R+}
    Assert(Cardinal(index) < Cardinal(FCount));
{$ENDIF}
  if nbVals > 0 then
  begin
    if Index + Integer(nbVals) < FCount then
    begin
      System.Move(FBaseList[(Index + Integer(nbVals)) * FItemSize],
        FBaseList[Index * FItemSize],
        (FCount - Index - Integer(nbVals)) * FItemSize);
    end;
    Dec(FCount, nbVals);
    Inc(FRevision);
  end;
end;

// Exchange
//
procedure TBaseList.Exchange(index1, index2: Integer);
var
  buf: Integer;
  p:   PIntegerArray;
begin
{$IFOPT R+}
    Assert((Cardinal(index1) < Cardinal(FCount)) and (Cardinal(index2) < Cardinal(FCount)));
{$ENDIF}
  if FItemSize = 4 then
  begin
    p := PIntegerArray(FBaseList);
    buf := p^[index1];
    p^[index1] := p^[index2];
    p^[index2] := buf;
  end
  else
  begin
    System.Move(FBaseList[index1 * FItemSize], BufferItem[0], FItemSize);
    System.Move(FBaseList[index2 * FItemSize], FBaseList[index1 * FItemSize], FItemSize);
    System.Move(BufferItem[0], FBaseList[index2 * FItemSize], FItemSize);
  end;
  Inc(FRevision);
end;

// Move
//
procedure TBaseList.Move(curIndex, newIndex: Integer);
begin
  if curIndex <> newIndex then
  begin
{$IFOPT R+}
        Assert(Cardinal(newIndex) < Cardinal(Count));
        Assert(Cardinal(curIndex) < Cardinal(Count));
{$ENDIF}
    if FItemSize = 4 then
      PInteger(BufferItem)^ := PInteger(@FBaseList[curIndex * FItemSize])^
    else
      System.Move(FBaseList[curIndex * FItemSize], BufferItem[0], FItemSize);
    if curIndex < newIndex then
    begin
      // curIndex+1 necessarily exists since curIndex<newIndex and newIndex<Count
      System.Move(FBaseList[(curIndex + 1) * FItemSize], FBaseList[curIndex * FItemSize],
        (newIndex - curIndex - 1) * FItemSize);
    end
    else
    begin
      // newIndex+1 necessarily exists since newIndex<curIndex and curIndex<Count
      System.Move(FBaseList[newIndex * FItemSize], FBaseList[(newIndex + 1) * FItemSize],
        (curIndex - newIndex - 1) * FItemSize);
    end;
    if FItemSize = 4 then
      PInteger(@FBaseList[newIndex * FItemSize])^ := PInteger(BufferItem)^
    else
      System.Move(BufferItem[0], FBaseList[newIndex * FItemSize], FItemSize);
    Inc(FRevision);
  end;
end;

// Reverse
//
procedure TBaseList.Reverse;
var
  s, e: Integer;
begin
  s := 0;
  e := Count - 1;
  while s < e do
  begin
    Exchange(s, e);
    Inc(s);
    Dec(e);
  end;
  Inc(FRevision);
end;

// ------------------
// ------------------ TBaseVectorList ------------------
// ------------------

// WriteToFiler
//
procedure TBaseVectorList.WriteToFiler(writer: TVirtualWriter);
begin
  inherited;
  if Self is TTexPointList then
    exit;
  with writer do
  begin
    WriteInteger(0); // Archive Version 0
    // nothing
  end;
end;

// ReadFromFiler
//
procedure TBaseVectorList.ReadFromFiler(reader: TVirtualReader);
var
  archiveVersion: Integer;
begin
  inherited;
  if Self is TTexPointList then
    exit;
  archiveVersion := reader.ReadInteger;
  if archiveVersion = 0 then
    with reader do
    begin
      // nothing
    end
  else
    RaiseFilerException(archiveVersion);
end;

// GetExtents
//
procedure TBaseVectorList.GetExtents(out min, max: TAffineVector);
var
  I, K: Integer;
  f:    Single;
  ref:  PFloatArray;
const
  cBigValue: Single   = 1E50;
  cSmallValue: Single = -1E50;
begin
  SetVector(min, cBigValue, cBigValue, cBigValue);
  SetVector(max, cSmallValue, cSmallValue, cSmallValue);
  for I := 0 to Count - 1 do
  begin
    ref := ItemAddress[I];
    for K := 0 to 2 do
    begin
      f := ref^[K];
      if f < min.V[K] then
        min.V[K] := f;
      if f > max.V[K] then
        max.V[K] := f;
    end;
  end;
end;

// Sum
//
function TBaseVectorList.Sum: TAffineVector;
var
  I: Integer;
begin
  Result := NullVector;
  for I := 0 to Count - 1 do
    AddVector(Result, PAffineVector(ItemAddress[I])^);
end;

// Normalize
//
procedure TBaseVectorList.Normalize;
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    NormalizeVector(PAffineVector(ItemAddress[I])^);
  Inc(FRevision);
end;

// MaxSpacing
//
function TBaseVectorList.MaxSpacing(list2: TBaseVectorList): Single;
var
  I: Integer;
  s: Single;
begin
  Assert(list2.Count = Count);
  Result := 0;
  for I := 0 to Count - 1 do
  begin
    s := VectorSpacing(PAffineVector(ItemAddress[I])^,
      PAffineVector(list2.ItemAddress[I])^);
    if s > Result then
      Result := s;
  end;
end;

// Translate (delta)
//
procedure TBaseVectorList.Translate(const delta: TAffineVector);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    AddVector(PAffineVector(ItemAddress[I])^, delta);
  Inc(FRevision);
end;

// Translate (TBaseVectorList)
//
procedure TBaseVectorList.Translate(const delta: TBaseVectorList);
var
  I: Integer;
begin
  Assert(Count <= delta.Count);
  for I := 0 to Count - 1 do
    AddVector(PAffineVector(ItemAddress[I])^, PAffineVector(delta.ItemAddress[I])^);
  Inc(FRevision);
end;

// TranslateInv (TBaseVectorList)
//
procedure TBaseVectorList.TranslateInv(const delta: TBaseVectorList);
var
  I: Integer;
begin
  Assert(Count <= delta.Count);
  for I := 0 to Count - 1 do
    SubtractVector(PAffineVector(ItemAddress[I])^, PAffineVector(delta.ItemAddress[I])^);
  Inc(FRevision);
end;

// AngleLerp
//
procedure TBaseVectorList.AngleLerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
var
  I: Integer;
begin
  Assert(list1.Count = list2.Count);
  if list1 <> list2 then
  begin
    if lerpFactor = 0 then
      Assign(list1)
    else
    if lerpFactor = 1 then
      Assign(list2)
    else
    begin
      Capacity := list1.Count;
      FCount := list1.Count;
      for I := 0 to list1.Count - 1 do
        PAffineVector(ItemAddress[I])^ := VectorAngleLerp(PAffineVector(list1.ItemAddress[I])^,
          PAffineVector(list2.ItemAddress[I])^,
          lerpFactor);
    end;
  end
  else
    Assign(list1);
  Inc(FRevision);
end;

// AngleCombine
//
procedure TBaseVectorList.AngleCombine(const list1: TBaseVectorList; intensity: Single);
var
  I: Integer;
begin
  Assert(list1.Count = Count);
  for I := 0 to Count - 1 do
    PAffineVector(ItemAddress[I])^ := VectorAngleCombine(PAffineVector(ItemAddress[I])^,
      PAffineVector(list1.ItemAddress[I])^,
      intensity);
  Inc(FRevision);
end;

// Combine
//
procedure TBaseVectorList.Combine(const list2: TBaseVectorList; factor: Single);
var
  I: Integer;
begin
  Assert(list2.Count >= Count);
  for I := 0 to Count - 1 do
    CombineVector(PAffineVector(ItemAddress[I])^,
      PAffineVector(list2.ItemAddress[I])^,
      factor);
  Inc(FRevision);
end;

// GetItemAddress
//
function TBaseVectorList.GetItemAddress(Index: Integer): PFloatArray;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := PFloatArray(@FBaseList[Index * FItemSize]);
end;

// ------------------
// ------------------ TAffineVectorList ------------------
// ------------------

// Create
//
constructor TAffineVectorList.Create;
begin
  FItemSize := SizeOf(TAffineVector);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//
procedure TAffineVectorList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TAffineVectorList) then
      System.Move(TAffineVectorList(Src).FList^, FList^, FCount * SizeOf(TAffineVector));
  end
  else
    Clear;
end;

// Add (affine)
//
function TAffineVectorList.Add(const item: TAffineVector): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
  Inc(FRevision);
end;

// Add (hmg)
//
function TAffineVectorList.Add(const item: TVector): Integer;
begin
  Result := Add(PAffineVector(@item)^);
end;

// Add (2 affine)
//
procedure TAffineVectorList.Add(const i1, i2: TAffineVector);
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[FCount - 2] := i1;
  FList^[FCount - 1] := i2;
  Inc(FRevision);
end;

// Add (3 affine)
//
procedure TAffineVectorList.Add(const i1, i2, i3: TAffineVector);
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[FCount - 3] := i1;
  FList^[FCount - 2] := i2;
  FList^[FCount - 1] := i3;
  Inc(FRevision);
end;

// Add (vector2f)
//
function TAffineVectorList.Add(const item: TVector2f): Integer;
begin
  Result := Add(AffineVectorMake(item.V[0], item.V[1], 0));
end;

// Add (texpoint)
//
function TAffineVectorList.Add(const item: TTexPoint): Integer;
begin
  Result := Add(AffineVectorMake(item.S, item.T, 0));
end;

// Add
//
function TAffineVectorList.Add(const X, Y: Single): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  Inc(FCount);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  v := @List[Result];
  v^.V[0] := X;
  v^.V[1] := Y;
  v^.V[2] := 0;
  Inc(FRevision);
end;

// Add
//
function TAffineVectorList.Add(const X, Y, Z: Single): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  Inc(FCount);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  v := @List[Result];
  v^.V[0] := X;
  v^.V[1] := Y;
  v^.V[2] := Z;
  Inc(FRevision);
end;

// Add (3 ints)
//
function TAffineVectorList.Add(const X, Y, Z: Integer): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  v := @List[Result];
  v^.V[0] := X;
  v^.V[1] := Y;
  v^.V[2] := Z;
  Inc(FCount);
  Inc(FRevision);
end;

// Add (3 ints, no capacity check)
//
function TAffineVectorList.AddNC(const X, Y, Z: Integer): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  v := @List[Result];
  v^.V[0] := X;
  v^.V[1] := Y;
  v^.V[2] := Z;
  Inc(FCount);
  Inc(FRevision);
end;

// Add (2 ints in array + 1)
//
function TAffineVectorList.Add(const xy: PIntegerArray; const Z: Integer): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  v := @List[Result];
  v^.V[0] := xy^[0];
  v^.V[1] := xy^[1];
  v^.V[2] := Z;
  Inc(FCount);
  Inc(FRevision);
end;

// AddNC (2 ints in array + 1, no capacity check)
//
function TAffineVectorList.AddNC(const xy: PIntegerArray; const Z: Integer): Integer;
var
  v: PAffineVector;
begin
  Result := FCount;
  v := @List[Result];
  v^.V[0] := xy^[0];
  v^.V[1] := xy^[1];
  v^.V[2] := Z;
  Inc(FCount);
  Inc(FRevision);
end;

// Add
//
procedure TAffineVectorList.Add(const list: TAffineVectorList);
begin
  if Assigned(list) and (list.Count > 0) then
  begin
    if Count + list.Count > Capacity then
      Capacity := Count + list.Count;
    System.Move(list.FList[0], FList[Count], list.Count * SizeOf(TAffineVector));
    Inc(FCount, list.Count);
  end;
  Inc(FRevision);
end;

// Get
//
function TAffineVectorList.Get(Index: Integer): TAffineVector;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//
procedure TAffineVectorList.Insert(Index: Integer; const Item: TAffineVector);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(TAffineVector));
  FList^[Index] := Item;
  Inc(FCount);
  Inc(FRevision);
end;

// IndexOf
//
function TAffineVectorList.IndexOf(const item: TAffineVector): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to Count - 1 do
    if VectorEquals(item, FList^[I]) then
    begin
      Result := I;
      Break;
    end;
end;

// FindOrAdd
//
function TAffineVectorList.FindOrAdd(const item: TAffineVector): Integer;
begin
  Result := IndexOf(item);
  if Result < 0 then
  begin
    Result := Add(item);
    Inc(FRevision);
  end;
end;

// Put
//
procedure TAffineVectorList.Put(Index: Integer; const Item: TAffineVector);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
  Inc(FRevision);
end;

// SetCapacity
//
procedure TAffineVectorList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PAffineVectorArray(FBaseList);
end;

// Push
//
procedure TAffineVectorList.Push(const Val: TAffineVector);
begin
  Add(Val);
end;

// Pop
//
function TAffineVectorList.Pop: TAffineVector;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
    Inc(FRevision);
  end
  else
    Result := NullVector;
end;

// Translate (delta)
//
procedure TAffineVectorList.Translate(const delta: TAffineVector);
begin
  VectorArrayAdd(FList, delta, Count, FList);
  Inc(FRevision);
end;

// Translate (delta, range)
//
procedure TAffineVectorList.Translate(const delta: TAffineVector; base, nb: Integer);
begin
  VectorArrayAdd(@FList[base], delta, nb, @FList[base]);
  Inc(FRevision);
end;

// TranslateItem
//

procedure TAffineVectorList.TranslateItem(Index: Integer; const delta: TAffineVector);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  AddVector(FList^[Index], delta);
  Inc(FRevision);
end;

// TranslateItems
//
procedure TAffineVectorList.TranslateItems(Index: Integer; const delta: TAffineVector; nb: Integer);
begin
  nb := Index + nb;
{$IFOPT R+}
    Assert(Cardinal(index) < Cardinal(FCount));
    if nb > FCount then
        nb := FCount;
{$ENDIF}
  VectorArrayAdd(@FList[Index], delta, nb - Index, @FList[Index]);
  Inc(FRevision);
end;

// CombineItem
//
procedure TAffineVectorList.CombineItem(Index: Integer; const vector: TAffineVector; const f: Single);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  CombineVector(FList^[Index], vector, @f);
  Inc(FRevision);
end;

// TransformAsPoints
//
procedure TAffineVectorList.TransformAsPoints(const matrix: TMatrix);
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    FList^[I] := VectorTransform(FList^[I], matrix);
  Inc(FRevision);
end;

// TransformAsVectors (hmg)
//
procedure TAffineVectorList.TransformAsVectors(const matrix: TMatrix);
var
  m: TAffineMatrix;
begin
  if FCount > 0 then
  begin
    SetMatrix(m, matrix);
    TransformAsVectors(m);
  end;
end;

// TransformAsVectors (affine)
//

procedure TAffineVectorList.TransformAsVectors(const matrix: TAffineMatrix);
var
  I: Integer;
begin
  for I := 0 to FCount - 1 do
    FList^[I] := VectorTransform(FList^[I], matrix);
  Inc(FRevision);
end;

// Normalize
//
procedure TAffineVectorList.Normalize;
begin
  NormalizeVectorArray(List, Count);
  Inc(FRevision);
end;

// Lerp
//
procedure TAffineVectorList.Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
begin
  if (list1 is TAffineVectorList) and (list2 is TAffineVectorList) then
  begin
    Assert(list1.Count = list2.Count);
    Capacity := list1.Count;
    FCount := list1.Count;
    VectorArrayLerp(TAffineVectorList(list1).List, TAffineVectorList(list2).List,
      lerpFactor, FCount, List);
    Inc(FRevision);
  end;
end;

// Scale (scalar)
//
procedure TAffineVectorList.Scale(factor: Single);
begin
  if (Count > 0) and (factor <> 1) then
  begin
    ScaleFloatArray(@FList[0].V[0], Count * 3, factor);
    Inc(FRevision);
  end;
end;

// Scale (affine)
//
procedure TAffineVectorList.Scale(const factors: TAffineVector);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    ScaleVector(FList^[I], factors);
  Inc(FRevision);
end;

// ------------------
// ------------------ TVectorList ------------------
// ------------------

// Create
//

constructor TVectorList.Create;
begin
  FItemSize := SizeOf(TVector);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TVectorList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TVectorList) then
      System.Move(TVectorList(Src).FList^, FList^, FCount * SizeOf(TVector));
  end
  else
    Clear;
end;

// Add
//

function TVectorList.Add(const item: TVector): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// Add
//

function TVectorList.Add(const item: TAffineVector; w: Single): Integer;
begin
  Result := Add(VectorMake(item, w));
end;

// Add
//

function TVectorList.Add(const X, Y, Z, w: Single): Integer;
begin
  Result := Add(VectorMake(X, Y, Z, w));
end;

// Add (3 affine)
//

procedure TVectorList.Add(const i1, i2, i3: TAffineVector; w: Single);
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  PAffineVector(@FList[FCount - 3])^ := i1;
  FList^[FCount - 3].V[3] := w;
  PAffineVector(@FList[FCount - 2])^ := i2;
  FList^[FCount - 2].V[3] := w;
  PAffineVector(@FList[FCount - 1])^ := i3;
  FList^[FCount - 1].V[3] := w;
end;

// AddVector
//

function TVectorList.AddVector(const item: TAffineVector): Integer;
begin
  Result := Add(VectorMake(item));
end;

// AddPoint
//

function TVectorList.AddPoint(const item: TAffineVector): Integer;
begin
  Result := Add(PointMake(item));
end;

// AddPoint
//

function TVectorList.AddPoint(const X, Y: Single; const Z: Single = 0): Integer;
begin
  Result := Add(PointMake(X, Y, Z));
end;

// Get
//

function TVectorList.Get(Index: Integer): TVector;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TVectorList.Insert(Index: Integer; const Item: TVector);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(TVector));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TVectorList.Put(Index: Integer; const Item: TVector);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TVectorList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PVectorArray(FBaseList);
end;

// Push
//

procedure TVectorList.Push(const Val: TVector);
begin
  Add(Val);
end;

// Pop
//

function TVectorList.Pop: TVector;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := NullHmgVector;
end;

// IndexOf
//

function TVectorList.IndexOf(const item: TVector): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to Count - 1 do
    if VectorEquals(item, FList^[I]) then
    begin
      Result := I;
      Break;
    end;
end;

// FindOrAdd
//

function TVectorList.FindOrAdd(const item: TVector): Integer;
begin
  Result := IndexOf(item);
  if Result < 0 then
    Result := Add(item);
end;

// FindOrAddPoint
//

function TVectorList.FindOrAddPoint(const item: TAffineVector): Integer;
var
  ptItem: TVector;
begin
  MakePoint(ptItem, item);
  Result := IndexOf(ptItem);
  if Result < 0 then
    Result := Add(ptItem);
end;

// Lerp
//

procedure TVectorList.Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
begin
  if (list1 is TVectorList) and (list2 is TVectorList) then
  begin
    Assert(list1.Count = list2.Count);
    Capacity := list1.Count;
    FCount := list1.Count;
    VectorArrayLerp(TVectorList(list1).List, TVectorList(list2).List,
      lerpFactor, FCount, List);
  end;
end;

// ------------------
// ------------------ TTexPointList ------------------
// ------------------

// Create
//

constructor TTexPointList.Create;
begin
  FItemSize := SizeOf(TTexPoint);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TTexPointList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TTexPointList) then
      System.Move(TTexPointList(Src).FList^, FList^, FCount * SizeOf(TTexPoint));
  end
  else
    Clear;
end;

// IndexOf
//

function TTexPointList.IndexOf(const item: TTexpoint): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to Count - 1 do
    if TexpointEquals(FList^[I], item) then
    begin
      Result := I;
      Break;
    end;
end;

// FindOrAdd
//

function TTexPointList.FindOrAdd(const item: TTexPoint): Integer;
begin
  Result := IndexOf(item);
  if Result < 0 then
    Result := Add(item);
end;

// Add
//

function TTexPointList.Add(const item: TTexPoint): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// Add
//

function TTexPointList.Add(const item: TVector2f): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := PTexPoint(@Item)^;
  Inc(FCount);
end;

// Add
//

function TTexPointList.Add(const texS, Text: Single): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  with FList^[Result] do
  begin
    s := texS;
    t := Text;
  end;
  Inc(FCount);
end;

// Add
//

function TTexPointList.Add(const texS, Text: Integer): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  with FList^[Result] do
  begin
    s := texS;
    t := Text;
  end;
  Inc(FCount);
end;

// AddNC
//

function TTexPointList.AddNC(const texS, Text: Integer): Integer;
begin
  Result := FCount;
  with FList^[Result] do
  begin
    s := texS;
    t := Text;
  end;
  Inc(FCount);
end;

// Add
//

function TTexPointList.Add(const texST: PIntegerArray): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  with FList^[Result] do
  begin
    s := texST^[0];
    t := texST^[1];
  end;
  Inc(FCount);
end;

// AddNC
//

function TTexPointList.AddNC(const texST: PIntegerArray): Integer;
begin
  Result := FCount;
  with FList^[Result] do
  begin
    s := texST^[0];
    t := texST^[1];
  end;
  Inc(FCount);
end;

// Get
//

function TTexPointList.Get(Index: Integer): TTexPoint;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TTexPointList.Insert(Index: Integer; const Item: TTexPoint);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(TTexPoint));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TTexPointList.Put(Index: Integer; const Item: TTexPoint);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TTexPointList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PTexPointArray(FBaseList);
end;

// Push
//

procedure TTexPointList.Push(const Val: TTexPoint);
begin
  Add(Val);
end;

// Pop
//

function TTexPointList.Pop: TTexPoint;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := NullTexPoint;
end;

// Translate
//

procedure TTexPointList.Translate(const delta: TTexPoint);
begin
  TexPointArrayAdd(List, delta, FCount, FList);
end;

// ScaleAndTranslate
//

procedure TTexPointList.ScaleAndTranslate(const scale, delta: TTexPoint);
begin
  TexPointArrayScaleAndAdd(FList, delta, FCount, scale, FList);
end;

// ScaleAndTranslate
//

procedure TTexPointList.ScaleAndTranslate(const scale, delta: TTexPoint; base, nb: Integer);
var
  p: PTexPointArray;
begin
  p := @FList[base];
  TexPointArrayScaleAndAdd(p, delta, nb, scale, p);
end;

// Lerp
//

procedure TTexPointList.Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
begin
  if (list1 is TTexPointList) and (list2 is TTexPointList) then
  begin
    Assert(list1.Count = list2.Count);
    Capacity := list1.Count;
    FCount := list1.Count;
    VectorArrayLerp(TTexPointList(list1).List, TTexPointList(list2).List,
      lerpFactor, FCount, List);
  end;
end;

// ------------------
// ------------------ TIntegerList ------------------
// ------------------

// Create
//

constructor TIntegerList.Create;
begin
  FItemSize := SizeOf(Integer);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TIntegerList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TIntegerList) then
      System.Move(TIntegerList(Src).FList^, FList^, FCount * SizeOf(Integer));
  end
  else
    Clear;
end;

// Add (simple)
//

function TIntegerList.Add(const item: Integer): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// AddNC (simple, no capacity check)
//

function TIntegerList.AddNC(const item: Integer): Integer;
begin
  Result := FCount;
  FList^[Result] := Item;
  Inc(FCount);
end;

// Add (two at once)
//

procedure TIntegerList.Add(const i1, i2: Integer);
var
  tmpList : PIntegerArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
end;

// Add (three at once)
//

procedure TIntegerList.Add(const i1, i2, i3: Integer);
var
  tmpList : PIntegerArray;
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 3];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
end;

// Add (list)
//

procedure TIntegerList.Add(const AList: TIntegerList);
begin
  if Assigned(AList) and (AList.Count > 0) then
  begin
    if Count + AList.Count > Capacity then
      Capacity := Count + AList.Count;
    System.Move(AList.FList[0], FList[Count], AList.Count * SizeOf(Integer));
    Inc(FCount, AList.Count);
  end;
end;

// Get
//

function TIntegerList.Get(Index: Integer): Integer;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TIntegerList.Insert(Index: Integer; const Item: Integer);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1], (FCount - Index) * SizeOf(Integer));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Remove
//

procedure TIntegerList.Remove(const item: Integer);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
  begin
    if FList^[I] = item then
    begin
      System.Move(FList[I + 1], FList[I], (FCount - 1 - I) * SizeOf(Integer));
      Dec(FCount);
      Break;
    end;
  end;
end;

// Put
//

procedure TIntegerList.Put(Index: Integer; const Item: Integer);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TIntegerList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PIntegerArray(FBaseList);
end;

// Push
//

procedure TIntegerList.Push(const Val: Integer);
begin
  Add(Val);
end;

// Pop
//

function TIntegerList.Pop: Integer;
begin
  if FCount > 0 then
  begin
    Result := FList^[FCount - 1];
    Delete(FCount - 1);
  end
  else
    Result := 0;
end;

// AddSerie
//

procedure TIntegerList.AddSerie(aBase, aDelta, aCount: Integer);
var
  tmpList : PInteger;
  I:    Integer;
begin
  if aCount <= 0 then
    Exit;
  AdjustCapacityToAtLeast(Count + aCount);
  tmpList := @FList[Count];
  for I := Count to Count + aCount - 1 do
  begin
    tmpList^ := aBase;
    Inc(tmpList);
    aBase := aBase + aDelta;
  end;
  FCount := Count + aCount;
end;

// AddIntegers (pointer & n)
//

procedure TIntegerList.AddIntegers(const First: PInteger; n: Integer);
begin
  if n < 1 then
    Exit;
  AdjustCapacityToAtLeast(Count + n);
  System.Move(First^, FList[FCount], n * SizeOf(Integer));
  FCount := FCount + n;
end;

// AddIntegers (TIntegerList)
//

procedure TIntegerList.AddIntegers(const aList: TIntegerList);
begin
  if not Assigned(aList) then
    Exit;
  AddIntegers(@aList.List[0], aList.Count);
end;

// AddIntegers (array)
//

procedure TIntegerList.AddIntegers(const anArray: array of Integer);
var
  n: Integer;
begin
  n := Length(anArray);
  if n > 0 then
    AddIntegers(@anArray[0], n);
end;

// IntegerSearch
//

function IntegerSearch(item: Integer; list: PIntegerVector; Count: Integer): Integer; register;
{$IFDEF GLS_NO_ASM}
var i : integer;
begin
  result:=-1;
  for i := 0 to Count-1 do begin
    if list^[i]=item then begin
      result:=i;
      break;
    end;
  end;
end;
{$ELSE}
asm
  push edi;

  test ecx, ecx
  jz @@NotFound

  mov edi, edx;
  mov edx, ecx;
  repne scasd;
  je @@FoundIt

  @@NotFound:
  xor eax, eax
  dec eax
  jmp @@end;

  @@FoundIt:
  sub edx, ecx;
  dec edx;
  mov eax, edx;

  @@end:
  pop edi;
end;
{$ENDIF}

// IndexOf
//

function TIntegerList.IndexOf(item: Integer): Integer; register;
begin
  Result := IntegerSearch(item, FList, FCount);
end;

// MinInteger
//

function TIntegerList.MinInteger: Integer;
var
  I: Integer;
  locList: PIntegerVector;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] < Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// MaxInteger
//

function TIntegerList.MaxInteger: Integer;
var
  I: Integer;
  locList: PIntegerVector;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] > Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// IntegerQuickSort
//

procedure IntegerQuickSort(sortList: PIntegerArray; left, right: Integer);
var
  I, J: Integer;
  p, t: Integer;
begin
  repeat
    I := left;
    J := right;
    p := sortList^[(left + right) shr 1];
    repeat
      while sortList^[I] < p do
        Inc(I);
      while sortList^[J] > p do
        Dec(J);
      if I <= J then
      begin
        t := sortList^[I];
        sortList^[I] := sortList^[J];
        sortList^[J] := t;
        Inc(I);
        Dec(J);
      end;
    until I > J;
    if left < J then
      IntegerQuickSort(sortList, left, J);
    left := I;
  until I >= right;
end;

// Sort
//

procedure TIntegerList.Sort;
begin
  if (FList <> nil) and (Count > 1) then
    IntegerQuickSort(FList, 0, Count - 1);
end;

// SortAndRemoveDuplicates
//

procedure TIntegerList.SortAndRemoveDuplicates;
var
  I, J, lastVal: Integer;
  localList:     PIntegerArray;
begin
  if (FList <> nil) and (Count > 1) then
  begin
    IntegerQuickSort(FList, 0, Count - 1);
    J := 0;
    localList := FList;
    lastVal := localList^[J];
    for I := 1 to Count - 1 do
    begin
      if localList^[I] <> lastVal then
      begin
        lastVal := localList^[I];
        Inc(J);
        localList^[J] := lastVal;
      end;
    end;
    FCount := J + 1;
  end;
end;

// BinarySearch
//

function TIntegerList.BinarySearch(const Value: Integer): Integer;
var
  found: Boolean;
begin
  Result := BinarySearch(Value, False, found);
end;

// BinarySearch
//

function TIntegerList.BinarySearch(const Value: Integer; returnBestFit: Boolean; var found: Boolean): Integer;
var
  Index:   Integer;
  min, max, mid: Integer;
  intList: PIntegerArray;
begin
  // Assume we won't find it
  found := False;
  // If the list is empty, we won't find the sought value!
  if Count = 0 then
  begin
    Result := -1;
    Exit;
  end;

  min := -1; // ONE OFF!
  max := Count; // ONE OFF!

  // We now know that Min and Max AREN'T the values!
  Index := -1;
  intList := List;
  repeat
    // Find the middle of the current scope
    mid := (min + max) shr 1;
    // Reduce the search scope by half
    if intList^[mid] <= Value then
    begin
      // Is this the one?
      if intList^[mid] = Value then
      begin
        Index := mid;
        found := True;
        Break;
      end
      else
        min := mid;
    end
    else
      max := mid;
  until min + 1 = max;

  if returnBestFit then
  begin
    if Index >= 0 then
      Result := Index
    else
      Result := min;
  end
  else
    Result := Index;
end;

// AddSorted
//

function TIntegerList.AddSorted(const Value: Integer; const ignoreDuplicates: Boolean = False): Integer;
var
  Index: Integer;
  found: Boolean;
begin
  Index := BinarySearch(Value, True, found);
  if ignoreDuplicates and Found then
    Result := -1
  else
  begin
    Insert(Index + 1, Value);
    Result := Index + 1;
  end;
end;

// RemoveSorted
//

procedure TIntegerList.RemoveSorted(const Value: Integer);
var
  Index: Integer;
begin
  Index := BinarySearch(Value);
  if Index >= 0 then
    Delete(Index);
end;

// Offset (all)
//

procedure TIntegerList.Offset(delta: Integer);
var
  I: Integer;
  locList: PIntegerArray;
begin
  locList := FList;
  for I := 0 to FCount - 1 do
    locList^[I] := locList^[I] + delta;
end;

// Offset (range)
//

procedure TIntegerList.Offset(delta: Integer; const base, nb: Integer);
var
  I: Integer;
  locList: PIntegerArray;
begin
  locList := FList;
  for I := base to base + nb - 1 do
    locList^[I] := locList^[I] + delta;
end;

// ------------------
// ------------------ TSingleList ------------------
// ------------------

// Create
//

constructor TSingleList.Create;
begin
  FItemSize := SizeOf(Single);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TSingleList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TSingleList) then
      System.Move(TSingleList(Src).FList^, FList^, FCount * SizeOf(Single));
  end
  else
    Clear;
end;

// Add
//

function TSingleList.Add(const item: Single): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

procedure TSingleList.Add(const i1, i2: Single);
var
  tmpList : PSingleArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
end;

procedure TSingleList.AddSingles(const First: PSingle; n: Integer);
begin
  if n < 1 then
    Exit;
  AdjustCapacityToAtLeast(Count + n);
  System.Move(First^, FList[FCount], n * SizeOf(Single));
  FCount := FCount + n;
end;

procedure TSingleList.AddSingles(const anArray: array of Single);
var
  n: Integer;
begin
  n := Length(anArray);
  if n > 0 then
    AddSingles(@anArray[0], n);
end;

// Get
//

function TSingleList.Get(Index: Integer): Single;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TSingleList.Insert(Index: Integer; const Item: Single);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(Single));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TSingleList.Put(Index: Integer; const Item: Single);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TSingleList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PSingleArrayList(FBaseList);
end;

// Push
//

procedure TSingleList.Push(const Val: Single);
begin
  Add(Val);
end;

// Pop
//

function TSingleList.Pop: Single;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := 0;
end;

// AddSerie
//

procedure TSingleList.AddSerie(aBase, aDelta: Single; aCount: Integer);
var
  tmpList : PSingle;
  I:    Integer;
begin
  if aCount <= 0 then
    Exit;
  AdjustCapacityToAtLeast(Count + aCount);
  tmpList := @FList[Count];
  for I := Count to Count + aCount - 1 do
  begin
    tmpList^ := aBase;
    Inc(tmpList);
    aBase := aBase + aDelta;
  end;
  FCount := Count + aCount;
end;

// Offset (single)
//

procedure TSingleList.Offset(delta: Single);
begin
  OffsetFloatArray(PFloatVector(FList), FCount, delta);
end;

// Offset (list)
//

procedure TSingleList.Offset(const delta: TSingleList);
begin
  if FCount = delta.FCount then
    OffsetFloatArray(PFloatVector(FList), PFloatVector(delta.FList), FCount)
  else
    raise Exception.Create('SingleList count do not match');
end;

// Scale
//

procedure TSingleList.Scale(factor: Single);
begin
  ScaleFloatArray(PFloatVector(FList), FCount, factor);
end;

// Sqr
//

procedure TSingleList.Sqr;
var
  I: Integer;
  locList: PSingleArrayList;
begin
  locList := FList;
  for I := 0 to Count - 1 do
    locList^[I] := locList^[I] * locList^[I];
end;

// Sqrt
//

procedure TSingleList.Sqrt;
var
  I: Integer;
  locList: PSingleArrayList;
begin
  locList := FList;
  for I := 0 to Count - 1 do
    locList^[I] := System.Sqrt(locList^[I]);
end;

// Sum
//

function TSingleList.Sum: Single;
{$IFNDEF GLS_NO_ASM}
  function ComputeSum(list: PSingleArrayList; nb: Integer): Single; register;
  asm
    fld   dword ptr [eax]
    @@Loop:
    dec   edx
    fadd  dword ptr [eax+edx*4]
    jnz   @@Loop
  end;

begin
  if FCount > 0 then
    Result := ComputeSum(FList, FCount)
  else
    Result := 0;
{$ELSE}
var
  i: Integer;
begin
  Result := 0;
  for i := 0 to FCount-1 do
    Result := Result + FList^[i];
{$ENDIF}
end;

// Min
//
function TSingleList.Min: Single;
var
  I: Integer;
  locList: PSingleArrayList;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] < Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// Max
//
function TSingleList.Max: Single;
var
  I: Integer;
  locList: PSingleArrayList;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] > Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// ------------------
// ------------------ TByteList ------------------
// ------------------

// Create
//

constructor TByteList.Create;
begin
  FItemSize := SizeOf(Byte);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TByteList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TByteList) then
      System.Move(TByteList(Src).FList^, FList^, FCount * SizeOf(Byte));
  end
  else
    Clear;
end;

// Add
//

function TByteList.Add(const item: Byte): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// Get
//

function TByteList.Get(Index: Integer): Byte;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TByteList.Insert(Index: Integer; const Item: Byte);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(Byte));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TByteList.Put(Index: Integer; const Item: Byte);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TByteList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PByteArray(FBaseList);
end;

// ------------------
// ------------------ TDoubleList ------------------
// ------------------

// Create
//

constructor TDoubleList.Create;
begin
  FItemSize := SizeOf(Double);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TDoubleList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TDoubleList) then
      System.Move(TDoubleList(Src).FList^, FList^, FCount * SizeOf(Double));
  end
  else
    Clear;
end;

// Add
//

function TDoubleList.Add(const item: Double): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// Get
//

function TDoubleList.Get(Index: Integer): Double;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TDoubleList.Insert(Index: Integer; const Item: Double);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(Double));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TDoubleList.Put(Index: Integer; const Item: Double);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TDoubleList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PDoubleArrayList(FBaseList);
end;

// Push
//

procedure TDoubleList.Push(const Val: Double);
begin
  Add(Val);
end;

// Pop
//

function TDoubleList.Pop: Double;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := 0;
end;

// AddSerie
//

procedure TDoubleList.AddSerie(aBase, aDelta: Double; aCount: Integer);
var
  tmpList: PDouble;
  I:    Integer;
begin
  if aCount <= 0 then
    Exit;
  AdjustCapacityToAtLeast(Count + aCount);
  tmpList := @FList[Count];
  for I := Count to Count + aCount - 1 do
  begin
    tmpList^ := aBase;
    Inc(tmpList);
    aBase := aBase + aDelta;
  end;
  FCount := Count + aCount;
end;

// Offset (Double)
//

procedure TDoubleList.Offset(delta: Double);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    FList^[I] := FList^[I] + delta;
end;

// Offset (list)
//

procedure TDoubleList.Offset(const delta: TDoubleList);
var
  I: Integer;
begin
  if FCount = delta.FCount then
    for I := 0 to Count - 1 do
      FList^[I] := FList^[I] + delta[I]
  else
    raise Exception.Create('DoubleList count do not match');
end;

// Scale
//

procedure TDoubleList.Scale(factor: Double);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
    FList^[I] := FList^[I] * factor;
end;

// Sqr
//

procedure TDoubleList.Sqr;
var
  I: Integer;
  locList: PDoubleArrayList;
begin
  locList := FList;
  for I := 0 to Count - 1 do
    locList^[I] := locList^[I] * locList^[I];
end;

// Sqrt
//

procedure TDoubleList.Sqrt;
var
  I: Integer;
  locList: PDoubleArrayList;
begin
  locList := FList;
  for I := 0 to Count - 1 do
    locList^[I] := System.Sqrt(locList^[I]);
end;

// Sum
//

function TDoubleList.Sum: Double;
{$IFNDEF GLS_NO_ASM}
  function ComputeSum(list: PDoubleArrayList; nb: Integer): Double; register;
  asm
    fld   dword ptr [eax]
    @@Loop:
    dec   edx
    fadd  dword ptr [eax+edx*4]
    jnz   @@Loop
  end;

begin
  if FCount > 0 then
    Result := ComputeSum(FList, FCount)
  else
    Result := 0;
{$ELSE}
var
  i: Integer;
begin
    Result := 0;
    for i := 0 to FCount-1 do
    Result := Result + FList^[i];
{$ENDIF}
end;

// Min
//
function TDoubleList.Min: Single;
var
  I: Integer;
  locList: PDoubleArrayList;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] < Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// Max
//
function TDoubleList.Max: Single;
var
  I: Integer;
  locList: PDoubleArrayList;
begin
  if FCount > 0 then
  begin
    locList := FList;
    Result := locList^[0];
    for I := 1 to FCount - 1 do
      if locList^[I] > Result then
        Result := locList^[I];
  end
  else
    Result := 0;
end;

// ------------------
// ------------------ TQuaternionList ------------------
// ------------------

// Create
//

constructor TQuaternionList.Create;
begin
  FItemSize := SizeOf(TQuaternion);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TQuaternionList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TQuaternionList) then
      System.Move(TQuaternionList(Src).FList^, FList^, FCount * SizeOf(TQuaternion));
  end
  else
    Clear;
end;

// Add
//

function TQuaternionList.Add(const item: TQuaternion): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// Add
//

function TQuaternionList.Add(const item: TAffineVector; w: Single): Integer;
begin
  Result := Add(QuaternionMake(item.V, w));
end;

// Add
//

function TQuaternionList.Add(const X, Y, Z, w: Single): Integer;
begin
  Result := Add(QuaternionMake([X, Y, Z], w));
end;

// Get
//

function TQuaternionList.Get(Index: Integer): TQuaternion;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TQuaternionList.Insert(Index: Integer; const Item: TQuaternion);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(TQuaternion));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Put
//

procedure TQuaternionList.Put(Index: Integer; const Item: TQuaternion);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TQuaternionList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PQuaternionArray(FBaseList);
end;

// Push
//

procedure TQuaternionList.Push(const Val: TQuaternion);
begin
  Add(Val);
end;

// Pop
//

function TQuaternionList.Pop: TQuaternion;
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := IdentityQuaternion;
end;

// IndexOf
//

function TQuaternionList.IndexOf(const item: TQuaternion): Integer;
var
  I: Integer;
  curItem: PQuaternion;
begin
  for I := 0 to Count - 1 do
  begin
    curItem := @FList[I];
    if (item.RealPart = curItem^.RealPart) and VectorEquals(item.ImagPart, curItem^.ImagPart) then
    begin
      Result := I;
      Exit;
    end;
  end;
  Result := -1;
end;

// FindOrAdd
//

function TQuaternionList.FindOrAdd(const item: TQuaternion): Integer;
begin
  Result := IndexOf(item);
  if Result < 0 then
    Result := Add(item);
end;

// Lerp
//

procedure TQuaternionList.Lerp(const list1, list2: TBaseVectorList; lerpFactor: Single);
var
  I: Integer;
begin
  if (list1 is TQuaternionList) and (list2 is TQuaternionList) then
  begin
    Assert(list1.Count = list2.Count);
    Capacity := list1.Count;
    FCount := list1.Count;
    for I := 0 to FCount - 1 do
      Put(I, QuaternionSlerp(TQuaternionList(list1)[I], TQuaternionList(list2)[I], lerpFactor));
  end;
end;

// Combine
//

procedure TQuaternionList.Combine(const list2: TBaseVectorList; factor: Single);

  procedure CombineQuaternion(var q1: TQuaternion; const q2: TQuaternion; factor: Single);
  begin
    q1 := QuaternionMultiply(q1, QuaternionSlerp(IdentityQuaternion, q2, factor));
  end;

var
  I: Integer;
begin
  Assert(list2.Count >= Count);
  if list2 is TQuaternionList then
  begin
    for I := 0 to Count - 1 do
    begin
      CombineQuaternion(PQuaternion(ItemAddress[I])^,
        PQuaternion(list2.ItemAddress[I])^,
        factor);
    end;
  end
  else
    inherited;
end;

// ------------------
// ------------------ T4ByteList ------------------
// ------------------

// Create
//

constructor T4ByteList.Create;
begin
  FItemSize := SizeOf(T4ByteList);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure T4ByteList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is T4ByteList) then
      System.Move(T4ByteList(Src).FList^, FList^, FCount * SizeOf(T4ByteData));
  end
  else
    Clear;
end;

// Add
//

function T4ByteList.Add(const item: T4ByteData): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1: Single);
var
  tmpList: PSingle;
begin
  Inc(FCount);
  if FCount >= FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 1];
  tmpList^ := i1;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1, i2: Single);
var
  tmpList: PSingleArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3: Single);
var
  tmpList: PSingleArray;
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 3];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3, i4: Single);
var
  tmpList: PSingleArray;
begin
  Inc(FCount, 4);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 4];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  tmpList^[3] := i4;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1: Integer);
var
  tmpList: PInteger;
begin
  Inc(FCount);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 1];
  tmpList^ := i1;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1, i2: Integer);
var
  tmpList: PIntegerArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3: Integer);
var
  tmpList: PIntegerArray;
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 3];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3, i4: Integer);
var
  tmpList: PIntegerArray;
begin
  Inc(FCount, 4);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 4];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  tmpList^[3] := i4;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1: Cardinal);
var
  tmpList: PLongWord;
begin
  Inc(FCount);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 1];
  tmpList^ := i1;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const i1, i2: Cardinal);
var
  tmpList: PLongWordArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3: Cardinal);
var
  tmpList: PLongWordArray;
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 3];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  Inc(FRevision);
end;


procedure T4ByteList.Add(const i1, i2, i3, i4: Cardinal);
var
  tmpList: PLongWordArray;
begin
  Inc(FCount, 4);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 4];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
  tmpList^[3] := i4;
  Inc(FRevision);
end;

procedure T4ByteList.Add(const AList: T4ByteList);
begin
  if Assigned(AList) and (AList.Count > 0) then
  begin
    if Count + AList.Count > Capacity then
      Capacity := Count + AList.Count;
    System.Move(AList.FList[0], FList[Count], AList.Count * SizeOf(T4ByteData));
    Inc(FCount, AList.Count);
    Inc(FRevision);
  end;
end;

// Get
//

function T4ByteList.Get(Index: Integer): T4ByteData;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure T4ByteList.Insert(Index: Integer; const Item: T4ByteData);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1],
      (FCount - Index) * SizeOf(T4ByteData));
  FList^[Index] := Item;
  Inc(FCount);
  Inc(FRevision);
end;

// Put
//

procedure T4ByteList.Put(Index: Integer; const Item: T4ByteData);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
  INc(FRevision);
end;

// SetCapacity
//

procedure T4ByteList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := P4ByteArrayList(FBaseList);
end;

// Push
//

procedure T4ByteList.Push(const Val: T4ByteData);
begin
  Add(Val);
end;

// Pop
//

function T4ByteList.Pop: T4ByteData;
const
  Zero : T4ByteData = ( Int: (Value:0) );
begin
  if FCount > 0 then
  begin
    Result := Get(FCount - 1);
    Delete(FCount - 1);
  end
  else
    Result := Zero;
end;

// ------------------
// ------------------ TLongWordList ------------------
// ------------------

// Create
//

constructor TLongWordList.Create;
begin
  FItemSize := SizeOf(LongWord);
  inherited Create;
  FGrowthDelta := cDefaultListGrowthDelta;
end;

 
//

procedure TLongWordList.Assign(Src: TPersistent);
begin
  if Assigned(Src) then
  begin
    inherited;
    if (Src is TLongWordList) then
      System.Move(TLongWordList(Src).FList^, FList^, FCount * SizeOf(LongWord));
  end
  else
    Clear;
end;

// Add (simple)
//

function TLongWordList.Add(const item: LongWord): Integer;
begin
  Result := FCount;
  if Result = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  FList^[Result] := Item;
  Inc(FCount);
end;

// AddNC (simple, no capacity check)
//

function TLongWordList.AddNC(const item: LongWord): Integer;
begin
  Result := FCount;
  FList^[Result] := Item;
  Inc(FCount);
end;

// Add (two at once)
//

procedure TLongWordList.Add(const i1, i2: LongWord);
var
  tmpList : PLongWordArray;
begin
  Inc(FCount, 2);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 2];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
end;

// Add (three at once)
//

procedure TLongWordList.Add(const i1, i2, i3: LongWord);
var
  tmpList : PLongWordArray;
begin
  Inc(FCount, 3);
  while FCount > FCapacity do
    SetCapacity(FCapacity + FGrowthDelta);
  tmpList := @FList[FCount - 3];
  tmpList^[0] := i1;
  tmpList^[1] := i2;
  tmpList^[2] := i3;
end;

// Add (list)
//

procedure TLongWordList.Add(const AList: TLongWordList);
begin
  if Assigned(AList) and (AList.Count > 0) then
  begin
    if Count + AList.Count > Capacity then
      Capacity := Count + AList.Count;
    System.Move(AList.FList[0], FList[Count], AList.Count * SizeOf(LongWord));
    Inc(FCount, AList.Count);
  end;
end;

// Get
//

function TLongWordList.Get(Index: Integer): LongWord;
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  Result := FList^[Index];
end;

// Insert
//

procedure TLongWordList.Insert(Index: Integer; const Item: LongWord);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  if FCount = FCapacity then
    SetCapacity(FCapacity + FGrowthDelta);
  if Index < FCount then
    System.Move(FList[Index], FList[Index + 1], (FCount - Index) * SizeOf(LongWord));
  FList^[Index] := Item;
  Inc(FCount);
end;

// Remove
//

procedure TLongWordList.Remove(const item: LongWord);
var
  I: Integer;
begin
  for I := 0 to Count - 1 do
  begin
    if FList^[I] = item then
    begin
      System.Move(FList[I + 1], FList[I], (FCount - 1 - I) * SizeOf(LongWord));
      Dec(FCount);
      Break;
    end;
  end;
end;

// Put
//

procedure TLongWordList.Put(Index: Integer; const Item: LongWord);
begin
{$IFOPT R+}
    Assert(Cardinal(Index) < Cardinal(FCount));
{$ENDIF}
  FList^[Index] := Item;
end;

// SetCapacity
//

procedure TLongWordList.SetCapacity(NewCapacity: Integer);
begin
  inherited;
  FList := PLongWordArray(FBaseList);
end;

// Push
//

procedure TLongWordList.Push(const Val: LongWord);
begin
  Add(Val);
end;

// Pop
//

function TLongWordList.Pop: LongWord;
begin
  if FCount > 0 then
  begin
    Result := FList^[FCount - 1];
    Delete(FCount - 1);
  end
  else
    Result := 0;
end;

// AddLongWords (pointer & n)
//

procedure TLongWordList.AddLongWords(const First: PLongWord; n: Integer);
begin
  if n < 1 then
    Exit;
  AdjustCapacityToAtLeast(Count + n);
  System.Move(First^, FList[FCount], n * SizeOf(LongWord));
  FCount := FCount + n;
end;

// AddLongWords (TLongWordList)
//

procedure TLongWordList.AddLongWords(const aList: TLongWordList);
begin
  if not Assigned(aList) then
    Exit;
  AddLongWords(@aList.List[0], aList.Count);
end;

// AddLongWords (array)
//

procedure TLongWordList.AddLongWords(const anArray: array of LongWord);
var
  n: Integer;
begin
  n := Length(anArray);
  if n > 0 then
    AddLongWords(@anArray[0], n);
end;

// LongWordSearch
//

function LongWordSearch(item: LongWord; list: PLongWordVector; Count: Integer): Integer; register;
{$IFDEF GLS_NO_ASM}
var i : integer;
begin
  result:=-1;
  for i := 0 to Count-1 do begin
    if list^[i]=item then begin
      result:=i;
      break;
    end;
  end;
end;
{$ELSE}
asm
  push edi;

  test ecx, ecx
  jz @@NotFound

  mov edi, edx;
  mov edx, ecx;
  repne scasd;
  je @@FoundIt

  @@NotFound:
  xor eax, eax
  dec eax
  jmp @@end;

  @@FoundIt:
  sub edx, ecx;
  dec edx;
  mov eax, edx;

  @@end:
  pop edi;
end;
{$ENDIF}

function TLongWordList.IndexOf(item: Integer): LongWord; register;
begin
  Result := LongWordSearch(item, FList, FCount);
end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
initialization
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------

  RegisterClasses([TAffineVectorList, TVectorList, TTexPointList, TSingleList,
                   TDoubleList, T4ByteList, TLongWordList]);

end.
