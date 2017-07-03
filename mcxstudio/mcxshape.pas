unit mcxshape;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ValEdit, ExtCtrls, StdCtrls;

type

  { TfmShapeEditor }

  TfmShapeEditor = class(TForm)
    btOK: TButton;
    btCancel: TButton;
    Panel1: TPanel;
    plEditor: TValueListEditor;
    procedure btOKClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure plEditorClick(Sender: TObject);
  private
    { private declarations }
  public
    JSON: TStringList;
    constructor Create(AOwner: TComponent; PropertyList: TStringList);
    { public declarations }
  end;

var
  fmShapeEditor: TfmShapeEditor;

implementation

{ TfmShapeEditor }

procedure TfmShapeEditor.FormCreate(Sender: TObject);
begin
end;

constructor TfmShapeEditor.Create(AOwner: TComponent; PropertyList: TStringList);
var
     i: integer;
begin
     inherited Create(AOwner);

     JSON:=TStringList.Create;
     JSON.QuoteChar:=' ';
     if(PropertyList.Count =0) then exit;
     plEditor.RowCount:=PropertyList.Count+1;
     for i:=1 to PropertyList.Count do begin
         plEditor.Cells[0,i]:=PropertyList.Names[i-1];
         plEditor.Cells[1,i]:=PropertyList.Values[PropertyList.Names[i-1]];
     end;
end;

procedure TfmShapeEditor.FormDestroy(Sender: TObject);
begin
     JSON.Free;
end;

procedure TfmShapeEditor.btOKClick(Sender: TObject);
var
     i: integer;
     ss: string;
begin
     JSON.Clear;
     for i:=1 to plEditor.RowCount do begin
         if(Length(plEditor.Cells[0,i])=0) or (Length(plEditor.Cells[1,i])=0) then begin
             exit;
         end else begin
             ss:='"'+plEditor.Cells[0,i]+'":'+plEditor.Cells[1,i];
             JSON.Add(ss);
         end;
     end;
end;

procedure TfmShapeEditor.plEditorClick(Sender: TObject);
begin

end;

initialization
  {$I mcxshape.lrs}

end.

