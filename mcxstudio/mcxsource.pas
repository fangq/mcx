unit mcxsource;

{$mode objfpc}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls, ValEdit;

type

  { TfmSource }

  TfmSource = class(TForm)
    btCancel: TButton;
    btOK: TButton;
    edSource: TComboBox;
    Label4: TLabel;
    Label5: TLabel;
    Panel1: TPanel;
    plEditor: TValueListEditor;
    procedure btOKClick(Sender: TObject);
    procedure edSourceEditingDone(Sender: TObject);
    procedure ChangeParams(sourceidx: integer);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
  public
     SrcParam1, SrcParam2: TStringList;
    { public declarations }
  end;

var
  fmSource: TfmSource;

implementation

const
    SrcParams : Array[0..15] of string =
     ('',
      '',
      'Half-angle(radian)=0.52360',
      'Waist radius(voxel)=5',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|y-edge vector Vy(1)=0|y-edge vector Vy(2)=40|y-edge vector Vy(3)=0',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|x-dimension=100|y-edge vector Vy(1)=0|y-edge vector Vy(2)=40|y-edge vector Vy(3)=0|y-dimension=100',
      'x-dimension Nx=40|y-dimension Ny=40|z-dimension Nz=40',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|kx+x-phase shift=3|y-edge vector Vy(1)=0|y-edge vector Vy(2)=40|y-edge vector Vy(3)=0|ky+y-phase shift=0',
      '',
      'disk radius(voxel)=10',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|y-edge vector norm=40|kx=3|ky=0|phase shift=0.33|modulation depth=0.5',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|y-edge vector norm=40|kx=3|ky=0|x phase shift=0.33|y phase shift=0',
      'Angular Gaussian variance(radian)=0.2',
      'Line vector x=20|Line vector y=20|Line vector z=0',
      'Line vector x=20|Line vector y=20|Line vector z=0',
      'x-edge vector Vx(1)=40|x-edge vector Vx(2)=0|x-edge vector Vx(3)=0|x-dimension=100|y-edge vector Vy(1)=0|y-edge vector Vy(2)=40|y-edge vector Vy(3)=0|y-dimension=100'
      );
    ParamMask : Array[0..15] of string =
     ('00000000',
      '00000000',
      '10000000',
      '10000000',
      '11101110',
      '11111111',
      '11100000',
      '11111111',
      '00000000',
      '10000000',
      '11111111',
      '11111111',
      '10000000',
      '11100000',
      '11100000',
      '11111111'
      );
{ TfmSource }

procedure TfmSource.btOKClick(Sender: TObject);
var
    i, count, srcid: integer;
begin
  try
    if(Length(edSource.Text)=0) then
         raise Exception.Create('Source type can not be empty');
    if (edSource.Items.IndexOf(edSource.Text)<0) then
         raise Exception.Create('Your specified source type is not supported!');
    srcid:=edSource.Items.IndexOf(edSource.Text);
    SrcParam1.Clear;
    SrcParam2.Clear;
    count:=1;
    for i:=1 to 4 do begin
        if(ParamMask[srcid][i]='1') then begin
            SrcParam1.Add(plEditor.Cells[1,count]);
            count:=count+1;
        end else
            SrcParam1.Add('0');

    end;
    for i:=1 to 4 do begin
        if(ParamMask[srcid][i+4]='1') then begin
            SrcParam2.Add(plEditor.Cells[1,count]);
            count:=count+1;
        end else
            SrcParam2.Add('0');
    end;
    ModalResult := mrOK;
  except
    On E : Exception do
      MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
  end;
end;

procedure TfmSource.ChangeParams(sourceidx: integer);
var
    fs: TStringList;
    i: integer;
begin
    if(sourceidx>=edSource.Items.Count) then exit;

    fs:=TStringList.Create;
    fs.StrictDelimiter := true;
    fs.Delimiter:='|';
    fs.DelimitedText:=SrcParams[sourceidx];

    plEditor.RowCount:=fs.Count+1;
    if(fs.Count > 0) then begin
     for i:=1 to fs.Count do begin
        plEditor.Cells[0,i]:=fs.Names[i-1];
        plEditor.Cells[1,i]:=fs.Values[fs.Names[i-1]];
     end;
    end;
    fs.Free;
end;

procedure TfmSource.FormCreate(Sender: TObject);
begin
    SrcParam1:=TStringList.Create;
    SrcParam2:=TStringList.Create;
end;

procedure TfmSource.FormDestroy(Sender: TObject);
begin
    SrcParam1.Free;
    SrcParam2.Free;
end;

procedure TfmSource.edSourceEditingDone(Sender: TObject);
begin
    if (edSource.Items.IndexOf(edSource.Text)<0) then begin
         MessageDlg('Input Error', 'Your specified source type is not supported!', mtError, [mbOK],0);
         exit;
    end;
    ChangeParams(edSource.Items.IndexOf(edSource.Text));
end;

initialization
  {$I mcxsource.lrs}

end.

