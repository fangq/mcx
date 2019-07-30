unit mcxstoprun;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, LResources, Forms, Controls, Graphics, Dialogs, ComCtrls,
  StdCtrls;

type

  { TfmStop }

  TfmStop = class(TForm)
    Memo1: TMemo;
    pbProgress: TProgressBar;
    tbtStop: TToolButton;
    ToolBar1: TToolBar;
    ToolButton3: TToolButton;
    ToolButton8: TToolButton;
    procedure ToolButton3Click(Sender: TObject);
  private

  public

  end;

var
  fmStop: TfmStop;

implementation

{ TfmStop }

procedure TfmStop.ToolButton3Click(Sender: TObject);
begin
  Hide;
end;

initialization
  {$I mcxstoprun.lrs}

end.

