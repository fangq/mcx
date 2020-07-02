    GL_RGB:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R);
        SetChannel(ASource[n].G);
        SetChannel(ASource[n].B);
      end;

    GL_RGB_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R);
        SetChannelI(ASource[n].G);
        SetChannelI(ASource[n].B);
      end;

    GL_BGR:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].B);
        SetChannel(ASource[n].G);
        SetChannel(ASource[n].R);
      end;

    GL_BGR_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].B);
        SetChannelI(ASource[n].G);
        SetChannelI(ASource[n].R);
      end;

    GL_RGBA:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R);
        SetChannel(ASource[n].G);
        SetChannel(ASource[n].B);
        SetChannel(ASource[n].A);
      end;

    GL_RGBA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R);
        SetChannelI(ASource[n].G);
        SetChannelI(ASource[n].B);
        SetChannelI(ASource[n].A);
      end;

    GL_BGRA:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].B);
        SetChannel(ASource[n].G);
        SetChannel(ASource[n].R);
        SetChannel(ASource[n].A);
      end;

    GL_BGRA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].B);
        SetChannelI(ASource[n].G);
        SetChannelI(ASource[n].R);
        SetChannelI(ASource[n].A);
      end;

    GL_ALPHA:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].A);
      end;

    GL_ALPHA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].A);
      end;

    GL_LUMINANCE:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R + ASource[n].G + ASource[n].B / 3.0);
      end;

    GL_LUMINANCE_INTEGER_EXT:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R + ASource[n].G + ASource[n].B / 3.0);
      end;

    GL_LUMINANCE_ALPHA:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R + ASource[n].G + ASource[n].B / 3.0);
        SetChannel(ASource[n].A);
      end;

    GL_LUMINANCE_ALPHA_INTEGER_EXT:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R + ASource[n].G + ASource[n].B / 3.0);
        SetChannelI(ASource[n].A);
      end;

    GL_INTENSITY:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R + ASource[n].G + ASource[n].B / 3.0);
      end;

    GL_RED:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R);
      end;

    GL_RED_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R);
      end;

    GL_GREEN:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].G);
      end;

    GL_GREEN_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].G);
      end;

    GL_BLUE:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].B);
      end;

    GL_BLUE_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].B);
      end;

    GL_RG:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannel(ASource[n].R);
        SetChannel(ASource[n].G);
      end;

    GL_RG_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        SetChannelI(ASource[n].R);
        SetChannelI(ASource[n].G);
      end;
