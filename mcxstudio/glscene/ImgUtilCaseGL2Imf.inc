    GL_RGB, GL_RGB_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := GetChannel;
        ADest[n].G := GetChannel;
        ADest[n].B := GetChannel;
        ADest[n].A := 255.0;
      end;

    GL_BGR, GL_BGR_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].B := GetChannel;
        ADest[n].G := GetChannel;
        ADest[n].R := GetChannel;
        ADest[n].A := 255.0;
      end;

    GL_RGBA, GL_RGBA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := GetChannel;
        ADest[n].G := GetChannel;
        ADest[n].B := GetChannel;
        ADest[n].A := GetChannel;
      end;

    GL_BGRA, GL_BGRA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].B := GetChannel;
        ADest[n].G := GetChannel;
        ADest[n].R := GetChannel;
        ADest[n].A := GetChannel;
      end;

    GL_ALPHA, GL_ALPHA_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := 0;
        ADest[n].G := 0;
        ADest[n].B := 0;
        ADest[n].A := GetChannel;
      end;

    GL_LUMINANCE, GL_LUMINANCE_INTEGER_EXT:
      for n := 0 to AWidth*AHeight-1 do
      begin
        c0 := GetChannel;
        ADest[n].R := c0;
        ADest[n].G := c0;
        ADest[n].B := c0;
        ADest[n].A := 255.0;
      end;

    GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA_INTEGER_EXT:
      for n := 0 to AWidth*AHeight-1 do
      begin
        c0 := GetChannel;
        ADest[n].R := c0;
        ADest[n].G := c0;
        ADest[n].B := c0;
        ADest[n].A := GetChannel;
      end;

    GL_INTENSITY:
      for n := 0 to AWidth*AHeight-1 do
      begin
        c0 := GetChannel;
        ADest[n].R := c0;
        ADest[n].G := c0;
        ADest[n].B := c0;
        ADest[n].A := c0;
      end;

    GL_RED, GL_RED_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := GetChannel;
        ADest[n].G := 0;
        ADest[n].B := 0;
        ADest[n].A := 255;
      end;

    GL_GREEN, GL_GREEN_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := 0;
        ADest[n].G := GetChannel;
        ADest[n].B := 0;
        ADest[n].A := 255;
      end;

    GL_BLUE, GL_BLUE_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := 0;
        ADest[n].G := 0;
        ADest[n].B := GetChannel;
        ADest[n].A := 255;
      end;

    GL_RG, GL_RG_INTEGER:
      for n := 0 to AWidth*AHeight-1 do
      begin
        ADest[n].R := GetChannel;
        ADest[n].G := GetChannel;
        ADest[n].B := 0;
        ADest[n].A := 255;
      end;
