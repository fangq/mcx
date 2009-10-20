#!/bin/sh

# commands to update the document pages from homepage

lynx -dump -width 100 "http://mcx.sourceforge.net/cgi-bin/index.cgi?keywords=Download&embed=1" > Download.txt
lynx -dump "http://mcx.sourceforge.net/cgi-bin/index.cgi?keywords=Doc/Installation&embed=1" > INSTALL.txt
lynx -dump "http://mcx.sourceforge.net/cgi-bin/index.cgi?keywords=Doc/Basics&embed=1" > Get_Started.txt
lynx -dump "http://mcx.sourceforge.net/cgi-bin/index.cgi?keywords=Doc/FAQ&embed=1" > FAQ.txt

wget http://mcx.sourceforge.net/upload/mcx_diagram_paper.png -Omcx_workflow.png
