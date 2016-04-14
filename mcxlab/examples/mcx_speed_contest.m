%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% MCX Speed Contest
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prompt = {sprintf(['Please answer all below questions. Your public name can be your ' ...
    'name or Internet name. Please also give your computer a name to ' ...
    'distinguish your submission from different machines.\n\n' ...
    'Your full name: (will not publish)']),'Email: (will not publish)',...
    'Your public name:','Institution:','Machine name:','Comment: (optional)'};
dlg_title = 'MCX Speed Contest Submission';
num_lines = 1;
defaultans = {'','','','','',''};

if(exist('contest_info.mat','file'))
    load contest_info.mat;
end

mcxuser = inputdlg(prompt,dlg_title,num_lines,defaultans);

if(isempty(mcxuser))
    return;
end
while(isempty(mcxuser{1}) || isempty(mcxuser{2}) || isempty(mcxuser{3}) || isempty(mcxuser{4}) || isempty(mcxuser{5}))
    hw=errordlg('missing required fields');
    uiwait(hw);
    mcxuser = inputdlg(prompt,dlg_title,num_lines,mcxuser);
    if(isempty(mcxuser))
        error('abort, user canceled submission.');
    end
end

defaultans=mcxuser;
save contest_info.mat defaultans;

choice = questdlg('Do you want to run the speed benchmarks? this may take a few minutes','Start Test','Yes','No','Yes');
if(isempty(choice) || strcmp(choice,'No'))
    return;
end

mcx_speed_benchmarks

mcxbenchmark.mcxversion='$Rev::       $';
mcxbenchmark.userinfo=struct('name',mcxuser{1},'email',mcxuser{2},'nickname',mcxuser{3},...
  'institution',mcxuser{4},'machine',mcxuser{5},'comment',mcxuser{6});

savejson(mcxbenchmark)

choice = questdlg('Your benchmark results are printed. Press OK to submit','Submission Confirmation','Yes','No','Yes');
if(isempty(choice) || strcmp(choice,'No'))
    return;
end

[s, status]=urlread('http://mcx.space/contest/speedcontest.cgi','POST',['result=' urlencode(savejson(mcxbenchmark))]);

if(isempty(s))
    error('submission failed');
end