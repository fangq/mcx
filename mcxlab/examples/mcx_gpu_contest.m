%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% MCX GPU Contest
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  ask user's basic information

if(exist('savejson','file')~=2)
    error('this example requires savejson.m. please download it from https://github.com/fangq/jsonlab');
end

prompt = {sprintf(['Please answer all below questions. Your public name can be your ' ...
    'name or Internet name. Please also give your computer a name to ' ...
    'distinguish your submission from different machines.\n\n' ...
    'Your full name: (will not publish)']),'Email: (will not publish)',...
    'Your public name:','Institution:','Machine name:','Comment: (optional)'};
dlg_title = 'MCX GPU Contest Submission';
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
    hw=errordlg('missing required information');
    uiwait(hw);
    mcxuser = inputdlg(prompt,dlg_title,num_lines,mcxuser);
    if(isempty(mcxuser))
        error('user aborted submission.');
    end
end

defaultans=mcxuser;
save contest_info.mat defaultans;  % save the filled data to be load in the future to save time

%%  run the built-in benchmarks

choice = questdlg('Do you want to run the speed benchmarks? this may take a few minutes','Start Test','Yes','No','Yes');
if(isempty(choice) || strcmp(choice,'No'))
    return;
end

mcx_gpu_benchmarks

%%  generate benchmark report

mcxbenchmark.version=1;
mcxbenchmark.mcxversion='$Date::                       $';
mcxbenchmark.userinfo=struct('name',mcxuser{1},'email',mcxuser{2},'nickname',mcxuser{3},...
  'institution',mcxuser{4},'machine',mcxuser{5},'comment',mcxuser{6});

savejson(mcxbenchmark)

choice = questdlg(sprintf('Your benchmark score is %.0f. Press Yes to submit; No to cancel',mcxbenchmark.speedsum),...
  'Submission Confirmation','Yes','No','Yes');
if(isempty(choice) || strcmp(choice,'No'))
    return;
end

%% submit the benchmark report to the web portal

gpustr=sprintf('%s/' ,mcxbenchmark.gpu(:).name);
gpustr=regexprep(gpustr,'GeForce\s*','');
machinestr=sprintf('%s:/%s',mcxbenchmark.userinfo.machine,gpustr);

submitstr={'name',urlencode(mcxbenchmark.userinfo.nickname), 'time',sprintf('%ld',round(8.64e4 * (now - datenum('1970', 'yyyy')))),...
   'ver',urlencode(mcxbenchmark.mcxversion(8:end-1)), 'b1', sprintf('%.2f',speed(1)), 'b2', sprintf('%.2f',speed(2)), ...
   'b3', sprintf('%.2f',speed(3)), 'score', sprintf('%.2f',mcxbenchmark.speedsum), ...
   'computer',urlencode(machinestr), 'report',urlencode(savejson(mcxbenchmark))};

[s, status]=urlread('http://mcx.space/gpubench/gpucontest.cgi','post',submitstr);
pause(2);

if(isempty(s))
    error('submission failed');
elseif(strcmp(s,sprintf('SUCCESS\n')))
    hs=msgbox({'Submission is successful.','Please browse http://mcx.space/gpubench/ to see results.'},'Success','modal');
    uiwait(hs);
end
