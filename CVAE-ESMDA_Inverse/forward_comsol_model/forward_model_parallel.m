clc
clear
close all

load coord_HHT.mat   % measurements at 80 locations
load coord_c.mat     % measurements at 20 locations
load coord_ERT.mat   % measurements at 120 locations

parpool(20);
addpath('/share/home/shixq1/COMSOL54/multiphysics/mli');
delete('/share/home/shixq1/.matlab/R2019b/matlabprefs.mat');
HHT_all=zeros(8*79*6,20);
c_all=zeros(20*1,20);
ERT_all=zeros(118*8,20);

parfor i=1:1:20
comsolPort=41036+(i-1)*1000;
for itea=1:1:60
      try
          mphstart(comsolPort); 
          break
      catch 
          comsolPort=comsolPort+1;
      end
end

cd(['./parallel',num2str(i)]);
%pause(rand(1)*60);  %make sure the loading is done one worker by one worker
%pause(3*i-2.9);
model = mphload('HHT_tracer_ERT');

%%
%import the logk
model.func('int1').discardData; %int1 is the Tag for interpolation function in comsol
model.func('int1').set('filename', './inputK.txt');
model.func('int1').importData;
%import the sn
model.func('int2').discardData; %int2 is the Tag for interpolation function in comsol
model.func('int2').set('filename', './inputSn.txt');
model.func('int2').importData;

%%
%defined the coord for observation
HHT=zeros(8,79,6); % 8 timestep 79 ports 6 pumping
c=zeros(20,1);
ERT=zeros(118,8);  % 118 ports 8 dipoles

%run steady flow model - study 3
model.study('std3').run;

% HHT simulation - study 2
for i_well=1:6
i_well_str=num2str(i_well);
model.component('comp1').variable('var1').set('i_well', i_well_str);
%run HHT model
model.study('std2').run
%read the solution at certain point, delt the location of pumping well 
coord=obs_loc_HHT(coor_HHT,i_well);
HHT(:,:,i_well) = mphinterp(model,'dl.H','edim',2,'coord',coord,'dataset','dset1','t',[5:5:40]);
end

% DNAPL steady dissolution simulation - study 4
model.study('std4').run
coord=coor_c;
c = mphinterp(model,'c','edim',2,'coord',coord,'dataset','dset4');
% force c >= 0 & c < 1
c(c<0)=0;
c(c>1)=1;

% ERT simulation - study 5
for i_elec=1:8
i_elec_str=num2str(i_elec);
model.component('comp1').variable('var1').set('i_elec', i_elec_str);
%run ERT for each dipole
model.study('std5').run
%read the solution at certain point, delt the location of pumping well 
coord=obs_loc_ERT(coor_ERT,i_elec);
ERT(:,i_elec) = mphinterp(model,'V','edim',2,'coord',coord,'dataset','dset5');
end

HHT=reshape(HHT,8*79*6,1);
HHT_all(:,i)=HHT;
c_all(:,i)=c;
ERT=reshape(ERT,118*8,1);
ERT_all(:,i)=ERT;
cd ..
end

obs_all=[HHT_all;c_all;ERT_all];
save output_obs.dat obs_all -ascii
exit;

function [coord_new] = obs_loc_ERT(coord_old,i_elec)
coord_new=coord_old;
switch i_elec
case 1
coord_new(:,26)=[];
coord_new(:,16)=[];
case 2 
coord_new(:,36)=[];
coord_new(:,6)=[];
case 3
coord_new(:,46)=[];
coord_new(:,36)=[];
case 4 
coord_new(:,56)=[];
coord_new(:,26)=[];
case 5
coord_new(:,66)=[];
coord_new(:,56)=[];
case 6
coord_new(:,76)=[];
coord_new(:,46)=[];
case 7
coord_new(:,107)=[];
coord_new(:,81)=[];
otherwise
coord_new(:,120)=[];
coord_new(:,94)=[];
end
end

function [coord_new] = obs_loc_HHT(coord_old,i_well)
coord_new=coord_old;
switch i_well
case 1
coord_new(:,11)=[];
case 2 
coord_new(:,34)=[];
case 3
coord_new(:,28)=[];
case 4 
coord_new(:,54)=[];
case 5
coord_new(:,48)=[];
otherwise
coord_new(:,71)=[];
end
end

