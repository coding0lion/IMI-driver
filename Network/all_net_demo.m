%This is an example.
cancer='BRCA';
addpath(genpath('CMI'));%
demo_cmi(cancer);

addpath(genpath('co_net'));%
demo_co(cancer);
demo_Meco(cancer);

addpath(genpath('sm_net'));
demo_sm(cancer);

addpath(genpath('ceRNA'));%
demo_ceRNA(cancer);

