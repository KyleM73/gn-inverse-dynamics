clear all; close all; clc;

output0 = load('resultfile_output0.csv');
output1 = load('resultfile_output1.csv');
output2 = load('resultfile_output2.csv');
output3 = load('resultfile_output3.csv');
% output4 = load('resultfile_output4.csv');
target = load('resultfile_target.csv');

output = output2;

% % output0 = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_output0.csv');
% % output1 = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_output1.csv');
% % output2 = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_output2.csv');
% % output3 = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_output3.csv');
% % output4 = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_output4.csv');% % 
% % target = load('/home/jelee/GNN/graph-nets-physics/magneto-tf2/results/resultfile_target.csv');


check_idx = 1:length(output);
% check_idx = [1200:1500];

figure(1);
figure(2);
fig1 = 0; fig2 = 0;
for i=1:24
    if(mod(i-1,6)<3)
        fig_idx = 1;
        fig1 = fig1+1;
        j = fig1;
    else
        fig_idx = 2;
        fig2 = fig2+1;
        j = fig2;
    end
    figure(fig_idx);
    subplot(4,3,j);
%     plot(output0(check_idx,i));
%     plot(output1(check_idx,i));
%     plot(output2(check_idx,i));
%     plot(output3(check_idx,i));
    plot(output(check_idx,i),'Linewidth',2); hold on;
    plot(target(check_idx,i),'k--'); ylim([-6,6])

end

% legend( 'output0', 'output1', 'output2', 'output3', 'output4', 'target')
legend( 'output', 'target')  
    
