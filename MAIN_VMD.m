
%%
clc
clear all
close all
fs=1;
Ts=1/fs;
STA=1; 
X=xlsread('dataset.xlsx');
X = X(:,end); 
L=length(X);
t=(0:L-1)*Ts;
K = 10;
% u = vmd(X,'NumIMFs',K,'PenaltyFactor',2300);
% save vmd_data u
% u = u';
THRESH_BWR = 0.25; 
BSP_ORDER = 26 ; 
u = tvf_emd(X, THRESH_BWR, BSP_ORDER); 
figure(1);
imfn=u;
n=size(imfn,1);
subplot(n+1,1,1);
plot(t,X); 
ylabel('Streamflow','fontsize',12,'fontname','宋体');

for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:));
    ylabel(['IMF' int2str(n1)]);
end
xlabel('Time\itt/day','fontsize',12,'fontname','Times New Roman');

figure 
for i = 1:K
    Hy(i,:)= abs(hilbert(u(i,:)));
    subplot(K,1,i);
    plot(t,u(i,:),'k',t,Hy(i,:),'r');
    xlabel('IMFs'); ylabel('Amplitude')
    grid; legend('IMF','Envelope');
end
title('IMFs and envelope');
set(gcf,'color','w');




figure('Name','Envelope spectrum','Color','white');
nfft=fix(L/2);
for i = 1:K
    Hy(i,:)= abs(hilbert(u(i,:)));
    p=abs(fft(Hy(i,:))); 
    p = p/length(p)*2;
    p = p(1: fix(length(p)/2));
    subplot(K,1,i);
    aa=(0:nfft-1)/nfft*fs/2;
    plot(aa,p);   
    % xlim([0.01 0.14]) 
    if i ==1
        title('Envelope spectrum'); xlabel('Frequency'); ylabel('Amplitude')
    else
        xlabel('Frequency'); ylabel('Amplitude')
    end
end
set(gcf,'color','w');


figure('Name','Spectrum diagram','Color','white');
for i = 1:K
    p=abs(fft(u(i,:)));
    subplot(K,1,i);
    bb=(0:L-1)*fs/L;
    plot(bb,p)
    xlim([0 fs/2])
    if i ==1
        title('Spectrum diagram'); xlabel('Frequency'); ylabel(['IMF' int2str(i)]);
    else
        xlabel('Frequency');  ylabel(['IMF' int2str(i)]);
    end
end
set(gcf,'color','w');